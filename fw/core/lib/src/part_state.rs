// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Typed, property-routed partition state accessors.
//!
//! This module is the canonical core-side view of per-partition state.
//! Every public function here is a thin, typed wrapper over the generic
//! property API on [`HsmPartitionManager`]
//! ([`part_prop_get_*`](HsmPartitionManager::part_prop_get_u8) /
//! [`part_prop_set_*`](HsmPartitionManager::part_prop_set_u8) /
//! [`part_prop_clear`](HsmPartitionManager::part_prop_clear)) and
//! mirrors the legacy typed `part_*` methods on `HsmPartitionManager`
//! one-for-one in name and behaviour.
//!
//! # Position in the stack
//!
//! ```text
//!     core call site
//!         │
//!         ▼
//!   crate::part_state::part_*   ← you are here
//!         │
//!         ▼
//!   HsmPartitionManager::part_prop_get_* / set_* / clear   (PAL trait)
//!         │
//!         ▼
//!     PAL impl backing storage (flat region + presence bitmap + undo log)
//! ```
//!
//! Two views back the same storage in the PAL: the typed `part_*`
//! methods on [`HsmPartitionManager`] and the generic `part_prop_*`
//! methods.  Readers of one observe writes from the other.  Core
//! logic uses *only* this module so that:
//!
//! - The full set of partition state read or written by the firmware
//!   is enumerable from a single place (the [`HsmPartPropId`] catalogue
//!   in the PAL traits crate).  New partition state is added there
//!   first; new accessors get added here second; existing call sites
//!   do not move.
//! - State-rollback / undo-log emission can hang off the property
//!   layer in the PAL without core ever caring.  The PAL is free to
//!   intercept `part_prop_set_*` / `part_prop_clear` and journal an
//!   inverse operation; core call sites stay shape-stable.
//! - Future PAL refactors that retire the typed `part_*` methods (or
//!   rearrange the storage layout) touch zero core call sites — only
//!   the bodies of the wrappers in this file move.
//!
//! # Conventions
//!
//! ## Arguments
//!
//! Every accessor takes `pal: &P` where `P: HsmPartitionManager` and
//! `io: &P::Io`, in that order.  The PAL resolves the target
//! partition from private IO metadata; this module never names a partition
//! id directly.
//!
//! ## Byte-valued properties use [`HsmDmaBuf`]
//!
//! All byte-valued accessors take or return `&HsmDmaBuf` instead of the
//! legacy `Option<&mut [u8]>` query/copy pattern.  This keeps data
//! flowing through PAL crypto primitives zero-copy and avoids the
//! two-pass length/copy idiom at every caller.
//!
//! Setters accept any `&HsmDmaBuf` whose length matches the property's
//! [`HsmPartPropKind`]; the PAL impl validates length and returns
//! [`HsmError::InvalidArg`] on a mismatch.  Getters return a borrow
//! tied to the PAL impl's storage; callers must not assume the
//! lifetime extends past the next PAL mutation against the same
//! partition.
//!
//! ## Presence semantics
//!
//! Property getters return [`HsmError::PartPropNotFound`] when the
//! addressed slot has not yet been written (or was last cleared).
//! Wrappers here propagate that error unchanged; callers that want
//! to treat absence as a non-error condition must match on
//! [`HsmError::PartPropNotFound`] explicitly.  **No `Option` wrapper
//! is layered on top — the error variant *is* the presence signal.**
//!
//! For properties whose PAL meta marks them as
//! [`HsmPartPropDefault::RequiredPresent`] the error is unreachable in
//! well-formed builds (the PAL guarantees the slot is initialised on
//! partition allocation) and callers may treat the result as
//! infallible w.r.t. presence — PAL transport / storage errors still
//! propagate.  The doc comment on each accessor calls out which
//! presence class applies.
//!
//! ## Sensitivity
//!
//! Properties whose PAL meta marks them as `sensitive` (PSKs,
//! credentials, nonce, sealed / masked BK_BOOT,
//! firmware seed) are zeroised by the PAL when overwritten or
//! cleared.  Callers in this module never log the value of those
//! properties; downstream code that takes a `&HsmDmaBuf` of sensitive
//! material is expected to honour the same rule.
//!
//! ## Type-narrowed views over the property kind
//!
//! Two property categories store wider scalars than their typed view
//! exposes:
//!
//! - **Lifecycle state** ([`HsmPartPropId::PART_STATE`]) is stored as `U8` but
//!   exposed here as [`HsmPartState`].  The mapping is the
//!   `#[repr(u8)]` discriminant on the enum.  An unknown byte
//!   coming back from storage is treated as PAL corruption and
//!   surfaces as [`HsmError::InternalError`] from [`part_state`].
//! - **Vault key references** (the `*_KEY_ID` ids) are stored as
//!   `U16` and exposed here as [`HsmKeyId`] (a `u16` newtype) via the
//!   `key_id_get` / `key_id_set` helpers below.  Conversion is
//!   loss-less because both representations are `u16`.
//!
//! ## Naming
//!
//! Every accessor uses the `part_` prefix so that, when re-exported
//! through `super::*` into the rest of the crate, the names stay
//! self-describing without a `part_state::` qualifier at the call site.
//! Setters are `part_set_<name>`; clear operations are
//! `part_clear_<name>`.

use zeroize::Zeroize;

use super::*;

// ─── Identity, lifecycle, and platform state ──────────────────────────────
//
// Properties that name *what* this partition is and *where* it is in
// its lifecycle.  These are the first things every DDI handler reads
// to decide whether to accept a request.

/// Lifecycle state of the calling partition.
///
/// Wraps [`HsmPartPropId::PART_STATE`] (`U8` decoded into [`HsmPartState`]).  The
/// slot is `RequiredPresent`, so this never returns
/// [`HsmError::PartPropNotFound`].  A stored byte that does not name
/// a known [`HsmPartState`] is treated as PAL-side corruption and
/// surfaces as [`HsmError::InternalError`].
pub fn part_state<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<HsmPartState> {
    let raw = pal.part_prop_get_u8(io, HsmPartPropId::PART_STATE)?;
    HsmPartState::from_u8(raw).ok_or(HsmError::InternalError)
}

/// Presence probe for an `AbsentUntilSet` property slot.
///
/// Maps a getter result to a presence flag: `Ok(true)` when the getter
/// succeeds, `Ok(false)` **only** for [`HsmError::PartPropNotFound`]
/// (the slot is genuinely absent), and propagates every other read error
/// so a real fault (e.g. internal corruption / IO failure) is never
/// silently misread as "absent".  Mirrors the pattern used by
/// [`part_is_provisioned`].
#[inline]
pub(crate) fn prop_present<T>(result: HsmResult<T>) -> HsmResult<bool> {
    match result {
        Ok(_) => Ok(true),
        Err(HsmError::PartPropNotFound) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Set the partition lifecycle state.
///
/// Wraps [`HsmPartPropId::PART_STATE`].  **Transition policy lives here, in the
/// upper layer:** the generic PAL `part_prop_set_u8(STATE)` is a raw
/// mechanism (it writes the discriminant directly), so the undo log can
/// restore any prior state through it.  This wrapper accepts only the two
/// caller-facing transitions — `Enabled → Initializing` (`PartInit`,
/// which additionally requires the four write-once provisioning fields —
/// PTA key, UPS key, policy hash, POTA thumbprint — to be present) and
/// `Initializing → Initialized` (`PartFinal`).  A same-state write is a
/// no-op; any other pair is rejected with [`HsmError::InvalidArg`].
pub fn part_set_state<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    s: HsmPartState,
) -> HsmResult<()> {
    let current = part_state(pal, io)?;
    match (current, s) {
        (HsmPartState::Enabled, HsmPartState::Initializing) => {
            let provisioned = prop_present(part_pta_key_id(pal, io))?
                && prop_present(part_ups_key_id(pal, io))?
                && prop_present(part_policy_hash(pal, io))?
                && prop_present(part_pota_thumbprint(pal, io))?;
            if !provisioned {
                return Err(HsmError::InvalidArg);
            }
        }
        (HsmPartState::Initializing, HsmPartState::Initialized) => {}
        (cur, tgt) if cur == tgt => return Ok(()),
        _ => return Err(HsmError::InvalidArg),
    }
    pal.part_prop_set_u8(io, HsmPartPropId::PART_STATE, s as u8)
}

/// [`HsmPartPropId`] backing [`part_state`] / [`part_set_state`].
#[inline]
pub const fn part_state_prop_id() -> HsmPartPropId {
    HsmPartPropId::PART_STATE
}

/// Quarantine the partition by forcing it to [`HsmPartState::Faulted`].
///
/// Called by the dispatcher when an undo walk reports a poisoned
/// (consistency-critical) failure: the partition's in-memory state is
/// incoherent, so it is fenced off — the `partition_enabled` gate then drops
/// all further host IO, and neither `PartInit`/`PartFinal` nor an admin
/// `PfnEnable` can transition out of `Faulted`.  Only a free/realloc cycle
/// (or reboot) clears it.  Uses the raw STATE setter because this is an
/// out-of-band fault transition, not a normal lifecycle step.
pub fn part_set_faulted<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<()> {
    pal.part_prop_set_u8(io, HsmPartPropId::PART_STATE, HsmPartState::Faulted as u8)
}

/// Monotonic partition generation counter.
///
/// Wraps [`HsmPartPropId::PART_GEN`] (`U32`, `RequiredPresent`, **read-only**).
/// Incremented by the PAL on every allocate/free cycle; used by
/// lifetime guards (e.g. session handles) to detect partition reuse
/// across a free/realloc and refuse to operate on a stale generation.
pub fn part_gen<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<u32> {
    pal.part_prop_get_u32(io, HsmPartPropId::PART_GEN)
}

/// [`HsmPartPropId`] backing [`part_gen`].
#[inline]
pub const fn part_gen_prop_id() -> HsmPartPropId {
    HsmPartPropId::PART_GEN
}

/// Security version number bound into this partition's derivation
/// lineage.
///
/// Current firmware SVN — the manufacturer-seed (BKS1) selector.
///
/// Device-global (not per-partition); served by
/// [`HsmSeedStore::mfgr_svn`](tsra_fw_hsm_pal_traits::HsmSeedStore::mfgr_svn).
/// Used as a tweak input to partition-bound key derivations so that
/// material derived under firmware version N is not reachable from
/// firmware version N-1.
pub fn part_mfgr_svn(pal: &impl HsmSeedStore) -> u64 {
    pal.mfgr_svn()
}

/// Number of host-allocated SQ/CQ resource pairs.
///
/// Wraps [`HsmPartPropId::PART_RES_COUNT`] (`U8`, `RequiredPresent`,
/// read-only).  Read-only from the caller's perspective; set by the
/// PAL at partition allocation time from the resource grant.
pub fn part_res_count<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<u8> {
    pal.part_prop_get_u8(io, HsmPartPropId::PART_RES_COUNT)
}

/// [`HsmPartPropId`] backing [`part_res_count`].
#[inline]
pub const fn part_res_count_prop_id() -> HsmPartPropId {
    HsmPartPropId::PART_RES_COUNT
}

/// Opaque 16-byte partition identity blob.
///
/// Wraps [`HsmPartPropId::PART_ID`] (`FixedBytes { len: 16 }`,
/// `AbsentUntilSet`, **read-only**).  Returns
/// [`HsmError::PartPropNotFound`] before the PAL has populated it
/// for this partition.  The bytes are opaque to core; only the host
/// management layer interprets them.
pub fn part_id<'a, P: HsmPartitionManager>(pal: &'a P, io: &P::Io) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::PART_ID)
}

/// [`HsmPartPropId`] backing [`part_id`].
#[inline]
pub const fn part_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::PART_ID
}

/// Unique Device Secret (32 B).  **Sensitive.**
///
/// Device-global (not per-partition); served by
/// [`HsmSeedStore::uds`](tsra_fw_hsm_pal_traits::HsmSeedStore::uds).
/// The UDS is the device root secret used as the KBKDF key for
/// partition-bound derivations (the partition-distinguishing inputs are
/// the KBKDF context); the returned borrow must not be logged or copied
/// outside crypto primitives.
pub fn part_uds(pal: &impl HsmSeedStore) -> &[u8] {
    pal.uds()
}

/// Firmware-supplied secret seed (48 B).  **Sensitive.**
///
/// Device-global (not per-partition); served by
/// [`HsmSeedStore::fw_seed`](tsra_fw_hsm_pal_traits::HsmSeedStore::fw_seed).
/// Used as the KBKDF key in boot-key (`BKx`) derivations.
pub fn part_fw_seed(pal: &impl HsmSeedStore) -> &[u8] {
    pal.fw_seed()
}

// ─── Vault references ─────────────────────────────────────────────────────
//
// All vault-ref properties are stored in the property table as `U16`
// (matching today's [`HsmKeyId`] width).  The typed core view here
// re-exposes them as [`HsmKeyId`] (a `u16` newtype) for call-site
// ergonomics; conversion is loss-less.  If the catalogue ever grows
// to a wider scalar kind, the `key_id_get` helper is the single
// point at which narrowing happens, and an out-of-range stored value
// will surface as [`HsmError::InternalError`].

/// Read a vault key reference as an [`HsmKeyId`] (`u16`).
///
/// Returns [`HsmError::PartPropNotFound`] if the slot is absent;
/// callers handle the absence/error split per the presence
/// semantics in the module docs.
#[inline]
fn key_id_get<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    id: HsmPartPropId,
) -> HsmResult<HsmKeyId> {
    let raw = pal.part_prop_get_u16(io, id)?;
    Ok(HsmKeyId::from(raw))
}

/// Write a vault key reference (`HsmKeyId` is a `u16`).
#[inline]
fn key_id_set<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    id: HsmPartPropId,
    key_id: HsmKeyId,
) -> HsmResult<()> {
    pal.part_prop_set_u16(io, id, u16::from(key_id))
}

/// Enforce write-once for an `AbsentUntilSet` key-id property.
///
/// Write-once is **upper-layer policy**: the generic PAL
/// `part_prop_set_u16` is a raw mechanism (so the undo log can restore a
/// prior id), so the typed setters that must be write-once gate on the
/// current presence here.  Returns `err` if the slot is already
/// populated, and propagates any non-absent read error.
fn ensure_key_id_unset<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    id: HsmPartPropId,
    err: HsmError,
) -> HsmResult<()> {
    if prop_present(key_id_get(pal, io, id))? {
        return Err(err);
    }
    Ok(())
}

/// Vault id of the partition identity (ECC-P384) key.
///
/// Wraps [`HsmPartPropId::PART_ID_ECC_KEY_ID`] (`U16 → HsmKeyId`,
/// `AbsentUntilSet`, **read-only**).
pub fn part_id_key_id<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<HsmKeyId> {
    key_id_get(pal, io, HsmPartPropId::PART_ID_ECC_KEY_ID)
}

/// [`HsmPartPropId`] backing [`part_id_key_id`].
#[inline]
pub const fn part_id_key_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::PART_ID_ECC_KEY_ID
}

/// Vault id of the partition masking key.
///
/// Wraps [`HsmPartPropId::MK_KEY_ID`] (`U16 → HsmKeyId`,
/// `AbsentUntilSet`).
pub fn part_mk_key_id<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<HsmKeyId> {
    key_id_get(pal, io, HsmPartPropId::MK_KEY_ID)
}

/// Set the partition masking key id.
pub fn part_set_mk_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    key_id: HsmKeyId,
) -> HsmResult<()> {
    key_id_set(pal, io, HsmPartPropId::MK_KEY_ID, key_id)
}

/// [`HsmPartPropId`] backing [`part_mk_key_id`] / [`part_set_mk_key_id`].
#[inline]
pub const fn part_mk_key_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::MK_KEY_ID
}

/// Whether the partition has been fully provisioned.
///
/// A partition is provisioned once its masking key (MK) has been
/// imported into the vault and recorded via
/// [`part_set_mk_key_id`].  This is the gate
/// `EstablishCredential` uses to prevent double-provisioning, and
/// `OpenSession` uses to require provisioning before a session can
/// be opened.  Derived from [`HsmPartPropId::MK_KEY_ID`] presence
/// (`AbsentUntilSet`).
pub fn part_is_provisioned<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<bool> {
    match part_mk_key_id(pal, io) {
        Ok(_) => Ok(true),
        Err(HsmError::PartPropNotFound) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Vault id of the partition Unique Machine Secret derived key.
///
/// Wraps [`HsmPartPropId::UPS_KEY_ID`] (`U16 → HsmKeyId`,
/// `AbsentUntilSet`).
pub fn part_ups_key_id<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<HsmKeyId> {
    key_id_get(pal, io, HsmPartPropId::UPS_KEY_ID)
}

/// Set the partition UPS-derived key id.
pub fn part_set_ups_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    key_id: HsmKeyId,
) -> HsmResult<()> {
    ensure_key_id_unset(
        pal,
        io,
        HsmPartPropId::UPS_KEY_ID,
        HsmError::UpsKeyAlreadySet,
    )?;
    key_id_set(pal, io, HsmPartPropId::UPS_KEY_ID, key_id)
}

/// [`HsmPartPropId`] backing [`part_ups_key_id`] / [`part_set_ups_key_id`].
#[inline]
pub const fn part_ups_key_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::UPS_KEY_ID
}

/// Vault id of the Partition Trust Anchor (PTA) key.
///
/// Wraps [`HsmPartPropId::PTA_KEY_ID`] (`U16 → HsmKeyId`,
/// `AbsentUntilSet`).  Bound by the TBOR `PartInit` handler.
pub fn part_pta_key_id<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<HsmKeyId> {
    key_id_get(pal, io, HsmPartPropId::PTA_KEY_ID)
}

/// Set the PTA key id.
pub fn part_set_pta_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    key_id: HsmKeyId,
) -> HsmResult<()> {
    ensure_key_id_unset(
        pal,
        io,
        HsmPartPropId::PTA_KEY_ID,
        HsmError::PtaKeyAlreadySet,
    )?;
    key_id_set(pal, io, HsmPartPropId::PTA_KEY_ID, key_id)
}

/// [`HsmPartPropId`] backing [`part_pta_key_id`] / [`part_set_pta_key_id`].
#[inline]
pub const fn part_pta_key_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::PTA_KEY_ID
}

/// Vault id of the partition-local key masking key (`PartLocalMK`).
///
/// Wraps [`HsmPartPropId::LOCAL_MK_KEY_ID`] (`U16 → HsmKeyId`,
/// `AbsentUntilSet`).  Bound by the TBOR `PartFinal` handler.
pub fn part_local_mk_key_id<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<HsmKeyId> {
    key_id_get(pal, io, HsmPartPropId::LOCAL_MK_KEY_ID)
}

/// Set the partition-local masking-key id.
pub fn part_set_local_mk_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    key_id: HsmKeyId,
) -> HsmResult<()> {
    key_id_set(pal, io, HsmPartPropId::LOCAL_MK_KEY_ID, key_id)
}

/// [`HsmPartPropId`] backing [`part_local_mk_key_id`] /
/// [`part_set_local_mk_key_id`].
#[inline]
pub const fn part_local_mk_key_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::LOCAL_MK_KEY_ID
}

/// Vault id of the partition's ephemeral key masking key
/// (`EphemeralMK`).
///
/// Wraps [`HsmPartPropId::EPHEMERAL_MK_KEY_ID`] (`U16 → HsmKeyId`,
/// `AbsentUntilSet`).  Bound by the TBOR `PartFinal` handler.
pub fn part_ephemeral_mk_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
) -> HsmResult<HsmKeyId> {
    key_id_get(pal, io, HsmPartPropId::EPHEMERAL_MK_KEY_ID)
}

/// Set the partition ephemeral masking-key id.
pub fn part_set_ephemeral_mk_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    key_id: HsmKeyId,
) -> HsmResult<()> {
    key_id_set(pal, io, HsmPartPropId::EPHEMERAL_MK_KEY_ID, key_id)
}

/// [`HsmPartPropId`] backing [`part_ephemeral_mk_key_id`] /
/// [`part_set_ephemeral_mk_key_id`].
#[inline]
pub const fn part_ephemeral_mk_key_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::EPHEMERAL_MK_KEY_ID
}

/// Vault id of the security-domain key masking key (`SDMK`).
///
/// Wraps [`HsmPartPropId::SD_MK_KEY_ID`] (`U16 → HsmKeyId`,
/// `AbsentUntilSet`).  Bound by the TBOR `SdCreateRemoteBackup` handler;
/// resolves the [`SecurityDomain`](tsra_fw_hsm_pal_traits::HsmKeyScope::SecurityDomain)
/// scope to its live masking key.
pub fn part_sd_mk_key_id<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<HsmKeyId> {
    key_id_get(pal, io, HsmPartPropId::SD_MK_KEY_ID)
}

/// Set the security-domain masking-key id.
pub fn part_set_sd_mk_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    key_id: HsmKeyId,
) -> HsmResult<()> {
    key_id_set(pal, io, HsmPartPropId::SD_MK_KEY_ID, key_id)
}

/// [`HsmPartPropId`] backing [`part_sd_mk_key_id`] /
/// [`part_set_sd_mk_key_id`].
#[inline]
pub const fn part_sd_mk_key_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::SD_MK_KEY_ID
}

/// Vault id of the partition's unwrapping key.
///
/// Wraps [`HsmPartPropId::RSA_UNWRAPPING_KEY_ID`] (`U16 → HsmKeyId`,
/// `AbsentUntilSet`).
pub fn part_unwrapping_key_id<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<HsmKeyId> {
    key_id_get(pal, io, HsmPartPropId::RSA_UNWRAPPING_KEY_ID)
}

/// Set the partition's unwrapping key id.
///
/// Used by `EstablishCredential` to re-import the host-persisted key
/// after a live migration.
pub fn part_set_unwrapping_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    key_id: HsmKeyId,
) -> HsmResult<()> {
    key_id_set(pal, io, HsmPartPropId::RSA_UNWRAPPING_KEY_ID, key_id)
}

/// [`HsmPartPropId`] backing [`part_unwrapping_key_id`].
#[inline]
pub const fn part_unwrapping_key_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::RSA_UNWRAPPING_KEY_ID
}

/// Vault id of the long-lived session-encryption (ECDH) key.
///
/// Wraps [`HsmPartPropId::SESSION_ENC_KEY_ID`] (`U16 → HsmKeyId`,
/// `AbsentUntilSet`).
pub fn part_session_enc_key_id<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<HsmKeyId> {
    key_id_get(pal, io, HsmPartPropId::SESSION_ENC_KEY_ID)
}

/// Set the session-encryption key id.
pub fn part_set_session_enc_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    key_id: HsmKeyId,
) -> HsmResult<()> {
    key_id_set(pal, io, HsmPartPropId::SESSION_ENC_KEY_ID, key_id)
}

/// [`HsmPartPropId`] backing [`part_session_enc_key_id`] /
/// [`part_set_session_enc_key_id`].
#[inline]
pub const fn part_session_enc_key_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::SESSION_ENC_KEY_ID
}

/// Vault id of the one-shot establish-credential RSA-OAEP key.
///
/// Wraps [`HsmPartPropId::ESTABLISH_CRED_KEY_ID`] (`U16 → HsmKeyId`,
/// `AbsentUntilSet`).  This is a one-shot key: the establish-cred
/// handler clears the slot with [`part_clear_establish_cred_key`] as
/// soon as the credential has been unwrapped, so a successful return
/// here means the key is still consumable.
pub fn part_establish_cred_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
) -> HsmResult<HsmKeyId> {
    key_id_get(pal, io, HsmPartPropId::ESTABLISH_CRED_KEY_ID)
}

/// Set the establish-credential key id.
pub fn part_set_establish_cred_key_id<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    key_id: HsmKeyId,
) -> HsmResult<()> {
    key_id_set(pal, io, HsmPartPropId::ESTABLISH_CRED_KEY_ID, key_id)
}

/// Clear the establish-credential key id (one-shot consumed).
///
/// Wraps [`HsmPartitionManager::part_prop_clear`] for
/// [`HsmPartPropId::ESTABLISH_CRED_KEY_ID`].  Idempotent on an already
/// absent slot (returns `Ok(())`).  After this call,
/// [`part_establish_cred_key_id`] returns
/// [`HsmError::PartPropNotFound`].
pub fn part_clear_establish_cred_key<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<()> {
    pal.part_prop_clear(io, HsmPartPropId::ESTABLISH_CRED_KEY_ID)
}

/// [`HsmPartPropId`] backing [`part_establish_cred_key_id`] /
/// [`part_set_establish_cred_key_id`] /
/// [`part_clear_establish_cred_key`].
#[inline]
pub const fn part_establish_cred_key_id_prop_id() -> HsmPartPropId {
    HsmPartPropId::ESTABLISH_CRED_KEY_ID
}

// ─── Retained authentication policy ──────────────────────────────────────

/// Load the partition-wide CO/CU PSK policy, rejecting corrupt storage.
pub fn part_psk_policy<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
) -> HsmResult<crate::psk_policy::PskPolicy> {
    let bytes = pal.part_prop_get_bytes(io, HsmPartPropId::PSK_POLICY)?;
    let snapshot = PskPolicySnapshot::from_bytes(bytes)?;
    Ok(crate::psk_policy::PskPolicy::from_snapshot(snapshot))
}

/// Persist a complete PSK policy synchronously, without command undo logging.
///
/// Call under the partition lock. Do not retain the loaded copy across an await
/// or journal this write: command failure must not undo authentication penalties.
/// A storage/allocation error must be propagated rather than admitting a session.
pub fn part_set_psk_policy<P: HsmPartitionManager + HsmAlloc>(
    pal: &P,
    io: &P::Io,
    policy: &crate::psk_policy::PskPolicy,
) -> HsmResult<()> {
    let snapshot = policy.snapshot()?;
    let bytes = pal.dma_alloc(io, PSK_POLICY_LEN)?;
    bytes.copy_from_slice(snapshot.as_bytes());
    pal.part_prop_set_bytes(io, HsmPartPropId::PSK_POLICY, bytes)
}

// ─── Caller-presented secrets ─────────────────────────────────────────────
//
// Material that the host or the management plane hands the partition
// to authenticate or key sessions.  All sensitive; the PAL zeroises
// previous values on overwrite or clear.

/// Read a pre-shared key.
///
/// `psk_id` selects between PSK slots, matching the legacy
/// [`HsmPartitionManager::part_psk`] enumeration:
///
/// | `psk_id` | Slot               | Property                 |
/// |----------|--------------------|--------------------------|
/// | `0`      | Crypto Officer PSK | [`HsmPartPropId::PSK_CO`]   |
/// | `1`      | Crypto User PSK    | [`HsmPartPropId::PSK_CU`]   |
///
/// Any other value returns [`HsmError::InvalidPskId`].  Both slots are
/// `RequiredPresent` (default-baked at allocation time from
/// [`DEFAULT_PSK_CO`] / [`DEFAULT_PSK_CU`]) so a successful call
/// always yields exactly [`PSK_LEN`] bytes.
pub fn part_psk<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
    psk_id: u8,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, part_psk_prop_id(psk_id)?)
}

/// Write a pre-shared key.
///
/// `data` must be exactly [`PSK_LEN`] bytes; see [`part_psk`] for the
/// `psk_id → slot` mapping.  Rotating from the default PSK is required
/// before exposing the partition to untrusted traffic.
pub fn part_set_psk<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    psk_id: u8,
    data: &HsmDmaBuf,
) -> HsmResult<()> {
    pal.part_prop_set_bytes(io, part_psk_prop_id(psk_id)?, data)
}

/// Translate the `0/1` PSK selector into the backing
/// [`HsmPartPropId`].  Returns [`HsmError::InvalidPskId`] for any other
/// selector.
///
/// Exposed so upper-layer undo-log emitters can name the slot they
/// are about to mutate without re-encoding the selector mapping.
#[inline]
pub fn part_psk_prop_id(psk_id: u8) -> HsmResult<HsmPartPropId> {
    match psk_id {
        0 => Ok(HsmPartPropId::PSK_CO),
        1 => Ok(HsmPartPropId::PSK_CU),
        _ => Err(HsmError::InvalidPskId),
    }
}

/// Caller-presented credential blob (32 B).  **Sensitive.**
///
/// Wraps [`HsmPartPropId::CREDENTIAL`] (`FixedBytes { len: 32 }`,
/// `AbsentUntilSet`, sensitive).  Returns
/// [`HsmError::PartPropNotFound`] before [`part_set_credential`] has
/// been called.  Verified by the upper layer via constant-time
/// compare against the returned bytes — callers must not branch on
/// the contents themselves.
pub fn part_credential<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::CREDENTIAL)
}

/// Write the caller-presented credential blob.  **Sensitive.**
///
/// `data` must be exactly 32 bytes (`id ‖ pin`, 16 + 16).  Write-once
/// per credential lifecycle: re-setting without an intervening
/// [`HsmPartPropId::CREDENTIAL`] clear returns
/// [`HsmError::VaultAppLimitReached`]; an all-zero `id` or `pin` half
/// is rejected with [`HsmError::InvalidAppCredentials`].
pub fn part_set_credential<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    data: &HsmDmaBuf,
) -> HsmResult<()> {
    pal.part_prop_set_bytes(io, HsmPartPropId::CREDENTIAL, data)
}

/// Constant-time-compare the supplied `id` and `pin` (16 B each)
/// against the partition's stored credential.
///
/// Returns `Ok(())` on match.  Returns:
///
/// - [`HsmError::InvalidArg`] if `id` or `pin` is not exactly 16 B
///   (caller bug — the wire shape is fixed).
/// - [`HsmError::InvalidAppCredentials`] when the credential has not
///   been set yet, or when either half differs from the stored
///   value.  The comparison runs over the full 32-byte width so a
///   timing side-channel cannot distinguish "wrong id" from "wrong
///   pin".
/// - [`HsmError::InternalError`] if the stored credential blob has
///   a length other than 32 bytes — that signals a property
///   catalogue / PAL storage corruption bug, not an authentication
///   failure, and must not be masked as one.
pub fn part_verify_credential<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    id: &[u8],
    pin: &[u8],
) -> HsmResult<()> {
    if id.len() != 16 || pin.len() != 16 {
        return Err(HsmError::InvalidArg);
    }
    let stored = match part_credential(pal, io) {
        Ok(buf) => buf,
        Err(HsmError::PartPropNotFound) => return Err(HsmError::InvalidAppCredentials),
        Err(e) => return Err(e),
    };
    if stored.len() != 32 {
        return Err(HsmError::InternalError);
    }
    let mut diff = 0u8;
    for i in 0..16 {
        diff |= stored[i] ^ id[i];
        diff |= stored[16 + i] ^ pin[i];
    }
    if diff == 0 {
        Ok(())
    } else {
        Err(HsmError::InvalidAppCredentials)
    }
}

/// [`HsmPartPropId`] backing [`part_credential`] / [`part_set_credential`].
#[inline]
pub const fn part_credential_prop_id() -> HsmPartPropId {
    HsmPartPropId::CREDENTIAL
}

/// Wire length of [`HsmPartPropId::NONCE`] (matches the catalogue
/// `FixedBytes { len: 32 }`).
pub const NONCE_LEN: usize = 32;

/// 32-byte partition nonce, refreshed per credential / session event.
/// **Sensitive.**
///
/// Wraps [`HsmPartPropId::NONCE`] (`FixedBytes { len: 32 }`,
/// `RequiredPresent`, sensitive).  Always present; the PAL
/// initialises it on partition allocation.  Refresh by writing fresh
/// PAL-RNG bytes via [`part_set_nonce`].
pub fn part_nonce<'a, P: HsmPartitionManager>(pal: &'a P, io: &P::Io) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::NONCE)
}

/// Overwrite the partition nonce with the supplied 32-byte buffer.
///
/// Wraps [`HsmPartPropId::NONCE`] setter.  Callers must source the
/// bytes from the PAL RNG (e.g. [`HsmRng::rng_fill_bytes`]) so the
/// new nonce is unpredictable to the host.
pub fn part_set_nonce<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    nonce: &HsmDmaBuf,
) -> HsmResult<()> {
    pal.part_prop_set_bytes(io, HsmPartPropId::NONCE, nonce)
}

/// [`HsmPartPropId`] backing [`part_nonce`].
#[inline]
pub const fn part_nonce_prop_id() -> HsmPartPropId {
    HsmPartPropId::NONCE
}

/// Constant-time-compare `nonce` against the partition's current
/// stored nonce.
///
/// Returns `Ok(())` on match, [`HsmError::NonceMismatch`] otherwise.
/// The comparison runs over the full 32-byte width regardless of
/// where bytes first diverge, so a timing side-channel cannot
/// localize the mismatch.  Length mismatch is also a mismatch.
pub fn part_verify_nonce<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    nonce: &[u8],
) -> HsmResult<()> {
    let stored = part_nonce(pal, io)?;
    if stored.len() != nonce.len() {
        return Err(HsmError::NonceMismatch);
    }
    let mut diff = 0u8;
    for (a, b) in stored.iter().zip(nonce.iter()) {
        diff |= a ^ b;
    }
    if diff == 0 {
        Ok(())
    } else {
        Err(HsmError::NonceMismatch)
    }
}

// ─── PAL-private root-of-trust seed rows ──────────────────────────────────
//
// Indexed properties exposing the manufacturer-provisioned and
// device-owner-provisioned seed tables that anchor the masking-key
// derivation in `derive_masking_key` (see `ddi::mbor::establish_credential`).
// Only rows actually provisioned by the current PAL are present;
// unprovisioned rows return [`HsmError::PartPropNotFound`].

/// Manufacturer-provisioned 32 B seed (BKS1) for a specific `svn`.
/// **Sensitive** device-global root-of-trust material.
///
/// Device-global (not per-partition); served by
/// [`HsmSeedStore::mfgr_seed`](tsra_fw_hsm_pal_traits::HsmSeedStore::mfgr_seed).
/// Returns [`HsmError::SeedNotFound`] when the PAL has not provisioned
/// a row for that SVN.
pub fn part_mfgr_seed(pal: &impl HsmSeedStore, svn: u64) -> HsmResult<&[u8]> {
    pal.mfgr_seed(svn)
}

/// Device-owner-provisioned 32 B seed (BKS2) for a specific owner
/// selector `svn`.  **Sensitive** device-global root-of-trust material.
///
/// Device-global (not per-partition); served by
/// [`HsmSeedStore::dev_owner_seed`](tsra_fw_hsm_pal_traits::HsmSeedStore::dev_owner_seed).
/// Returns [`HsmError::SeedNotFound`] when the PAL has not provisioned
/// a row for that selector.
pub fn part_owner_seed(pal: &impl HsmSeedStore, svn: u64) -> HsmResult<&[u8]> {
    pal.dev_owner_seed(svn)
}

// ─── Boot / launch-time bound material ────────────────────────────────────
//
// Material bound to a single boot / VM-launch incarnation.  Set once
// per power cycle (or once per VM launch) and either consumed by the
// session-establishment flow or used as inputs into derivations whose
// outputs are themselves boot-bound.

/// Sealed BK3 blob.  **Sensitive.**
///
/// Wraps [`HsmPartPropId::SEALED_BK3`]
/// (`VarBytes { max: SEALED_BK3_MAX_LEN }`, `AbsentUntilSet`,
/// sensitive).  Returns [`HsmError::PartPropNotFound`] before the
/// host has supplied it this power cycle.  Set at most once per
/// power cycle.
pub fn part_sealed_bk3<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::SEALED_BK3)
}

/// Write the sealed BK3 blob.  **Sensitive.**
///
/// `data` must be ≤ [`SEALED_BK3_MAX_LEN`] bytes; the PAL setter
/// is write-once per power cycle and returns
/// [`HsmError::SealedBk3AlreadySet`] if invoked a second time
/// without an intervening clear (free / NSSR).
pub fn part_set_sealed_bk3<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    data: &HsmDmaBuf,
) -> HsmResult<()> {
    pal.part_prop_set_bytes(io, HsmPartPropId::SEALED_BK3, data)
}

/// [`HsmPartPropId`] backing [`part_sealed_bk3`] / [`part_set_sealed_bk3`].
#[inline]
pub const fn part_sealed_bk3_prop_id() -> HsmPartPropId {
    HsmPartPropId::SEALED_BK3
}

/// Masked BK_BOOT blob (variable, ≤ [`MASKED_BK_BOOT_LEN`]).
/// **Sensitive.**
///
/// Wraps [`HsmPartPropId::MASKED_BK_BOOT`]
/// (`VarBytes { max: MASKED_BK_BOOT_LEN }`, `AbsentUntilSet`,
/// sensitive).
pub fn part_masked_bk_boot<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::MASKED_BK_BOOT)
}

/// Write the masked BK_BOOT blob.  **Sensitive.**
///
/// `data` must be ≤ [`MASKED_BK_BOOT_LEN`] bytes.
pub fn part_set_masked_bk_boot<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    data: &HsmDmaBuf,
) -> HsmResult<()> {
    pal.part_prop_set_bytes(io, HsmPartPropId::MASKED_BK_BOOT, data)
}

/// [`HsmPartPropId`] backing [`part_masked_bk_boot`] /
/// [`part_set_masked_bk_boot`].
#[inline]
pub const fn part_masked_bk_boot_prop_id() -> HsmPartPropId {
    HsmPartPropId::MASKED_BK_BOOT
}

/// VM-launch GUID (16 B), bound at session-establishment time.
///
/// Wraps [`HsmPartPropId::VM_LAUNCH_GUID`] (`FixedBytes { len: 16 }`,
/// `AbsentUntilSet`, **read-only**).  Returns
/// [`HsmError::PartPropNotFound`] before session establishment has
/// bound the value for the current launch.  Populated by the PAL.
pub fn part_vm_launch_guid<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::VM_LAUNCH_GUID)
}

/// [`HsmPartPropId`] backing [`part_vm_launch_guid`].
#[inline]
pub const fn part_vm_launch_guid_prop_id() -> HsmPartPropId {
    HsmPartPropId::VM_LAUNCH_GUID
}

/// Partition policy hash — the SHA-384 digest (48 bytes) of the
/// partition's PartPolicy blob.
///
/// Wraps [`HsmPartPropId::POLICY_HASH`] (fixed 48 bytes, `AbsentUntilSet`).
/// Set by the TBOR `PartInit` handler from the hash of the request's
/// `part_policy`, and consulted by DDI handlers that gate on policy
/// identity.
pub fn part_policy_hash<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::POLICY_HASH)
}

/// Write the partition policy hash.
///
/// `data` must be exactly the 48-byte SHA-384 digest of the PartPolicy
/// blob.
pub fn part_set_policy_hash<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    data: &HsmDmaBuf,
) -> HsmResult<()> {
    pal.part_prop_set_bytes(io, HsmPartPropId::POLICY_HASH, data)
}

/// [`HsmPartPropId`] backing [`part_policy_hash`] / [`part_set_policy_hash`].
#[inline]
pub const fn part_policy_hash_prop_id() -> HsmPartPropId {
    HsmPartPropId::POLICY_HASH
}

/// POTA thumbprint (48 B).  Set by `PartInit`.
///
/// Wraps [`HsmPartPropId::POTA_THUMBPRINT`] (`FixedBytes { len: 48 }`,
/// `AbsentUntilSet`).
pub fn part_pota_thumbprint<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::POTA_THUMBPRINT)
}

/// Write the POTA thumbprint.
///
/// `data` must be exactly 48 bytes.
pub fn part_set_pota_thumbprint<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    data: &HsmDmaBuf,
) -> HsmResult<()> {
    pal.part_prop_set_bytes(io, HsmPartPropId::POTA_THUMBPRINT, data)
}

/// [`HsmPartPropId`] backing [`part_pota_thumbprint`] /
/// [`part_set_pota_thumbprint`].
#[inline]
pub const fn part_pota_thumbprint_prop_id() -> HsmPartPropId {
    HsmPartPropId::POTA_THUMBPRINT
}

/// Read the SATA (Sealing Authority Trust Anchor) thumbprint.
pub fn part_sata_thumbprint<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::SATA_THUMBPRINT)
}

/// Write the SATA thumbprint.  `data` must be exactly 48 bytes.
pub fn part_set_sata_thumbprint<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    data: &HsmDmaBuf,
) -> HsmResult<()> {
    pal.part_prop_set_bytes(io, HsmPartPropId::SATA_THUMBPRINT, data)
}

/// Read the SAPOTA (Sealing Authority's POTA) thumbprint.
pub fn part_sapota_thumbprint<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::SAPOTA_THUMBPRINT)
}

/// Write the SAPOTA thumbprint.  `data` must be exactly 48 bytes.
pub fn part_set_sapota_thumbprint<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    data: &HsmDmaBuf,
) -> HsmResult<()> {
    pal.part_prop_set_bytes(io, HsmPartPropId::SAPOTA_THUMBPRINT, data)
}

/// Whether the PSK selected by `psk_id` is still the well-known
/// default value.
///
/// Reads the PSK via [`part_psk`] and compares against
/// [`DEFAULT_PSK_CO`] / [`DEFAULT_PSK_CU`] in constant time.  Returns
/// `Ok(true)` only when the stored bytes match the default.
pub fn part_psk_is_default<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    psk_id: u8,
) -> HsmResult<bool> {
    if let Some(is_default) = pal.part_psk_default_memo(io, psk_id) {
        return Ok(is_default);
    }
    let stored = part_psk(pal, io, psk_id)?;
    let default = match psk_id {
        0 => DEFAULT_PSK_CO.as_slice(),
        1 => DEFAULT_PSK_CU.as_slice(),
        // Unreachable: `part_psk` above already rejected any other
        // selector with `InvalidPskId`.  Returned here for defence in
        // depth so the match stays exhaustive.
        _ => return Err(HsmError::InvalidPskId),
    };
    if stored.len() != default.len() {
        return Ok(false);
    }
    // Compare a copy: the stored PSK sits in DMA memory, which may be
    // uncached, where one burst read costs about as much as a single byte
    // load. The copy is wiped once compared.
    let mut copy = PskCopy([0; PSK_LEN]);
    if !stored.copy_to_slice(&mut copy.0) {
        return Ok(false);
    }
    let mut diff = 0u8;
    for (a, b) in copy.0.iter().zip(default.iter()) {
        diff |= a ^ b;
    }
    copy.0.zeroize();
    // Nothing above awaits, so no write to the PSK can come between the
    // comparison and the PAL remembering its result.
    pal.part_psk_memo_default(io, psk_id, diff == 0);
    Ok(diff == 0)
}

/// A word-aligned buffer for a PSK copy (burst copies need alignment).
#[repr(C, align(4))]
struct PskCopy([u8; PSK_LEN]);

/// Whether the partition's caller-presented credential blob has been
/// set for the current incarnation.
///
/// Wraps `part_prop_get_bytes(CREDENTIAL)` and maps
/// [`HsmError::PartPropNotFound`] to `Ok(false)`; any other PAL error
/// propagates.
pub fn part_is_credential_set<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<bool> {
    match pal.part_prop_get_bytes(io, HsmPartPropId::CREDENTIAL) {
        Ok(_) => Ok(true),
        Err(HsmError::PartPropNotFound) => Ok(false),
        Err(e) => Err(e),
    }
}

// ─── Public keys, BKS2, BK3 session (Phase A property additions) ─────

/// Raw ECC-P384 public key (x ∥ y, 96 B) for the partition identity key.
pub fn part_id_pub_key<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::PART_ID_ECC_PUB_KEY)
}

/// Raw ECC-P384 public key (x ∥ y, 96 B) for the session encryption key.
pub fn part_session_enc_pub_key<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::SESSION_ENC_ECC_PUB_KEY)
}

/// Raw ECC-P384 public key (x ∥ y, 96 B) for the establish-credential key.
pub fn part_establish_cred_pub_key<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::ESTABLISH_CRED_ECC_PUB_KEY)
}

/// Raw ECC-P384 public-key coordinates (x ∥ y, 96 B) of the Partition
/// Trust Anchor.
pub fn part_pta_pub_key<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::PTA_ECC_PUB_KEY)
}

/// Validates a host-supplied SEC1-uncompressed P-384 PTA key (`0x04 ‖ x
/// ‖ y`, 97 B) and returns a view of the 96-byte raw `x ‖ y`
/// coordinates (the constant `0x04` prefix is dropped before storage).
fn pta_sec1_to_raw(sec1: &HsmDmaBuf) -> HsmResult<&HsmDmaBuf> {
    const PTA_SEC1_LEN: usize = 97;
    if sec1.len() != PTA_SEC1_LEN || sec1[0] != 0x04 {
        return Err(HsmError::InvalidArg);
    }
    Ok(sec1.split_at(1).1)
}

/// Store the PTA public key from its host-supplied SEC1 form: validates
/// the `0x04` prefix and persists the 96-byte raw coordinates.
pub fn part_set_pta_pub_key<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    sec1: &HsmDmaBuf,
) -> HsmResult<()> {
    let raw = pta_sec1_to_raw(sec1)?;
    pal.part_prop_set_bytes(io, HsmPartPropId::PTA_ECC_PUB_KEY, raw)
}

/// Bind both halves of the Partition Trust Anchor — the vault key id
/// and the public key — in a single composite write.  `pub_sec1` is the
/// host-supplied SEC1 form; its raw coordinates are stored.
pub fn part_set_pta_key<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    key_id: HsmKeyId,
    pub_sec1: &HsmDmaBuf,
) -> HsmResult<()> {
    let raw = pta_sec1_to_raw(pub_sec1)?;
    part_set_pta_key_id(pal, io, key_id)?;
    pal.part_prop_set_bytes(io, HsmPartPropId::PTA_ECC_PUB_KEY, raw)
}

/// BK3 session key (48 B). **Sensitive.**
pub fn part_bk3_session<'a, P: HsmPartitionManager>(
    pal: &'a P,
    io: &P::Io,
) -> HsmResult<&'a HsmDmaBuf> {
    pal.part_prop_get_bytes(io, HsmPartPropId::BK3_SESSION)
}

/// Set the BK3 session key (48 B).
pub fn part_set_bk3_session<P: HsmPartitionManager>(
    pal: &P,
    io: &P::Io,
    data: &HsmDmaBuf,
) -> HsmResult<()> {
    pal.part_prop_set_bytes(io, HsmPartPropId::BK3_SESSION, data)
}

/// Whether the partition has completed one-shot BK3 initialization.
pub fn part_is_bk3_initialized<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<bool> {
    pal.part_prop_get_bool(io, HsmPartPropId::BK3_INITIALIZED)
}

/// Atomically commit the partition's one-shot BK3 init state to
/// `true`.  Returns [`HsmError::Bk3AlreadyInitialized`] if the flag
/// was already set in the current partition incarnation.  This is the
/// authoritative race-winner gate for `DdiInitBk3`.
pub fn part_mark_bk3_initialized<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<()> {
    pal.part_prop_set_bool(io, HsmPartPropId::BK3_INITIALIZED, true)
}

/// Whether the partition has completed one-shot security-domain
/// initialization (its `SDMK` is provisioned).
pub fn part_is_sd_initialized<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<bool> {
    pal.part_prop_get_bool(io, HsmPartPropId::SD_INITIALIZED)
}

/// Atomically commit the partition's one-shot security-domain init
/// state to `true`.  Returns [`HsmError::SdAlreadyInitialized`] if the
/// flag was already set in the current partition incarnation.  This is
/// the authoritative race-winner gate for `SdCreateRemoteBackup`.
pub fn part_mark_sd_initialized<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<()> {
    pal.part_prop_set_bool(io, HsmPartPropId::SD_INITIALIZED, true)
}

/// Clear the partition's one-shot security-domain init flag.
///
/// Used only to best-effort roll back a just-made claim when its undo
/// inverse could not be recorded, so a full undo log cannot permanently
/// wedge the partition's SD gate.  (The one-shot setter permits
/// `→ false`; the normal rollback path is the undo log.)
pub fn part_clear_sd_initialized<P: HsmPartitionManager>(pal: &P, io: &P::Io) -> HsmResult<()> {
    pal.part_prop_set_bool(io, HsmPartPropId::SD_INITIALIZED, false)
}

/// Current owner-seed (BKS2) selector (BKS2 lineage; always 0 today).
///
/// Device-global (not per-partition); served by
/// [`HsmSeedStore::dev_owner_svn`](tsra_fw_hsm_pal_traits::HsmSeedStore::dev_owner_svn).
pub fn part_owner_svn(pal: &impl HsmSeedStore) -> u64 {
    pal.dev_owner_svn()
}
