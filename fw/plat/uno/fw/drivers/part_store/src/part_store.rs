// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Partition persistent store: typed GSRAM layout and the [`PartStore`]
//! accessor handle.

use core::mem::offset_of;
use core::mem::size_of;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU32;
use core::sync::atomic::AtomicU8;
use core::sync::atomic::Ordering::Relaxed;

use tsra_fw_hsm_pal_traits::HsmDmaBuf;
use tsra_fw_hsm_pal_traits::HsmError;
use tsra_fw_hsm_pal_traits::HsmKeyId;
use tsra_fw_hsm_pal_traits::HsmPartState;
use tsra_fw_hsm_pal_traits::HsmResult;
use tsra_fw_hsm_pal_traits::PskPolicySnapshot;
use tsra_fw_hsm_pal_traits::DEFAULT_PSK_CO;
use tsra_fw_hsm_pal_traits::DEFAULT_PSK_CU;
use tsra_fw_hsm_pal_traits::PSK_POLICY_LEN;
use tsra_fw_uno_reg_soc::io_gsram::IO_GSRAM_BASE;
use tsra_fw_uno_reg_soc::part_store_t::PART_STORE_T_BASE;

/// Number of partition slots in the persistent store.
pub const NUM_PARTITIONS: usize = 65;

/// Per partition, what has been found of whether each PSK still holds its
/// well-known default (see [`Partition::psk_default_memo`]), so the
/// default-PSK gate need not read a PSK back out of GSRAM, which the CPU
/// reads uncached, for every command. Every write to the partition's slot
/// clears it (see [`Partition::slot_mut`]), so no answer outlives a change
/// to either PSK.
static PSK_DEFAULT: [AtomicU8; NUM_PARTITIONS] = [const { AtomicU8::new(0) }; NUM_PARTITIONS];

/// Per partition, whether its unwrapping-key slot has been read as
/// [`UnwrappingKeySlot::PctPassed`], which every command checks for a key
/// awaiting its test. The SP only ever writes an empty slot, so a passed one
/// changes only when the HSM writes the slot, which clears this (see
/// [`Partition::slot_mut`]); until then the slot need not be read out of
/// GSRAM again.
static UNWRAP_PASSED: [AtomicBool; NUM_PARTITIONS] =
    [const { AtomicBool::new(false) }; NUM_PARTITIONS];

/// The [`PSK_DEFAULT`] bits of the crypto-officer (`co`) or crypto-user
/// PSK: whether it has been compared with its default, and whether it
/// matched.
const fn psk_memo_bits(co: bool) -> (u8, u8) {
    if co {
        (1 << 0, 1 << 1)
    } else {
        (1 << 2, 1 << 3)
    }
}

/// Logical partition ID shared by Uno's PAL and persistent-store drivers.
///
/// Transparent over `u8`, with checked construction from `u16`.
/// Partition operations validate the index against [`NUM_PARTITIONS`].
/// This is not a hardware AXI routing ID.
///
/// ```
/// use tsra_fw_uno_drivers_part_store::HsmPartId;
///
/// assert_eq!(u8::from(HsmPartId::from(3u8)), 3);
/// assert!(HsmPartId::try_from(300u16).is_err());
/// ```
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HsmPartId(u8);

impl From<u8> for HsmPartId {
    #[inline]
    fn from(v: u8) -> Self {
        Self(v)
    }
}

impl From<HsmPartId> for u8 {
    #[inline]
    fn from(id: HsmPartId) -> Self {
        id.0
    }
}

impl TryFrom<u16> for HsmPartId {
    type Error = core::num::TryFromIntError;

    #[inline]
    fn try_from(v: u16) -> Result<Self, Self::Error> {
        u8::try_from(v).map(Self)
    }
}

/// Persistent-store schema version stamped by this driver.
pub const STORE_VERSION: u8 = 1;

// ── Field sizes (mirror the reference `HsmPartPersistentStore`) ──────────
const ID_LEN: usize = 16;
const ID_PRIV_LEN: usize = 48;
const ID_PUB_LEN: usize = 97;
const UNWRAPPING_KEY_BK_LEN: usize = 516;
const PART_CERT_DATA_LEN: usize = 800;
const MASKED_BK_BOOT_DATA_LEN: usize = 300;
const SEALED_BK3_DATA_LEN: usize = 512;
const BK3_KEY_LEN: usize = 48;
const NONCE_LEN: usize = 32;
const GUID_LEN: usize = 16;
const SESSION_TABLE_LEN: usize = 18;
const POLICY_HASH_LEN: usize = 48;
const POTA_THUMBPRINT_LEN: usize = 48;
const SATA_THUMBPRINT_LEN: usize = 48;
const SAPOTA_THUMBPRINT_LEN: usize = 48;
const PUB_KEY_LEN: usize = 96;
const PSK_LEN: usize = 32;
const CREDENTIAL_LEN: usize = 32;
const RES_MASK_LEN: usize = 16;
/// Trailing reserved tail = reference `reserved3` (626) minus the appended
/// Uno fields (policy_hash + pota/sata/sapota thumbprints + the flat
/// working-state fields below).
const RESERVED3_LEN: usize = 626
    - POLICY_HASH_LEN
    - POTA_THUMBPRINT_LEN
    - SATA_THUMBPRINT_LEN
    - SAPOTA_THUMBPRINT_LEN
    - 4  // state
    - 4  // generation
    - RES_MASK_LEN
    - 10 * 2  // key handles (u16 + sentinel)
    - PUB_KEY_LEN  // ec_pub_key
    - PUB_KEY_LEN  // se_pub_key
    - PSK_LEN  // psk_co
    - PSK_LEN  // psk_cu
    - CREDENTIAL_LEN
    - 1  // valid_flags (8 packed presence/init bits)
    - PUB_KEY_LEN  // pta_pub_key
    - 2; // session_meta (pending_mask + psk_change_mask)

/// Bit positions packed into the per-partition `valid_flags` byte.
const FLAG_CREDENTIAL_VALID: u8 = 1 << 0;
const FLAG_POTA_THUMBPRINT_VALID: u8 = 1 << 1;
const FLAG_SATA_THUMBPRINT_VALID: u8 = 1 << 2;
const FLAG_SAPOTA_THUMBPRINT_VALID: u8 = 1 << 3;
const FLAG_PTA_PUB_KEY_VALID: u8 = 1 << 4;
const FLAG_POLICY_HASH_VALID: u8 = 1 << 5;
const FLAG_BK3_INITIALIZED: u8 = 1 << 6;
const FLAG_SD_INITIALIZED: u8 = 1 << 7;

/// Total per-partition slot size (matches the reference layout).
const STORE_SIZE: usize = 3072;

/// State of the SP-published RSA unwrapping-key slot. The discriminants are
/// the `UnwrappingKeyValidity` byte values that the SP and the HSM write into
/// the slot.
///
/// The SP writes the byte from another processor, so the slot stores a plain
/// `u8`, which can hold any value, and readers decode it with [`TryFrom`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnwrappingKeySlot {
    /// The SP hasn't published a key.
    Empty = 0,
    /// The SP published a key; its pairwise consistency test hasn't passed.
    PendingPct = 1,
    /// The key passed its pairwise consistency test and can be imported.
    PctPassed = 2,
}

impl TryFrom<u8> for UnwrappingKeySlot {
    /// The byte, which isn't a value that this firmware recognizes.
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, u8> {
        [Self::Empty, Self::PendingPct, Self::PctPassed]
            .into_iter()
            .find(|slot| *slot as u8 == value)
            .ok_or(value)
    }
}

/// Bytes in each of a [`Shadow`]'s two runs of slot fields: `state ‖
/// generation ‖ res_mask`, and the key handles followed by `valid_flags`.
const SHADOW_RUN: usize = 24;

/// Words in a [`Shadow`].
const SHADOW_WORDS: usize = 2 * SHADOW_RUN / 4;

/// Where each [`Shadow`] run starts in [`Storage`].
const SHADOW_HEAD: usize = offset_of!(Storage, state);
const SHADOW_TAIL: usize = offset_of!(Storage, id_key_id);

/// A DTCM copy of a partition slot's small, read-mostly fields (lifecycle
/// state, generation, resource mask, key handles and presence flags),
/// which nearly every command reads and CP1 would otherwise read from
/// GSRAM uncached, a load at a time. Loaded on first use; every write to
/// the slot drops it (see [`Partition::slot_mut`]). Only fields written
/// solely through [`Partition`]'s setters, never through a `&mut` the
/// store hands out (which could be written after the drop), belong here.
struct Shadow {
    valid: AtomicBool,
    words: [AtomicU32; SHADOW_WORDS],
}

impl Shadow {
    const fn new() -> Self {
        Self {
            valid: AtomicBool::new(false),
            words: [const { AtomicU32::new(0) }; SHADOW_WORDS],
        }
    }
}

/// Per partition, its [`Shadow`].
static SHADOWS: [Shadow; NUM_PARTITIONS] = [const { Shadow::new() }; NUM_PARTITIONS];

/// The [`Shadow`] word holding the [`Storage`] byte at `off`, which a run
/// covers, and the byte's shift within it.
const fn shadow_word(off: usize) -> (usize, u32) {
    let at = if off >= SHADOW_TAIL {
        SHADOW_RUN + off - SHADOW_TAIL
    } else {
        off - SHADOW_HEAD
    };
    (at / 4, 8 * (at % 4) as u32)
}

/// Sentinel stored in a key-handle field meaning "no key".
///
/// Vault key ids pack `(table << 8) | slot` with `table < 65`, so
/// `0xFFFF` (table 255) is never a real handle and is safe as the
/// absent marker. (`0` is a *valid* handle — table 0, slot 0 — so it
/// cannot be used.)
const KEY_ABSENT: u16 = u16::MAX;

/// Decodes a stored 2-byte handle field into an optional key id.
#[inline]
fn read_handle(bytes: [u8; 2]) -> Option<HsmKeyId> {
    let raw = u16::from_le_bytes(bytes);
    (raw != KEY_ABSENT).then(|| HsmKeyId::from(raw))
}

/// Encodes an optional key id into a 2-byte handle field.
#[inline]
fn write_handle(key: Option<HsmKeyId>) -> [u8; 2] {
    key.map(u16::from).unwrap_or(KEY_ABSENT).to_le_bytes()
}

/// Absolute GSRAM base address of partition slot 0.
#[cfg_attr(test, allow(dead_code))]
const PART_STORE_BASE: usize = (IO_GSRAM_BASE + PART_STORE_T_BASE) as usize;

/// Bytes between consecutive partition slots.
const STRIDE: usize = size_of::<Storage>();

/// Base address of partition slot 0's backing store.
///
/// In production this resolves to the fixed GSRAM MMIO region
/// ([`PART_STORE_BASE`]). Host unit tests have no such region, so a single
/// leaked, zero-initialised heap allocation laid out identically
/// (`NUM_PARTITIONS` x `STRIDE`, `Storage` alignment) stands in for it, letting
/// every [`Partition`] handle address real memory so the lifecycle clears can
/// be exercised and asserted off-target.
#[cfg(not(test))]
#[inline]
fn store_base() -> usize {
    PART_STORE_BASE
}

#[cfg(test)]
fn store_base() -> usize {
    use core::mem::align_of;
    use std::alloc::alloc_zeroed;
    use std::alloc::Layout;
    use std::sync::OnceLock;

    static BASE: OnceLock<usize> = OnceLock::new();
    *BASE.get_or_init(|| {
        let size = NUM_PARTITIONS
            .checked_mul(STRIDE)
            .expect("part-store test size overflow");
        let layout = Layout::from_size_align(size, align_of::<Storage>())
            .expect("valid part-store test layout");
        // SAFETY: `layout` has a non-zero size and an all-zero bit pattern is a
        // valid `Storage` (every field is a byte array or a zero-valued scalar).
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "part-store test backing allocation failed");
        ptr as usize
    })
}

// ── Lockout policy (reference `PinPolicy`) ───────────────────────────────

/// Pin-policy lockout state.
#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PinPolicyState {
    /// Unrestricted.
    #[default]
    Ready,
    /// Locked out; enforce the delay factor.
    Lockout,
}

/// Partition lockout / pin-policy context (14 bytes, reference layout).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct PinPolicy {
    /// Current lockout state.
    pub state: PinPolicyState,
    /// Back-off delay factor.
    pub delay_factor: u16,
    /// Allowed attempts before lockout.
    pub allowed_attempts: u16,
    /// Lockout timestamp (opaque 8-byte counter).
    pub lockout_time: [u8; 8],
}

// ── On-storage sub-structs (crate-private; mirror reference layout) ───────

/// Partition identity: random id, identity private key (reserved/unused on
/// Uno — the private key lives in the vault), and identity public key.
#[repr(C)]
#[derive(Clone, Copy)]
struct PartitionIdentifier {
    id: [u8; ID_LEN],
    priv_key: [u8; ID_PRIV_LEN],
    pub_key: [u8; ID_PUB_LEN],
}

/// Partition certificate (reserved/unused on Uno for now).
#[repr(C)]
#[derive(Clone, Copy)]
struct PartitionCert {
    length: u32,
    data: [u8; PART_CERT_DATA_LEN],
}

/// Masked boot key (length-prefixed blob).
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct MaskedBkBoot {
    len: u32,
    data: [u8; MASKED_BK_BOOT_DATA_LEN],
}

/// Sealed backup key 3 (length-prefixed blob).
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct SealedBk3 {
    len: u32,
    data: [u8; SEALED_BK3_DATA_LEN],
}

/// BK3 session key (validity flag + key material).
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Bk3SessionKey {
    is_valid: bool,
    _reserved: [u8; 3],
    key: [u8; BK3_KEY_LEN],
}

/// Per-partition persistent store slot.
///
/// The region from `version` through `bk3_session_key` is byte-for-byte
/// identical to the reference `HsmPartPersistentStore`; `policy_hash` and
/// `pota_thumbprint` are Uno-specific fields appended into what the
/// reference layout left as `reserved3`.
#[repr(C)]
struct Storage {
    // ===== identical to reference HsmPartPersistentStore =====
    version: u8,
    flags: [u8; 3],
    session_table: [u8; SESSION_TABLE_LEN],
    /// Gate-1 staging flag (offset 22, was `reserved1`): CP writes, SP reads.
    /// `false` = not armed, `true` = armed. When armed (set on `SetResource`
    /// with a non-zero mask), the SP stages the RSA-2048 unwrapping key into
    /// `unwrapping_key_bk` and marks `unwrapping_key_bk_valid` non-zero.
    unwrapping_key_required: bool,
    /// Unwrapping-key slot state: the SP writes `PendingPct` when it publishes
    /// a key, and the HSM writes `PctPassed` once the key passes its pairwise
    /// consistency test. `0` = empty; non-zero = occupied (the SP uses a
    /// 3-state `UnwrappingKeyValidity` enum — Empty / PendingPct / PctPassed —
    /// so this is a `u8`, not a `bool`; reading a value > 1 as a Rust `bool`
    /// would be UB).
    unwrapping_key_bk_valid: u8,
    unwrapping_key_bk: [u8; UNWRAPPING_KEY_BK_LEN],
    // Raw bytes preserve the legacy 14-byte slot without reading an unchecked
    // retained byte as a Rust enum. Decode through PskPolicySnapshot first.
    pin_policy: [u8; PSK_POLICY_LEN],
    vm_launch_guid: [u8; GUID_LEN],
    partition_id_valid: bool,
    reserved2: u8,
    partition_identifier: PartitionIdentifier,
    partition_cert_valid: bool,
    partition_cert: PartitionCert,
    masked_bk_boot: MaskedBkBoot,
    sealed_bk3: SealedBk3,
    nonce: [u8; NONCE_LEN],
    bk3_session_key: Bk3SessionKey,
    // ===== Uno-specific fields, appended flat into reserved3 =====
    // Ordered so the DMA-target public-key buffers (`ec_pub_key`,
    // `se_pub_key`, written directly by the ECC keygen engine) land on
    // 4-byte boundaries. The reference region ends 4-aligned, and every
    // field up to and including the buffers has a size that is a multiple
    // of 4, so each stays aligned; the only non-multiple-of-4 fields — the
    // seven `[u8;2]` key handles — are placed last, before `reserved3`,
    // where misalignment is harmless. All fields are byte arrays (align 1)
    // so no implicit padding is introduced; accessors do the LE conversions.
    state: [u8; 4],
    generation: [u8; 4],
    res_mask: [u8; RES_MASK_LEN],
    ec_pub_key: [u8; PUB_KEY_LEN],
    se_pub_key: [u8; PUB_KEY_LEN],
    pta_pub_key: [u8; PUB_KEY_LEN],
    psk_co: [u8; PSK_LEN],
    psk_cu: [u8; PSK_LEN],
    credential: [u8; CREDENTIAL_LEN],
    policy_hash: [u8; POLICY_HASH_LEN],
    pota_thumbprint: [u8; POTA_THUMBPRINT_LEN],
    sata_thumbprint: [u8; SATA_THUMBPRINT_LEN],
    sapota_thumbprint: [u8; SAPOTA_THUMBPRINT_LEN],
    id_key_id: [u8; 2],
    ec_key_id: [u8; 2],
    se_key_id: [u8; 2],
    mk_key_id: [u8; 2],
    ups_key_id: [u8; 2],
    pta_key_id: [u8; 2],
    unwrapping_key_id: [u8; 2],
    local_mk_key_id: [u8; 2],
    ephemeral_mk_key_id: [u8; 2],
    sd_mk_key_id: [u8; 2],
    // Presence flags for `AbsentUntilSet` byte fields plus the one-shot
    // InitBk3 and security-domain (`SD_INITIALIZED`) gates, bit-packed into
    // a single byte (see the `FLAG_*` constants).  Placed after the key
    // handles, before `reserved3`, where 1-byte alignment is harmless and
    // the DMA-target public keys above stay 4-aligned.
    valid_flags: u8,
    // Volatile TBOR session slot metadata: byte 0 = pending_mask
    // (bit N set while slot N's handshake is in flight), byte 1 =
    // psk_change_mask (bit N set once slot N has consumed its one
    // allowed PSK change). Owned by the session_store driver.
    session_meta: [u8; 2],
    reserved3: [u8; RESERVED3_LEN],
}

// Compile-time layout assertions (see `part_store/layout_asserts.rs`).
#[path = "part_store/layout_asserts.rs"]
mod layout_asserts;

/// GSRAM-backed partition persistent store.
///
/// Stateless entry point: [`PartStore::partition`] validates a raw
/// partition id and returns a [`Partition`] handle through which the
/// slot's fields are accessed. [`PartStore::init_default`] initializes
/// every slot on power-on boot; warm boots preserve the store.
#[derive(Debug)]
pub struct PartStore;

/// Selects how much partition state [`Partition::clear_state`] wipes.
///
/// Both kinds clear the per-tenant runtime state (enable-time key handles,
/// credential, BK3 session key, nonce, volatile session metadata, and PIN
/// policy); they differ only in whether the write-once provisioning material is
/// preserved. The persistent session table is left for the PAL to manage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartResetKind {
    /// NSSR `Migrate`: preserve the partition's provisioning material.
    Migrate,
    /// Partition deallocation / disable: also wipe the provisioning material.
    Disable,
}

/// A validated partition-store slot.
///
/// Obtained only via [`PartStore::partition`], so holding one is proof
/// that the partition index is in range (`< NUM_PARTITIONS`). Field
/// accessors are therefore infallible with respect to the index; only the
/// byte-buffer setters can fail (on a wrong input length).
#[derive(Debug, Clone, Copy)]
pub struct Partition(usize);

impl PartStore {
    /// Validates a raw partition id and returns a handle to its slot.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `pid >= NUM_PARTITIONS`.
    #[inline]
    pub fn partition(pid: HsmPartId) -> HsmResult<Partition> {
        let idx = usize::from(u8::from(pid));
        if idx < NUM_PARTITIONS {
            Ok(Partition(idx))
        } else {
            Err(HsmError::InvalidArg)
        }
    }

    /// Initializes every partition slot to a zeroed posture and stamps the
    /// store version. Called once during PAL init (GSRAM is not guaranteed
    /// zeroed).
    pub fn init_default() {
        for idx in 0..NUM_PARTITIONS {
            Partition(idx).reset();
        }
    }

    /// Re-arms Gate 1 (`unwrapping_key_required`) for every partition that
    /// still owns table resources, after a warm / fw-update reboot.
    ///
    /// The GSRAM-resident store survives a warm reset, so each slot's
    /// `res_mask` is already correct and no wipe is performed. But the SP
    /// consumes and clears Gate 1 on every boot. Re-arm
    /// `unwrapping_key_required` wherever the resource mask is non-zero,
    /// without relying on a subsequent `SetResource` replay, so the SP
    /// re-stages that partition's RSA unwrapping key. Mirrors the reference
    /// `PartStateImpl::restore_res_mask`.
    pub fn rearm_unwrapping_key_required() {
        for idx in 0..NUM_PARTITIONS {
            let part = Partition(idx);
            if part.res_mask() != 0 {
                part.set_unwrapping_key_required(true);
            }
        }
    }
}

impl Partition {
    /// Raw pointer to this partition's slot in GSRAM.
    #[inline]
    fn slot_ptr(self) -> *mut Storage {
        // `self.0 < NUM_PARTITIONS` by construction (see `PartStore::partition`).
        (store_base() + self.0 * STRIDE) as *mut Storage
    }

    /// Shared reference to this partition's slot.
    #[inline]
    fn slot(self) -> &'static Storage {
        // SAFETY: the index is in range by construction, keeping the slot
        // within the reserved partition-store GSRAM region; plain shared
        // SRAM, single-threaded executor guarantees no aliasing mutation.
        unsafe { &*self.slot_ptr() }
    }

    /// Word `i` of this slot's [`Shadow`], loading the shadow first if
    /// need be.
    #[inline]
    fn shadow(self, i: usize) -> u32 {
        let shadow = &SHADOWS[self.0];
        if !shadow.valid.load(Relaxed) {
            self.load_shadow();
        }
        shadow.words[i].load(Relaxed)
    }

    /// The byte of this slot's [`Storage`] at `off`, from the [`Shadow`].
    #[inline]
    fn shadow_byte(self, off: usize) -> u8 {
        let (word, shift) = shadow_word(off);
        (self.shadow(word) >> shift) as u8
    }

    /// The 16-bit little-endian field of this slot at `off` (even, so
    /// within one word), from the [`Shadow`].
    #[inline]
    fn shadow_u16(self, off: usize) -> u16 {
        let (word, shift) = shadow_word(off);
        (self.shadow(word) >> shift) as u16
    }

    /// Load this slot's [`Shadow`] with two burst reads.
    #[cold]
    #[inline(never)]
    fn load_shadow(self) {
        let slot = self.slot() as *const Storage as *const u8;
        let shadow = &SHADOWS[self.0];
        for (i, word) in shadow.words.iter().enumerate() {
            let off = if i < SHADOW_RUN / 4 {
                SHADOW_HEAD + 4 * i
            } else {
                SHADOW_TAIL + 4 * i - SHADOW_RUN
            };
            // SAFETY: both runs are word-aligned and lie within the slot
            // (asserted in `layout_asserts`), which `slot` borrows for
            // `'static`.
            let v = unsafe { slot.add(off).cast::<u32>().read() };
            word.store(u32::from_le(v), Relaxed);
        }
        shadow.valid.store(true, Relaxed);
    }

    /// Exclusive reference to this partition's slot, borrowed for the
    /// lifetime of the `&mut self` handle.
    ///
    /// Taking `&mut self` (rather than `self` by value) deliberately ties
    /// the returned `&mut Storage` to an exclusive borrow of the
    /// [`Partition`] handle, so the borrow checker forbids handing out two
    /// overlapping `&mut` into the same slot through one handle (the old
    /// `&'static mut`-from-`Copy` signature did not). Cross-handle aliasing
    /// (two `Partition`s for the same index) is prevented by call
    /// discipline: every accessor borrow is short-lived and the
    /// single-threaded cooperative executor only yields at `.await`, so no
    /// other task observes a slot mid-mutation.
    #[inline]
    fn slot_mut(&mut self) -> &mut Storage {
        // Any write may change a PSK or a shadowed field: forget them.
        PSK_DEFAULT[self.0].store(0, Relaxed);
        SHADOWS[self.0].valid.store(false, Relaxed);
        UNWRAP_PASSED[self.0].store(false, Relaxed);
        // SAFETY: the index is in range by construction (see
        // `PartStore::partition`), keeping the slot within the reserved
        // partition-store GSRAM region. The `&mut self` receiver scopes the
        // returned borrow; on the single-threaded executor no other context
        // accesses the same slot for that borrow's (non-`await`) duration.
        unsafe { &mut *self.slot_ptr() }
    }

    // ── lifecycle ────────────────────────────────────────────────────────

    /// Wipes this partition's slot (zeroizes all fields), re-stamps the
    /// store version, and resets all key handles to the "absent" sentinel
    /// (zeroing alone would otherwise read back as the valid handle 0).
    #[inline(never)]
    pub fn reset(mut self) {
        let slot = self.slot_mut();
        // SAFETY: `slot` is a valid, uniquely-borrowed `Storage` in GSRAM;
        // an all-zero bit pattern is a valid value for every field.
        unsafe {
            core::ptr::write_bytes(slot as *mut Storage as *mut u8, 0, STORE_SIZE);
        }
        slot.version = STORE_VERSION;
        let absent = KEY_ABSENT.to_le_bytes();
        slot.id_key_id = absent;
        slot.ec_key_id = absent;
        slot.se_key_id = absent;
        slot.mk_key_id = absent;
        slot.ups_key_id = absent;
        slot.pta_key_id = absent;
        slot.local_mk_key_id = absent;
        slot.ephemeral_mk_key_id = absent;
        slot.sd_mk_key_id = absent;
        slot.unwrapping_key_id = absent;
    }

    /// Zeroes the provisioned identity: the 16-byte id, its key handle, and
    /// the cached identity public key.
    #[inline(never)]
    pub fn clear_identity(mut self) {
        self.id_mut().fill(0);
        self.set_id_key_id(None);
        self.id_pub_key_mut().fill(0);
    }

    /// Zeroes the enable-time keys: the establish-credential and
    /// session-encryption key handles and their cached public keys.
    #[inline(never)]
    pub fn clear_enabled_keys(mut self) {
        self.set_ec_key_id(None);
        self.ec_pub_key_mut().fill(0);
        self.set_se_key_id(None);
        self.se_pub_key_mut().fill(0);
    }

    /// Clears a partition's state for a given [`PartResetKind`].
    ///
    /// The shared core (cleared for every kind) drops the enable-time and
    /// provisioning vault-key handles (the vault deletions themselves are the
    /// PAL's responsibility), the cached public keys, the caller-presented
    /// credential, the derived BK3 session key, the nonce, the volatile
    /// session metadata, and the PIN lockout policy — matching the
    /// reference `state.disable()` / `state.migrate()`, which both reset the
    /// policy.
    ///
    /// [`PartResetKind::Disable`] additionally wipes the write-once
    /// provisioning material (PTA public key, policy hash, POTA/SATA/SAPOTA
    /// thumbprints, sealed BK3 + incarnation flag, rotated PSKs, and the VM
    /// launch GUID); [`PartResetKind::Migrate`] preserves it so a host that
    /// resets via NSSR keeps its provisioning and only re-establishes its
    /// credential. This helper leaves the partition identity,
    /// `Masked_BK_BOOT`, persistent session table, resource mask, generation
    /// counter, lifecycle state, and the SP-published unwrapping-key slot
    /// (Gate 1 + backup, wiped only when the partition is freed so a
    /// re-enabled partition re-imports the same key) for the caller to
    /// manage. The PAL
    /// preserves established session reservations on disable and releases
    /// all slots on NSSR/migrate or free.
    #[inline(never)]
    pub fn clear_state(mut self, kind: PartResetKind) {
        // ── Per-tenant runtime state (cleared for every reset kind) ──
        // Enable-time keys + cached public keys.
        self.clear_enabled_keys();
        // Provisioning vault-key handles (material wiped wholesale by caller).
        self.set_mk_key_id(None);
        self.set_ups_key_id(None);
        self.set_pta_key_id(None);
        self.set_local_mk_key_id(None);
        self.set_ephemeral_mk_key_id(None);
        self.set_sd_mk_key_id(None);
        self.set_unwrapping_key_id(None);
        // The SD one-shot must follow its key. `set_sd_mk_key_id(None)` above
        // drops the SDMK handle and the caller wipes the vault material for
        // *every* reset kind, so leaving `sd_initialized` set would leave the
        // partition claiming a security domain whose masking key no longer
        // exists — and `SdCreateRemoteBackup`'s one-shot gate would then
        // refuse to mint a replacement, permanently. The std PAL clears this
        // on both `part_enable` and `clear_enabled_state`; uno previously
        // cleared it only on `Disable`, so an NSSR (`Migrate`) left the flag
        // stranded. Note `bk3_initialized` is deliberately *not* moved here:
        // its `sealed_bk3` blob is preserved by `Migrate`, so that flag stays
        // consistent with its material.
        self.set_sd_initialized(false);
        // Caller-presented secret + derived BK3 session key.
        self.clear_credential();
        self.clear_bk3_session();
        // Reset the PIN lockout policy to default (matches the reference
        // `state.disable()` and `state.migrate()`).
        self.slot_mut().pin_policy = [0; PSK_POLICY_LEN];
        // Per-tenant runtime: nonce and the volatile session metadata. The
        // persistent `session_table` is deliberately *not* touched here,
        // because the right disposition depends on the caller: a disable
        // or migrate (NSSR) keeps established slots for renegotiation so the
        // host can reopen or close the sessions it owns, while a free
        // releases the slots. That policy needs the session-store layout, so
        // it lives one layer up — see `SessionTable::mark_all_needs_renego` /
        // `clear_all`, which every caller of this function invokes. Mirrors
        // the reference, whose `disable` / `migrate` call
        // `session_table().restore(backup())` and whose
        // `clear_partition_info` calls `session_table().restore(0)`.
        {
            let slot = self.slot_mut();
            slot.nonce = [0u8; NONCE_LEN];
            slot.session_meta = [0u8; 2];
        }

        // ── Write-once provisioning material ──
        // Preserved by `Migrate` (NSSR keeps provisioning); wiped by `Disable`
        // (partition disable or deallocation).
        // An NSSR (`Migrate`) returns the partition to its just-allocated
        // state as far as the session PSKs are concerned: `part_psk` is
        // `RequiredPresent` and default-baked at allocation, so the slots
        // must read back as the defaults rather than keeping a rotated
        // value. Re-uses [`bake_default_psks`] so allocation and NSSR can
        // never drift apart. Mirrors the std PAL, whose migrated entry
        // resets `psk_co`/`psk_cu` to `None` and therefore reads back as
        // `DEFAULT_PSK_CO` / `DEFAULT_PSK_CU`. `Disable` instead zeroes
        // them below; the PAL restores defaults on a subsequent enable.
        if matches!(kind, PartResetKind::Migrate) {
            self.bake_default_psks();
        }

        if matches!(kind, PartResetKind::Disable) {
            self.clear_pta_pub_key();
            self.clear_policy_hash();
            self.clear_pota_thumbprint();
            self.clear_sata_thumbprint();
            self.clear_sapota_thumbprint();
            self.clear_sealed_bk3();
            self.set_bk3_initialized(false);
            let slot = self.slot_mut();
            slot.psk_co = [0u8; PSK_LEN];
            slot.psk_cu = [0u8; PSK_LEN];
            slot.vm_launch_guid = [0u8; GUID_LEN];
        }
    }

    /// What has been found of whether the crypto-officer (`co`) or
    /// crypto-user PSK still holds its default, if it has been compared
    /// since the slot was last written (see [`Self::memo_psk_default`]).
    pub fn psk_default_memo(self, co: bool) -> Option<bool> {
        let (checked, default) = psk_memo_bits(co);
        let known = PSK_DEFAULT[self.0].load(Relaxed);
        (known & checked != 0).then_some(known & default != 0)
    }

    /// Remembers whether the crypto-officer (`co`) or crypto-user PSK holds
    /// its default, as a comparison just found, until the slot is next
    /// written.
    pub fn memo_psk_default(self, co: bool, is_default: bool) {
        let (checked, default) = psk_memo_bits(co);
        let found = if is_default {
            checked | default
        } else {
            checked
        };
        let known = PSK_DEFAULT[self.0].load(Relaxed);
        PSK_DEFAULT[self.0].store(known | found, Relaxed);
    }

    /// Bakes the default pre-shared keys into the slot.
    ///
    /// `part_psk` is contractually `RequiredPresent` — "default-baked at
    /// allocation time" from [`DEFAULT_PSK_CO`] / [`DEFAULT_PSK_CU`] — so a
    /// freshly allocated partition must present the defaults rather than the
    /// all-zero GSRAM pattern. Without this the FW runs the TBOR session
    /// handshake's HPKE `auth_psk` schedule with a zero PSK while the host
    /// uses the default, so the derived export secrets diverge and the
    /// Phase-1 confirm MAC fails.
    ///
    /// Called from `part_alloc` and actual `part_enable` transitions. The
    /// reset paths seed their own slots:
    /// `clear_state([`PartResetKind::Migrate`])` re-bakes these same
    /// defaults on an NSSR (so a rotated PSK does not survive the reset),
    /// and [`PartResetKind::Disable`] zeroes them until the partition is
    /// enabled or allocated again.
    #[inline(never)]
    pub fn bake_default_psks(mut self) {
        let slot = self.slot_mut();
        slot.psk_co = DEFAULT_PSK_CO;
        slot.psk_cu = DEFAULT_PSK_CU;
    }

    /// Borrows the partition's 16-byte identity.
    #[inline(never)]
    pub fn id(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().partition_identifier.id) }
    }

    /// Mutably borrows the partition's 16-byte identity.
    #[inline(never)]
    pub fn id_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `id`, exclusively borrowed.
        unsafe { HsmDmaBuf::from_raw_mut(&mut self.slot_mut().partition_identifier.id) }
    }

    /// Sets the partition's 16-byte identity.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `ID_LEN` bytes.
    #[inline(never)]
    pub fn set_id(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != ID_LEN {
            return Err(HsmError::InvalidArg);
        }
        self.slot_mut().partition_identifier.id.copy_from_slice(src);
        Ok(())
    }

    /// Borrows the partition's identity public key (X ‖ Y, 96 B).
    ///
    /// The backing field is 97 B (reference SEC1 layout); Uno uses the
    /// first 96 B (raw X ‖ Y, no SEC1 prefix).
    #[inline(never)]
    pub fn id_pub_key(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().partition_identifier.pub_key[..PUB_KEY_LEN]) }
    }

    /// Mutably borrows the partition's identity public key (96 B).
    #[inline(never)]
    pub fn id_pub_key_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `id_pub_key`, exclusively borrowed.
        unsafe {
            HsmDmaBuf::from_raw_mut(
                &mut self.slot_mut().partition_identifier.pub_key[..PUB_KEY_LEN],
            )
        }
    }

    /// Sets the partition's identity public key (96 B, raw X ‖ Y).
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `PUB_KEY_LEN` bytes.
    #[inline(never)]
    pub fn set_id_pub_key(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != PUB_KEY_LEN {
            return Err(HsmError::InvalidArg);
        }
        self.slot_mut().partition_identifier.pub_key[..PUB_KEY_LEN].copy_from_slice(src);
        Ok(())
    }

    /// Whether the partition identity has been provisioned.
    #[inline(never)]
    pub fn id_valid(self) -> bool {
        self.slot().partition_id_valid
    }

    /// Sets the partition-identity-valid flag.
    #[inline(never)]
    pub fn set_id_valid(mut self, valid: bool) {
        self.slot_mut().partition_id_valid = valid;
    }

    // ── nonce ────────────────────────────────────────────────────────────

    /// Borrows the partition's 32-byte anti-replay nonce.
    #[inline(never)]
    pub fn nonce(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().nonce) }
    }

    /// Mutably borrows the partition's nonce.
    #[inline(never)]
    pub fn nonce_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `nonce`, exclusively borrowed.
        unsafe { HsmDmaBuf::from_raw_mut(&mut self.slot_mut().nonce) }
    }

    /// Sets the partition's nonce.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `NONCE_LEN` bytes.
    #[inline(never)]
    pub fn set_nonce(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != NONCE_LEN {
            return Err(HsmError::InvalidArg);
        }
        self.slot_mut().nonce.copy_from_slice(src);
        Ok(())
    }

    // ── vm launch guid ───────────────────────────────────────────────────

    /// Borrows the host-set VM-launch GUID (16 B).
    #[inline(never)]
    pub fn vm_launch_guid(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().vm_launch_guid) }
    }

    /// Mutably borrows the VM-launch GUID.
    #[inline(never)]
    pub fn vm_launch_guid_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `vm_launch_guid`, exclusively borrowed.
        unsafe { HsmDmaBuf::from_raw_mut(&mut self.slot_mut().vm_launch_guid) }
    }

    /// Sets the VM-launch GUID.
    ///
    /// The GUID arrives inline in the `SetResource` IPC message body
    /// (not DMA-sourced), so this takes a fixed-size array rather than a
    /// [`HsmDmaBuf`]. Mirrors the reference firmware's capture in
    /// `part_init::handle_set_res_cnt`.
    #[inline(never)]
    pub fn set_vm_launch_guid(mut self, v: &[u8; GUID_LEN]) {
        self.slot_mut().vm_launch_guid.copy_from_slice(v);
    }

    // ── masked boot key (variable length) ────────────────────────────────

    /// Borrows the masked boot key, trimmed to its stored length.
    #[inline(never)]
    pub fn masked_bk_boot(self) -> &'static HsmDmaBuf {
        let slot = self.slot();
        // Read the packed length field by value (no reference to it).
        let len = (slot.masked_bk_boot.len as usize).min(MASKED_BK_BOOT_DATA_LEN);
        // SAFETY: GSRAM bytes branded as DMA-accessible; `len` is clamped
        // in-bounds; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&slot.masked_bk_boot.data[..len]) }
    }

    /// Writes the masked boot key (stores length + data).
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` exceeds `MASKED_BK_BOOT_DATA_LEN`.
    #[inline(never)]
    pub fn set_masked_bk_boot(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() > MASKED_BK_BOOT_DATA_LEN {
            return Err(HsmError::InvalidArg);
        }
        let slot = self.slot_mut();
        slot.masked_bk_boot.len = src.len() as u32;
        slot.masked_bk_boot.data[..src.len()].copy_from_slice(src);
        // Zeroize any stale (sensitive) bytes left by a previously longer
        // blob so nothing survives past the new length.
        slot.masked_bk_boot.data[src.len()..].fill(0);
        Ok(())
    }

    // ── sealed BK3 (variable length) ─────────────────────────────────────

    /// Borrows the sealed BK3 blob, trimmed to its stored length.
    #[inline(never)]
    pub fn sealed_bk3(self) -> &'static HsmDmaBuf {
        let slot = self.slot();
        let len = (slot.sealed_bk3.len as usize).min(SEALED_BK3_DATA_LEN);
        // SAFETY: GSRAM bytes branded as DMA-accessible; `len` clamped
        // in-bounds; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&slot.sealed_bk3.data[..len]) }
    }

    /// Stores a sealed BK3 blob, overwriting any prior value.
    ///
    /// This is a raw store: write-once enforcement lives in the PAL
    /// property layer (`part_set_prop(SEALED_BK3)`).
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` exceeds `SEALED_BK3_DATA_LEN`.
    #[inline(never)]
    pub fn set_sealed_bk3(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() > SEALED_BK3_DATA_LEN {
            return Err(HsmError::InvalidArg);
        }
        let slot = self.slot_mut();
        slot.sealed_bk3.len = src.len() as u32;
        slot.sealed_bk3.data[..src.len()].copy_from_slice(src);
        // Clear the unused tail of the storage slot.
        slot.sealed_bk3.data[src.len()..].fill(0);
        Ok(())
    }

    // ── BK3 session key ──────────────────────────────────────────────────

    /// Borrows the 48-byte BK3 session key.
    #[inline(never)]
    pub fn bk3_session(self) -> &'static HsmDmaBuf {
        // SAFETY: `key` is an align-1 packed field; GSRAM bytes branded as
        // DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().bk3_session_key.key) }
    }

    /// Sets the BK3 session key and marks it valid.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `BK3_KEY_LEN` bytes.
    #[inline(never)]
    pub fn set_bk3_session(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != BK3_KEY_LEN {
            return Err(HsmError::InvalidArg);
        }
        let slot = self.slot_mut();
        slot.bk3_session_key.key.copy_from_slice(src);
        slot.bk3_session_key.is_valid = true;
        Ok(())
    }

    /// Whether a BK3 session key has been provisioned.
    #[inline(never)]
    pub fn bk3_session_valid(self) -> bool {
        self.slot().bk3_session_key.is_valid
    }

    /// Clears the BK3 session key (zeroizes and marks absent).
    #[inline(never)]
    pub fn clear_bk3_session(mut self) {
        let slot = self.slot_mut();
        slot.bk3_session_key.key = [0u8; BK3_KEY_LEN];
        slot.bk3_session_key.is_valid = false;
    }

    /// Clears the sealed BK3 blob (marks absent).
    #[inline(never)]
    pub fn clear_sealed_bk3(mut self) {
        let slot = self.slot_mut();
        slot.sealed_bk3.data = [0u8; SEALED_BK3_DATA_LEN];
        slot.sealed_bk3.len = 0;
    }

    /// Clears the masked boot key (marks absent).
    #[inline(never)]
    pub fn clear_masked_bk_boot(mut self) {
        let slot = self.slot_mut();
        slot.masked_bk_boot.data = [0u8; MASKED_BK_BOOT_DATA_LEN];
        slot.masked_bk_boot.len = 0;
    }

    /// Whether InitBk3 has already run for this partition (one-shot gate).
    #[inline(never)]
    pub fn bk3_initialized(self) -> bool {
        self.shadow_byte(offset_of!(Storage, valid_flags)) & FLAG_BK3_INITIALIZED != 0
    }

    /// Sets the one-shot InitBk3 gate.
    #[inline(never)]
    pub fn set_bk3_initialized(mut self, valid: bool) {
        let slot = self.slot_mut();
        if valid {
            slot.valid_flags |= FLAG_BK3_INITIALIZED;
        } else {
            slot.valid_flags &= !FLAG_BK3_INITIALIZED;
        }
    }

    /// Whether a security domain has already been created for this
    /// partition (one-shot gate).
    #[inline(never)]
    pub fn sd_initialized(self) -> bool {
        self.shadow_byte(offset_of!(Storage, valid_flags)) & FLAG_SD_INITIALIZED != 0
    }

    /// Sets the one-shot security-domain init gate.
    #[inline(never)]
    pub fn set_sd_initialized(mut self, valid: bool) {
        let slot = self.slot_mut();
        if valid {
            slot.valid_flags |= FLAG_SD_INITIALIZED;
        } else {
            slot.valid_flags &= !FLAG_SD_INITIALIZED;
        }
    }

    // ── lockout policy ───────────────────────────────────────────────────

    /// Borrow the retained PSK policy bytes. Callers must validate on decode.
    #[inline(never)]
    pub fn psk_policy(self) -> &'static HsmDmaBuf {
        // SAFETY: this slot is backed by DMA-accessible GSRAM.
        unsafe { HsmDmaBuf::from_raw(&self.slot().pin_policy) }
    }

    /// Validate and write a complete snapshot before returning.
    #[inline(never)]
    pub fn set_psk_policy(mut self, bytes: &[u8]) -> HsmResult<()> {
        let snapshot = PskPolicySnapshot::from_bytes(bytes)?;
        self.slot_mut()
            .pin_policy
            .copy_from_slice(snapshot.as_bytes());
        Ok(())
    }

    /// Reads the legacy PIN-policy view of the shared lockout slot.
    #[inline(never)]
    pub fn pin_policy(self) -> HsmResult<PinPolicy> {
        let snapshot = PskPolicySnapshot::from_bytes(self.psk_policy())?;
        Ok(PinPolicy {
            state: if snapshot.is_ready() {
                PinPolicyState::Ready
            } else {
                PinPolicyState::Lockout
            },
            delay_factor: snapshot.delay_factor(),
            allowed_attempts: snapshot.failed_attempts(),
            lockout_time: snapshot.lockout_deadline().to_le_bytes(),
        })
    }

    /// Sets the legacy view, validating it before touching retained storage.
    #[inline(never)]
    pub fn set_pin_policy(self, policy: PinPolicy) -> HsmResult<()> {
        let snapshot = PskPolicySnapshot::new(
            policy.state == PinPolicyState::Ready,
            policy.allowed_attempts,
            policy.delay_factor,
            u64::from_le_bytes(policy.lockout_time),
        )?;
        self.set_psk_policy(snapshot.as_bytes())
    }

    // ── Uno ext fields ───────────────────────────────────────────────────

    /// Borrows the partition policy hash (48 B).
    #[inline(never)]
    pub fn policy_hash(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().policy_hash) }
    }

    /// Mutably borrows the partition policy hash.
    #[inline(never)]
    pub fn policy_hash_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `policy_hash`, exclusively borrowed.
        unsafe { HsmDmaBuf::from_raw_mut(&mut self.slot_mut().policy_hash) }
    }

    /// Sets the partition policy hash (marks it present).
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `POLICY_HASH_LEN` bytes.
    #[inline(never)]
    pub fn set_policy_hash(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != POLICY_HASH_LEN {
            return Err(HsmError::InvalidArg);
        }
        let slot = self.slot_mut();
        slot.policy_hash.copy_from_slice(src);
        slot.valid_flags |= FLAG_POLICY_HASH_VALID;
        Ok(())
    }

    /// Whether the partition policy hash has been provisioned.
    #[inline(never)]
    pub fn policy_hash_valid(self) -> bool {
        self.shadow_byte(offset_of!(Storage, valid_flags)) & FLAG_POLICY_HASH_VALID != 0
    }

    /// Clears the partition policy hash (zeroizes and marks absent).
    #[inline(never)]
    pub fn clear_policy_hash(mut self) {
        let slot = self.slot_mut();
        slot.policy_hash = [0u8; POLICY_HASH_LEN];
        slot.valid_flags &= !FLAG_POLICY_HASH_VALID;
    }

    /// Borrows the POTA public-key thumbprint (48 B).
    #[inline(never)]
    pub fn pota_thumbprint(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().pota_thumbprint) }
    }

    /// Mutably borrows the POTA thumbprint.
    #[inline(never)]
    pub fn pota_thumbprint_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `pota_thumbprint`, exclusively borrowed.
        unsafe { HsmDmaBuf::from_raw_mut(&mut self.slot_mut().pota_thumbprint) }
    }

    /// Sets the POTA thumbprint.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `POTA_THUMBPRINT_LEN`
    ///   bytes.
    #[inline(never)]
    pub fn set_pota_thumbprint(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != POTA_THUMBPRINT_LEN {
            return Err(HsmError::InvalidArg);
        }
        let slot = self.slot_mut();
        slot.pota_thumbprint.copy_from_slice(src);
        slot.valid_flags |= FLAG_POTA_THUMBPRINT_VALID;
        Ok(())
    }

    /// Whether a POTA thumbprint has been provisioned.
    #[inline(never)]
    pub fn pota_thumbprint_valid(self) -> bool {
        self.shadow_byte(offset_of!(Storage, valid_flags)) & FLAG_POTA_THUMBPRINT_VALID != 0
    }

    /// Clears the POTA thumbprint (zeroizes and marks absent).
    #[inline(never)]
    pub fn clear_pota_thumbprint(mut self) {
        let slot = self.slot_mut();
        slot.pota_thumbprint = [0u8; POTA_THUMBPRINT_LEN];
        slot.valid_flags &= !FLAG_POTA_THUMBPRINT_VALID;
    }

    /// Borrows the SATA thumbprint (48 B).
    #[inline(never)]
    pub fn sata_thumbprint(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().sata_thumbprint) }
    }

    /// Sets the SATA thumbprint.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `SATA_THUMBPRINT_LEN`
    ///   bytes.
    #[inline(never)]
    pub fn set_sata_thumbprint(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != SATA_THUMBPRINT_LEN {
            return Err(HsmError::InvalidArg);
        }
        let slot = self.slot_mut();
        slot.sata_thumbprint.copy_from_slice(src);
        slot.valid_flags |= FLAG_SATA_THUMBPRINT_VALID;
        Ok(())
    }

    /// Whether a SATA thumbprint has been provisioned.
    #[inline(never)]
    pub fn sata_thumbprint_valid(self) -> bool {
        self.shadow_byte(offset_of!(Storage, valid_flags)) & FLAG_SATA_THUMBPRINT_VALID != 0
    }

    /// Clears the SATA thumbprint (zeroizes and marks absent).
    #[inline(never)]
    pub fn clear_sata_thumbprint(mut self) {
        let slot = self.slot_mut();
        slot.sata_thumbprint = [0u8; SATA_THUMBPRINT_LEN];
        slot.valid_flags &= !FLAG_SATA_THUMBPRINT_VALID;
    }

    /// Borrows the SAPOTA thumbprint (48 B).
    #[inline(never)]
    pub fn sapota_thumbprint(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().sapota_thumbprint) }
    }

    /// Sets the SAPOTA thumbprint.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `SAPOTA_THUMBPRINT_LEN`
    ///   bytes.
    #[inline(never)]
    pub fn set_sapota_thumbprint(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != SAPOTA_THUMBPRINT_LEN {
            return Err(HsmError::InvalidArg);
        }
        let slot = self.slot_mut();
        slot.sapota_thumbprint.copy_from_slice(src);
        slot.valid_flags |= FLAG_SAPOTA_THUMBPRINT_VALID;
        Ok(())
    }

    /// Whether a SAPOTA thumbprint has been provisioned.
    #[inline(never)]
    pub fn sapota_thumbprint_valid(self) -> bool {
        self.shadow_byte(offset_of!(Storage, valid_flags)) & FLAG_SAPOTA_THUMBPRINT_VALID != 0
    }

    /// Clears the SAPOTA thumbprint (zeroizes and marks absent).
    #[inline(never)]
    pub fn clear_sapota_thumbprint(mut self) {
        let slot = self.slot_mut();
        slot.sapota_thumbprint = [0u8; SAPOTA_THUMBPRINT_LEN];
        slot.valid_flags &= !FLAG_SAPOTA_THUMBPRINT_VALID;
    }

    // ── lifecycle state / generation / resource mask ─────────────────────

    /// Reads the partition lifecycle [`HsmPartState`].
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — the stored byte is not a known state.
    #[inline(never)]
    pub fn state(self) -> HsmResult<HsmPartState> {
        HsmPartState::from_u8(self.shadow_byte(offset_of!(Storage, state)))
            .ok_or(HsmError::InvalidArg)
    }

    /// Sets the partition lifecycle [`HsmPartState`].
    #[inline(never)]
    pub fn set_state(mut self, state: HsmPartState) {
        self.slot_mut().state = [state as u8, 0, 0, 0];
    }

    /// Reads the monotonic generation counter.
    #[inline(never)]
    pub fn generation(self) -> u32 {
        self.shadow(shadow_word(offset_of!(Storage, generation)).0)
    }

    /// Sets the generation counter.
    #[inline(never)]
    pub fn set_generation(mut self, generation: u32) {
        self.slot_mut().generation = generation.to_le_bytes();
    }

    /// Increments the generation counter (wrapping).
    #[inline(never)]
    pub fn bump_generation(self) {
        let next = self.generation().wrapping_add(1);
        self.set_generation(next);
    }

    /// Reads the 128-bit table-ownership resource mask.
    #[inline(never)]
    pub fn res_mask(self) -> u128 {
        let (first, _) = shadow_word(offset_of!(Storage, res_mask));
        (0..4).fold(0u128, |mask, i| {
            mask | u128::from(self.shadow(first + i)) << (32 * i)
        })
    }

    /// Reads word `word` (bits `32 * word..32 * word + 32`) of the
    /// resource mask; `0` past its end.
    #[inline]
    pub fn res_mask_word(self, word: usize) -> u32 {
        if word >= RES_MASK_LEN / 4 {
            return 0;
        }
        let (first, _) = shadow_word(offset_of!(Storage, res_mask));
        self.shadow(first + word)
    }

    /// Sets the resource mask.
    #[inline(never)]
    pub fn set_res_mask(mut self, mask: u128) {
        self.slot_mut().res_mask = mask.to_le_bytes();
    }

    // ── key handles (`None` = absent; see `KEY_ABSENT`) ──────────────────

    /// Reads the identity key handle.
    #[inline(never)]
    pub fn id_key_id(self) -> Option<HsmKeyId> {
        read_handle(
            self.shadow_u16(offset_of!(Storage, id_key_id))
                .to_le_bytes(),
        )
    }

    /// Sets (or clears, with `None`) the identity key handle.
    ///
    /// Keeps `partition_id_valid` in sync with handle presence: the PAL
    /// gates identity-property access on `id_key_id().is_some()`, so the
    /// stored validity flag tracks the same "identity provisioned" state
    /// (set on provisioning, cleared on `clear_identity` / free).
    #[inline(never)]
    pub fn set_id_key_id(mut self, key: Option<HsmKeyId>) {
        let slot = self.slot_mut();
        slot.id_key_id = write_handle(key);
        slot.partition_id_valid = key.is_some();
    }

    /// Reads the establish-credential key handle.
    #[inline(never)]
    pub fn ec_key_id(self) -> Option<HsmKeyId> {
        read_handle(
            self.shadow_u16(offset_of!(Storage, ec_key_id))
                .to_le_bytes(),
        )
    }

    /// Sets (or clears) the establish-credential key handle.
    #[inline(never)]
    pub fn set_ec_key_id(mut self, key: Option<HsmKeyId>) {
        self.slot_mut().ec_key_id = write_handle(key);
    }

    /// Reads the session-encryption key handle.
    #[inline(never)]
    pub fn se_key_id(self) -> Option<HsmKeyId> {
        read_handle(
            self.shadow_u16(offset_of!(Storage, se_key_id))
                .to_le_bytes(),
        )
    }

    /// Sets (or clears) the session-encryption key handle.
    #[inline(never)]
    pub fn set_se_key_id(mut self, key: Option<HsmKeyId>) {
        self.slot_mut().se_key_id = write_handle(key);
    }

    /// Reads the masking key handle.
    #[inline(never)]
    pub fn mk_key_id(self) -> Option<HsmKeyId> {
        read_handle(
            self.shadow_u16(offset_of!(Storage, mk_key_id))
                .to_le_bytes(),
        )
    }

    /// Sets (or clears) the masking key handle.
    #[inline(never)]
    pub fn set_mk_key_id(mut self, key: Option<HsmKeyId>) {
        self.slot_mut().mk_key_id = write_handle(key);
    }

    /// Reads the UMS (unique-machine-secret) key handle.
    #[inline(never)]
    pub fn ups_key_id(self) -> Option<HsmKeyId> {
        read_handle(
            self.shadow_u16(offset_of!(Storage, ups_key_id))
                .to_le_bytes(),
        )
    }

    /// Sets (or clears) the UMS key handle.
    #[inline(never)]
    pub fn set_ups_key_id(mut self, key: Option<HsmKeyId>) {
        self.slot_mut().ups_key_id = write_handle(key);
    }

    /// Reads the PTA (partition trust anchor) key handle.
    #[inline(never)]
    pub fn pta_key_id(self) -> Option<HsmKeyId> {
        read_handle(
            self.shadow_u16(offset_of!(Storage, pta_key_id))
                .to_le_bytes(),
        )
    }

    /// Sets (or clears) the PTA key handle.
    #[inline(never)]
    pub fn set_pta_key_id(mut self, key: Option<HsmKeyId>) {
        self.slot_mut().pta_key_id = write_handle(key);
    }

    /// Reads the RSA unwrapping key handle.
    #[inline(never)]
    pub fn unwrapping_key_id(self) -> Option<HsmKeyId> {
        read_handle(
            self.shadow_u16(offset_of!(Storage, unwrapping_key_id))
                .to_le_bytes(),
        )
    }

    /// Sets (or clears) the RSA unwrapping key handle.
    #[inline(never)]
    pub fn set_unwrapping_key_id(mut self, key: Option<HsmKeyId>) {
        self.slot_mut().unwrapping_key_id = write_handle(key);
    }

    // ── RSA unwrapping-key backup (SP-published) — two-gate CP↔SP handshake ──
    //
    // Gate 1: the CP arms `unwrapping_key_required` (via `set_unwrapping_key_required`,
    // on `SetResource` with a non-zero mask). Gate 2: the SP, seeing Gate 1
    // armed, stages a ready-to-use 516-byte PKA-LE RSA-2048 private key into
    // `unwrapping_key_bk` and marks the slot `PendingPct`. The HSM runs the
    // key's pairwise consistency test (PCT) on a copy of the slot bytes and,
    // if the slot still holds them, marks it `PctPassed`; only then does it
    // import the key into its vault (recording `unwrapping_key_id`). The key
    // persists in the slot (it is not consumed on import) until a boot,
    // partition deallocation, or a failed PCT wipes it. The 516-byte payload
    // is the RSA-2048 private key in PKA little-endian order,
    // `d(256) ‖ n(256) ‖ e(4)`; the wire public key is the trailing `n ‖ e`
    // (derived on demand by `rsa_priv_pub_key`).

    /// Arms/disarms Gate 1 (`unwrapping_key_required`): the CP-only flag the SP
    /// reads to decide whether to stage an unwrapping key for this partition.
    #[inline(never)]
    pub fn set_unwrapping_key_required(mut self, required: bool) {
        // Volatile: this is a CP→SP shared-memory mailbox byte. A plain store
        // could be reordered or elided by the compiler, breaking the handshake.
        // SAFETY: valid, aligned, writable byte in this partition's GSRAM slot.
        unsafe {
            core::ptr::write_volatile(&mut self.slot_mut().unwrapping_key_required, required);
        }
    }

    /// Whether Gate 1 (`unwrapping_key_required`) is currently armed.
    #[inline(never)]
    pub fn unwrapping_key_required(self) -> bool {
        // Volatile read of the CP↔SP mailbox byte.
        // SAFETY: valid, aligned, readable byte in this partition's GSRAM slot.
        unsafe { core::ptr::read_volatile(&self.slot().unwrapping_key_required) }
    }

    /// Reads the state of the SP-published unwrapping-key slot. Returns `None`
    /// if the slot holds a byte that this firmware doesn't recognize, which
    /// callers treat as not importable.
    ///
    /// The SP writes `PendingPct` when it publishes a key and treats any
    /// non-zero value as occupied, so the HSM can advance the slot to
    /// `PctPassed` without the SP noticing. A slot once read as passed is
    /// remembered as such until the HSM next writes it (see
    /// [`UNWRAP_PASSED`]).
    #[inline(never)]
    pub fn unwrapping_key_slot(self) -> Option<UnwrappingKeySlot> {
        if UNWRAP_PASSED[self.0].load(Relaxed) {
            return Some(UnwrappingKeySlot::PctPassed);
        }
        // Volatile read: the SP writes this mailbox byte from another processor,
        // so a cached/hoisted load could miss the publish.
        // SAFETY: valid, aligned, readable byte in this partition's GSRAM slot.
        let value = unsafe { core::ptr::read_volatile(&self.slot().unwrapping_key_bk_valid) };
        // Acquire fence (a `dmb` on the Cortex-M7): the SP writes the 516-byte
        // `unwrapping_key_bk` payload before it sets this byte, so the HSM must
        // not read the payload until it has read this byte.
        core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
        let slot = UnwrappingKeySlot::try_from(value).ok();
        if slot == Some(UnwrappingKeySlot::PctPassed) {
            UNWRAP_PASSED[self.0].store(true, Relaxed);
        }
        slot
    }

    /// Marks the published unwrapping key as having passed its pairwise
    /// consistency test, which allows the HSM to import it.
    #[inline(never)]
    pub fn mark_unwrapping_key_pct_passed(mut self) {
        // Release fence (a `dmb` on the Cortex-M7): every read of the tested
        // payload happens before the mark that allows the import.
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        // Volatile: CP↔SP mailbox byte.
        // SAFETY: valid, aligned, writable byte in this partition's GSRAM slot.
        unsafe {
            core::ptr::write_volatile(
                &mut self.slot_mut().unwrapping_key_bk_valid,
                UnwrappingKeySlot::PctPassed as u8,
            );
        }
    }

    /// Borrows the 516-byte PKA-LE RSA-2048 unwrapping-key backup published by
    /// the HSP. Only meaningful when the slot isn't
    /// [`UnwrappingKeySlot::Empty`] (see
    /// [`unwrapping_key_slot`](Self::unwrapping_key_slot)).
    #[inline(never)]
    pub fn unwrapping_key_bk(self) -> &'static HsmDmaBuf {
        // SAFETY: `unwrapping_key_bk` is an align-1 packed field; GSRAM bytes
        // branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().unwrapping_key_bk) }
    }

    /// Wipes the published unwrapping key and marks the slot empty, leaving
    /// Gate 1 armed. The HSM does this when the key fails its pairwise
    /// consistency test; [`clear_unwrapping_key`](Self::clear_unwrapping_key)
    /// also disarms Gate 1.
    #[inline(never)]
    pub fn discard_unwrapping_key(mut self) {
        let slot = self.slot_mut();
        // Wipe the payload before clearing the validity byte, so the SP never
        // sees an occupied slot that holds a wiped key. The validity byte is a
        // CP↔SP mailbox byte → volatile write.
        // SAFETY: valid, aligned bytes in this slot.
        unsafe {
            for byte in &mut slot.unwrapping_key_bk {
                core::ptr::write_volatile(byte, 0);
            }
            // Release fence (a `dmb` on the Cortex-M7), so the hardware keeps
            // that order.
            core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
            core::ptr::write_volatile(
                &mut slot.unwrapping_key_bk_valid,
                UnwrappingKeySlot::Empty as u8,
            );
        }
    }

    /// Resets the SP↔CP unwrapping-key slot on partition teardown: disarms
    /// Gate 1, volatile-wipes the 516-byte payload (so the private key can't be
    /// recovered from GSRAM), and clears the validity byte. A later
    /// `SetResource` re-arms Gate 1 and the SP re-stages a fresh key.
    #[inline(never)]
    pub fn clear_unwrapping_key(mut self) {
        // The gate is a CP↔SP mailbox byte → volatile write.
        // SAFETY: valid, aligned byte in this slot.
        unsafe {
            core::ptr::write_volatile(&mut self.slot_mut().unwrapping_key_required, false);
        }
        self.discard_unwrapping_key();
    }

    /// Reads the partition-local masking key (`PartLocalMK`) handle.
    #[inline(never)]
    pub fn local_mk_key_id(self) -> Option<HsmKeyId> {
        read_handle(
            self.shadow_u16(offset_of!(Storage, local_mk_key_id))
                .to_le_bytes(),
        )
    }

    /// Sets (or clears) the partition-local masking key handle.
    #[inline(never)]
    pub fn set_local_mk_key_id(mut self, key: Option<HsmKeyId>) {
        self.slot_mut().local_mk_key_id = write_handle(key);
    }

    /// Reads the partition ephemeral masking key (`EphemeralMK`) handle.
    #[inline(never)]
    pub fn ephemeral_mk_key_id(self) -> Option<HsmKeyId> {
        read_handle(
            self.shadow_u16(offset_of!(Storage, ephemeral_mk_key_id))
                .to_le_bytes(),
        )
    }

    /// Sets (or clears) the partition ephemeral masking key handle.
    #[inline(never)]
    pub fn set_ephemeral_mk_key_id(mut self, key: Option<HsmKeyId>) {
        self.slot_mut().ephemeral_mk_key_id = write_handle(key);
    }

    /// Reads the security-domain masking key (`SDMK`) handle.
    #[inline(never)]
    pub fn sd_mk_key_id(self) -> Option<HsmKeyId> {
        read_handle(
            self.shadow_u16(offset_of!(Storage, sd_mk_key_id))
                .to_le_bytes(),
        )
    }

    /// Sets (or clears) the security-domain masking key handle.
    #[inline(never)]
    pub fn set_sd_mk_key_id(mut self, key: Option<HsmKeyId>) {
        self.slot_mut().sd_mk_key_id = write_handle(key);
    }

    // ── cached public keys (establish-cred / session-enc) ────────────────

    /// Borrows the establish-credential public key (X ‖ Y, 96 B).
    #[inline(never)]
    pub fn ec_pub_key(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().ec_pub_key) }
    }

    /// Mutably borrows the establish-credential public key.
    #[inline(never)]
    pub fn ec_pub_key_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `ec_pub_key`, exclusively borrowed.
        unsafe { HsmDmaBuf::from_raw_mut(&mut self.slot_mut().ec_pub_key) }
    }

    /// Sets the establish-credential public key.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `PUB_KEY_LEN` bytes.
    #[inline(never)]
    pub fn set_ec_pub_key(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != PUB_KEY_LEN {
            return Err(HsmError::InvalidArg);
        }
        self.slot_mut().ec_pub_key.copy_from_slice(src);
        Ok(())
    }

    /// Borrows the session-encryption public key (X ‖ Y, 96 B).
    #[inline(never)]
    pub fn se_pub_key(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().se_pub_key) }
    }

    /// Mutably borrows the session-encryption public key.
    #[inline(never)]
    pub fn se_pub_key_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `se_pub_key`, exclusively borrowed.
        unsafe { HsmDmaBuf::from_raw_mut(&mut self.slot_mut().se_pub_key) }
    }

    /// Sets the session-encryption public key.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `PUB_KEY_LEN` bytes.
    #[inline(never)]
    pub fn set_se_pub_key(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != PUB_KEY_LEN {
            return Err(HsmError::InvalidArg);
        }
        self.slot_mut().se_pub_key.copy_from_slice(src);
        Ok(())
    }

    // ── platform trust-anchor public key ─────────────────────────────────

    /// Borrows the platform trust-anchor public key (X ‖ Y, 96 B).
    #[inline(never)]
    pub fn pta_pub_key(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().pta_pub_key) }
    }

    /// Sets the platform trust-anchor public key (write-once via caller gate).
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `PUB_KEY_LEN` bytes.
    #[inline(never)]
    pub fn set_pta_pub_key(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != PUB_KEY_LEN {
            return Err(HsmError::InvalidArg);
        }
        let slot = self.slot_mut();
        slot.pta_pub_key.copy_from_slice(src);
        slot.valid_flags |= FLAG_PTA_PUB_KEY_VALID;
        Ok(())
    }

    /// Whether the platform trust-anchor public key has been provisioned.
    #[inline(never)]
    pub fn pta_pub_key_valid(self) -> bool {
        self.shadow_byte(offset_of!(Storage, valid_flags)) & FLAG_PTA_PUB_KEY_VALID != 0
    }

    /// Clears the platform trust-anchor public key (zeroizes and marks absent).
    #[inline(never)]
    pub fn clear_pta_pub_key(mut self) {
        let slot = self.slot_mut();
        slot.pta_pub_key = [0u8; PUB_KEY_LEN];
        slot.valid_flags &= !FLAG_PTA_PUB_KEY_VALID;
    }

    /// Borrows the crypto-officer PSK (32 B).
    #[inline(never)]
    pub fn psk_co(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().psk_co) }
    }

    /// Mutably borrows the crypto-officer PSK.
    #[inline(never)]
    pub fn psk_co_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `psk_co`, exclusively borrowed.
        unsafe { HsmDmaBuf::from_raw_mut(&mut self.slot_mut().psk_co) }
    }

    /// Sets the crypto-officer PSK.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `PSK_LEN` bytes.
    #[inline(never)]
    pub fn set_psk_co(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != PSK_LEN {
            return Err(HsmError::InvalidArg);
        }
        self.slot_mut().psk_co.copy_from_slice(src);
        Ok(())
    }

    /// Borrows the crypto-user PSK (32 B).
    #[inline(never)]
    pub fn psk_cu(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().psk_cu) }
    }

    /// Mutably borrows the crypto-user PSK.
    #[inline(never)]
    pub fn psk_cu_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `psk_cu`, exclusively borrowed.
        unsafe { HsmDmaBuf::from_raw_mut(&mut self.slot_mut().psk_cu) }
    }

    /// Sets the crypto-user PSK.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `PSK_LEN` bytes.
    #[inline(never)]
    pub fn set_psk_cu(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != PSK_LEN {
            return Err(HsmError::InvalidArg);
        }
        self.slot_mut().psk_cu.copy_from_slice(src);
        Ok(())
    }

    // ── user credential ──────────────────────────────────────────────────

    /// Borrows the user credential (id ‖ pin, 32 B).
    #[inline(never)]
    pub fn credential(self) -> &'static HsmDmaBuf {
        // SAFETY: GSRAM bytes branded as DMA-accessible; valid for 'static.
        unsafe { HsmDmaBuf::from_raw(&self.slot().credential) }
    }

    /// Mutably borrows the user credential.
    #[inline(never)]
    pub fn credential_mut(&mut self) -> &mut HsmDmaBuf {
        // SAFETY: as `credential`, exclusively borrowed.
        unsafe { HsmDmaBuf::from_raw_mut(&mut self.slot_mut().credential) }
    }

    /// Sets the user credential.
    ///
    /// # Errors
    ///
    /// - [`HsmError::InvalidArg`] — `v` is not exactly `CREDENTIAL_LEN` bytes.
    #[inline(never)]
    pub fn set_credential(mut self, v: &HsmDmaBuf) -> HsmResult<()> {
        let src: &[u8] = v;
        if src.len() != CREDENTIAL_LEN {
            return Err(HsmError::InvalidArg);
        }
        let slot = self.slot_mut();
        slot.credential.copy_from_slice(src);
        slot.valid_flags |= FLAG_CREDENTIAL_VALID;
        Ok(())
    }

    /// Whether a user credential has been provisioned.
    #[inline(never)]
    pub fn credential_valid(self) -> bool {
        self.shadow_byte(offset_of!(Storage, valid_flags)) & FLAG_CREDENTIAL_VALID != 0
    }

    /// Clears the user credential (zeroizes and marks absent).
    #[inline(never)]
    pub fn clear_credential(mut self) {
        let slot = self.slot_mut();
        slot.credential = [0u8; CREDENTIAL_LEN];
        slot.valid_flags &= !FLAG_CREDENTIAL_VALID;
    }

    // ── session table (raw region; typed view in drivers/session_store) ──

    /// Borrows the raw 18-byte session-table region.
    #[inline(never)]
    pub fn session_table(self) -> &'static [u8; SESSION_TABLE_LEN] {
        &self.slot().session_table
    }

    /// Mutably borrows the raw session-table region.
    #[inline(never)]
    pub fn session_table_mut(&mut self) -> &mut [u8; SESSION_TABLE_LEN] {
        &mut self.slot_mut().session_table
    }

    /// Borrows the 2-byte volatile session metadata (`pending_mask`,
    /// `psk_change_mask`) for the TBOR session slots.
    #[inline(never)]
    pub fn session_meta(self) -> &'static [u8; 2] {
        &self.slot().session_meta
    }

    /// Mutably borrows the session metadata region.
    #[inline(never)]
    pub fn session_meta_mut(&mut self) -> &mut [u8; 2] {
        &mut self.slot_mut().session_meta
    }
}

#[cfg(test)]
#[path = "part_store/tests.rs"]
mod tests;
