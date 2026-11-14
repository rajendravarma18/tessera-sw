// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Platform Abstraction Layer (PAL) trait definitions for the Tessera
//! Integrated HSM firmware.
//!
//! This crate is the **central contract** between the platform-agnostic
//! HSM core (`tsra_fw_hsm_core`) and platform-specific implementations
//! (e.g. `tsra_fw_hsm_pal_std` for host-native simulation,
//! `tsra_fw_uno_pal` for hardware).  It is `#![no_std]` and
//! has no external dependencies beyond `open_enum`, so it compiles on
//! bare-metal targets.
//!
//! # Trait hierarchy
//!
//! The root trait is [`HsmPal`], a supertrait that bundles all required
//! capabilities:
//!
//! ```text
//! HsmPal
//!  ├── HsmAlloc            — per-IO bump-allocator scopes (DTCM and DMA SRAM)
//!  ├── HsmIoController     — I/O submission and completion
//!  ├── HsmGdmaController   — host↔device memory copies
//!  ├── HsmPartitionManager — partition lifecycle
//!  ├── HsmPartitionLock    — per-partition async mutex
//!  ├── HsmCertStore        — per-partition certificate chains
//!  ├── HsmSessionManager   — session allocation and state
//!  ├── HsmVault            — key storage and metadata
//!  ├── HsmKeyMasking       — in-place key masking
//!  └── HsmCrypto           — cryptographic operations
//!       ├── HsmRng         — random number generation
//!       ├── HsmHash        — SHA digest
//!       ├── HsmHmac        — HMAC sign/verify
//!       ├── HsmAes         — AES keygen + ECB/CBC
//!       ├── HsmAesGcm      — AES-GCM
//!       ├── HsmAesKw       — AES-KW/KWP key wrap
//!       ├── HsmAesXts      — AES-XTS
//!       ├── HsmEcc         — ECC keygen/sign/verify/ECDH
//!       ├── HsmRsa         — RSA keygen/mod_exp
//!       └── HsmKdf         — HKDF and KBKDF key derivation
//! ```
//!
//! Capabilities that use the platform's IO type share [`HsmIoTypes`].
//! This type-only trait does not require polling or completion methods;
//! only [`HsmPal`] requires [`HsmIoController`] alongside those capabilities.
//!
//! # Identifier newtypes
//!
//! Two lightweight newtypes — [`HsmKeyId`] and [`HsmSessId`] — prevent
//! accidental mixing of key and session indices. Each wraps a small integer, is
//! `#[repr(transparent)]`, and provides zero-cost [`From`] / [`Into`]
//! conversions to/from its underlying primitive.
//!
//! # Error model
//!
//! All fallible operations return [`HsmResult<T>`], which is a type
//! alias for `Result<T, HsmError>`.  [`HsmError`] is an
//! [`open_enum`] over `u32` with ~200 named variants covering
//! DDI-level, PAL-level, and cryptographic error codes; the numeric
//! values are wire-stable and reused as DDI status codes on the host
//! protocol.
//!
//! # Conventions
//!
//! The following conventions are used uniformly across all PAL
//! sub-traits in this crate:
//!
//! ## `&self` + interior mutability
//!
//! Every method takes `&self`.  PAL implementations are expected to
//! use plain `Cell`/`RefCell` (or static `UnsafeCell`-backed slots)
//! for shared state — the firmware is single-core and cooperatively
//! scheduled, so there are no atomics and no `&mut self` requirement.
//!
//! ## Implicit partition scoping via `HsmIo`
//!
//! Methods that operate on partition-scoped state (sessions, vault
//! keys, certificate chains, partition metadata) take an
//! `&Self::Io` handle from [`HsmIoTypes`] rather than an explicit
//! partition identifier. The platform resolves the partition from private IO
//! metadata, keeping partition selection out of generic command handlers
//! and the call sites uniform.
//!
//! ## Query/copy pattern for variable-length output
//!
//! Methods that return raw bytes into a caller buffer accept
//! `out: Option<&mut [u8]>`:
//!
//! - `out = None` — query mode: returns the required size without
//!   copying.
//! - `out = Some(buf)` — copy mode: writes the data into `buf[..size]`
//!   and returns the same `size`.  `buf.len()` must be ≥ `size` or
//!   the call returns [`HsmError::InvalidArg`].
//!
//! ## Creation operations
//!
//! [`HsmVault::vault_key_create`] and
//! [`HsmSessionManager::session_create`] commit the new entry
//! immediately and return its id. Rollback of a half-completed
//! operation (e.g. when response encoding fails afterwards) is left to
//! a future undo log rather than per-call RAII guards.

#![no_std]
#![allow(async_fn_in_trait)]

mod alloc;
mod cert;
mod cmd_buf;
mod crypto;
mod dispatch;
mod error;
mod gdma;
mod io;
mod key_masking;
mod lock;
mod pal;
mod part;
mod psk_policy;
mod seed;
mod session;
mod vault;

pub use alloc::*;

pub use cert::*;
pub use cmd_buf::*;
pub use crypto::*;
pub use dispatch::*;
pub use error::*;
pub use gdma::*;
pub use io::*;
pub use key_masking::*;
pub use lock::*;
pub use pal::*;
pub use part::*;
pub use psk_policy::*;
pub use seed::*;
pub use session::*;
pub use vault::*;

/// Key identifier — an opaque `u16` index into the vault's key table.
///
/// Returned by [`HsmVault::vault_key_create`] and passed to all subsequent
/// operations (lookup, delete, attribute queries).  The value is only
/// meaningful within the vault that created it; do not reuse a key
/// ID across partitions.
///
/// `#[repr(transparent)]` over `u16`.
///
/// # Conversions
///
/// ```
/// # use tsra_fw_hsm_pal_traits::HsmKeyId;
/// let kid = HsmKeyId::from(42u16);
/// assert_eq!(u16::from(kid), 42);
/// ```
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HsmKeyId(u16);

impl From<u16> for HsmKeyId {
    /// Wraps a raw `u16` index into a [`HsmKeyId`].
    ///
    /// # Parameters
    ///
    /// - `v` — vault key-table index.
    ///
    /// # Returns
    ///
    /// A [`HsmKeyId`] wrapping `v`.
    #[inline]
    fn from(v: u16) -> Self {
        Self(v)
    }
}

impl From<HsmKeyId> for u16 {
    /// Unwraps a [`HsmKeyId`] to its raw `u16` index.
    ///
    /// # Parameters
    ///
    /// - `id` — key identifier.
    ///
    /// # Returns
    ///
    /// The underlying `u16` table index.
    #[inline]
    fn from(id: HsmKeyId) -> Self {
        id.0
    }
}

/// Session identifier — an opaque `u16` slot index into the
/// per-partition session table.
///
/// Returned by [`HsmSessionManager::session_create`] and used by all subsequent
/// operations (state query, deletion).  A session ID is only valid
/// within the partition that allocated it.
///
/// In the standard PAL, slot indices range from 0 to 7 (8 sessions
/// per partition).  Hardware builds may use a different cap.
///
/// `#[repr(transparent)]` over `u16`.
///
/// # Conversions
///
/// ```
/// # use tsra_fw_hsm_pal_traits::HsmSessId;
/// let sid = HsmSessId::from(5u16);
/// assert_eq!(u16::from(sid), 5);
/// ```
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HsmSessId(u16);

impl From<u16> for HsmSessId {
    /// Wraps a raw `u16` slot index into a [`HsmSessId`].
    ///
    /// # Parameters
    ///
    /// - `v` — session-table slot index.
    ///
    /// # Returns
    ///
    /// A [`HsmSessId`] wrapping `v`.
    #[inline]
    fn from(v: u16) -> Self {
        Self(v)
    }
}

impl From<HsmSessId> for u16 {
    /// Unwraps a [`HsmSessId`] to its raw `u16` slot index.
    ///
    /// # Parameters
    ///
    /// - `id` — session identifier.
    ///
    /// # Returns
    ///
    /// The underlying `u16` slot index.
    #[inline]
    fn from(id: HsmSessId) -> Self {
        id.0
    }
}

impl HsmSessId {
    /// Returns the [`HsmSessionRole`] implied by this session's slot index.
    ///
    /// Slot 0 is reserved for [`HsmSessionRole::CryptoOfficer`] sessions;
    /// slots 1..=7 are [`HsmSessionRole::CryptoUser`].  The mapping is
    /// pinned by the session-establishment protocol and matches the
    /// PSK-based role gating performed in `OpenSessionInit`.
    ///
    /// # Returns
    ///
    /// [`HsmSessionRole::CryptoOfficer`] if this is slot 0, otherwise
    /// [`HsmSessionRole::CryptoUser`].
    #[inline]
    pub fn role(self) -> HsmSessionRole {
        if self.0 == 0 {
            HsmSessionRole::CryptoOfficer
        } else {
            HsmSessionRole::CryptoUser
        }
    }
}

// =============================================================================
// Session establishment protocol constants
// =============================================================================

/// Length of the public AppId prefix in a partition PSK.
pub const APP_ID_LEN: usize = 16;

/// Length in bytes of a partition pre-shared key (PSK).
///
/// Both the CO and CU PSKs are exactly this length.  Mixed into the
/// HPKE `mode_auth_psk` key schedule by the session-establishment
/// handshake.
pub const PSK_LEN: usize = 32;

/// Well-known default Crypto Officer (CO) PSK.
///
/// Returned by [`HsmPartitionManager::part_psk`] for `psk_id = 0`
/// until [`HsmPartitionManager::part_psk_set`] is called to rotate
/// it.  Public by design so partitions are usable immediately at
/// bring-up.  Deployment runbooks MUST rotate this before exposing the
/// partition to untrusted traffic.
pub const DEFAULT_PSK_CO: [u8; PSK_LEN] = [
    0x54, 0x53, 0x52, 0x41, 0x2d, 0x44, 0x45, 0x46, 0x41, 0x55, 0x4c, 0x54, 0x2d, 0x43, 0x4f, 0x2d,
    0x50, 0x53, 0x4b, 0x2d, 0x76, 0x31, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d,
];

/// Well-known default Crypto User (CU) PSK.
///
/// Returned by [`HsmPartitionManager::part_psk`] for `psk_id = 1`
/// until [`HsmPartitionManager::part_psk_set`] is called to rotate
/// it.  See [`DEFAULT_PSK_CO`] for the security caveat.
pub const DEFAULT_PSK_CU: [u8; PSK_LEN] = [
    0x54, 0x53, 0x52, 0x41, 0x2d, 0x44, 0x45, 0x46, 0x41, 0x55, 0x4c, 0x54, 0x2d, 0x43, 0x55, 0x2d,
    0x50, 0x53, 0x4b, 0x2d, 0x76, 0x31, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d, 0x2d,
];

/// Length in bytes of the per-handshake `seed` value supplied by the
/// VM in `OpenSessionInit`.
///
/// Mixed into the HPKE `info` field so the derived `exported` value
/// also depends on host-supplied entropy, and re-used at
/// `OpenSessionFinish` time to derive `BK_SESSION` for wrapping the
/// resumable `bmk_session` blob.
pub const SESSION_SEED_LEN: usize = 32;

/// Maximum size of the opaque Pending handshake-state blob stored in a
/// session slot between `OpenSessionInit` and `OpenSessionFinish`.
///
/// Holds: HPKE `exported` (48 B) ‖ `pk_init` (97 B, SEC1
/// uncompressed P-384) ‖ `pk_resp` (97 B, same) ‖ `session_type`
/// (1 B), suite (1 B), and credential binding (68 B). The host-supplied
/// `seed` is no longer part of the Pending
/// blob (it arrives encrypted in `OpenSessionFinish` instead).
/// Rounded up to give a small implementation margin.
pub const SESSION_PENDING_BLOB_MAX: usize = 320;

/// Session credential binding: role (4 B), AppId (16 B), PSK SHA-384 (48 B).
pub const SESSION_AUTH_BINDING_LEN: usize = 4 + APP_ID_LEN + 48;

/// HPKE `info` string for the session-establishment handshake.
///
/// Mixed into the HPKE key schedule on both sides; ensures the derived
/// `exported` value is domain-separated from any other HPKE usage.
///
/// `v2` introduces the `suite_id` byte appended after `psk_id` and
/// `session_type` so that any attempt to downgrade the negotiated
/// cryptographic suite would produce a different `exported` and fail
/// the Phase-1 confirm MAC.  The bump from `v1` → `v2` also
/// domain-separates against any pre-suite-id firmware/host pairings.
pub const SESSION_HPKE_INFO: &[u8] = b"tsra-session-v2";

/// HPKE exporter context for the session-establishment handshake.
pub const SESSION_HPKE_EXPORTER_CONTEXT: &[u8] = b"session-exporter";

/// HMAC label binding the Phase-1 (server-auth) confirm signature.
pub const SESSION_PHASE1_LABEL: &[u8] = b"phase1-confirm";

/// HMAC label binding the Phase-2 (client-auth) confirm signature.
pub const SESSION_PHASE2_LABEL: &[u8] = b"phase2-confirm";

/// HKDF-Expand label producing the per-session **param key** — a raw
/// 32-byte AES-256 key consumed by
/// [`tsra_fw_core_crypto_aead_envelope`] to AEAD-seal/open the
/// `seed_envelope` (in `OpenSessionFinish`) and per-parameter
/// envelopes carried by in-session commands such as `ChangePsk`.
///
/// Derived for every promoted TBOR session regardless of session
/// type.
pub const SESSION_PARAM_KEY_LABEL: &[u8] = b"tsra-session-param-v1";

/// HKDF-Expand label producing the **VM→HSM** message-MAC key.
/// Derived only for Authenticated sessions.
pub const SESSION_MAC_TX_LABEL: &[u8] = b"tsra-session-mac-tx-v1";

/// HKDF-Expand label producing the **HSM→VM** message-MAC key.
/// Derived only for Authenticated sessions.
pub const SESSION_MAC_RX_LABEL: &[u8] = b"tsra-session-mac-rx-v1";

/// SP 800-108 KBKDF label for deriving `BK_SESSION` from `BK_BOOT` and
/// the host-supplied `seed`.  Domain-separates the session-resumption
/// wrap key from any other `BK_BOOT`-derived key.  Mirrors the MBOR
/// `SESSION_BK` label.
pub const SESSION_BK_LABEL: &[u8] = b"SESSION_BK";

/// Length of `BK_SESSION` in bytes — a raw 32-byte AES-256 key,
/// consumed by [`tsra_fw_core_crypto_aead_envelope`] to seal the
/// `bmk_session` envelope returned by `OpenSessionFinish`.
pub const SESSION_BK_LEN: usize = 32;

/// Fixed role tag MBOR-encoded into the `bmk_session` envelope
/// metadata (mirrors MBOR `MK` / `SMK` style labels).
pub const SESSION_BMK_KEY_LABEL: &[u8] = b"SMK";

/// Length in bytes of the per-session `param_key` (TBOR per-parameter
/// confidentiality).
///
/// Raw 32-byte AES-256 key, consumed by
/// [`tsra_fw_core_crypto_aead_envelope`].
pub const SESSION_PARAM_KEY_LEN: usize = 32;

/// Length in bytes of the per-session `masking_key` for TBOR
/// **SessionEx (CU/CO)** sessions.
///
/// 32 B AES-256-GCM key used by the `key_masking::aead`-based TBOR
/// masked-key system (masked keys are scoped and wrapped under AES-GCM).
/// Present for both CO and CU SessionEx blobs.  Distinct from the legacy
/// MBOR `Session`-blob masking key (an 80 B `aes32 ‖ hmac48` cbc key,
/// sized by the std-PAL-local `SESSION_MASKING_KEY_SIZE`), which the
/// `key_masking::cbc`-based MBOR masked-key system continues to use.
pub const SESSION_MASKING_KEY_LEN: usize = 32;

/// Length in bytes of each directional message-MAC key (HMAC-SHA-384).
///
/// Derived only for **Authenticated** sessions; one key per direction
/// (VM→HSM, HSM→VM).
pub const SESSION_MAC_DIR_KEY_LEN: usize = 48;
