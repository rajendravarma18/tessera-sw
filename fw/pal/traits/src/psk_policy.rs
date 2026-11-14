// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Validated, platform-independent representation of retained PSK policy state.

use core::num::NonZeroU64;

use crate::HsmError;
use crate::HsmResult;

/// Size of the legacy partition lockout slot, including its reserved byte.
pub const PSK_POLICY_LEN: usize = 14;
/// Counted failures per delay level.
pub const PSK_POLICY_FAILURES_PER_LEVEL: u16 = 1_000;
/// Seconds per delay-factor increment.
pub const PSK_POLICY_DELAY_SECS: u64 = 60;
/// Maximum delay factor before rollover to Ready.
pub const PSK_POLICY_MAX_FACTOR: u16 = 32;

/// Validated snapshot: state, reserved byte, LE factor, LE failure count,
/// and LE deadline. All-zero bytes represent Ready.
///
/// The byte representation has no Rust enum discriminants or implicit padding,
/// so retained RAM can be validated before constructing a runtime policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PskPolicySnapshot([u8; PSK_POLICY_LEN]);

impl PskPolicySnapshot {
    /// Validate stored bytes. Corrupt state is an error, never a policy reset.
    pub fn from_bytes(bytes: &[u8]) -> HsmResult<Self> {
        let mut snapshot = Self(bytes.try_into().map_err(|_| HsmError::InternalError)?);
        // This byte was implicit padding in the legacy PinPolicy structure.
        // It carries no state and may be unspecified in an older firmware image.
        snapshot.0[1] = 0;
        if snapshot.0[0] > 1
            || snapshot.failed_attempts() >= PSK_POLICY_FAILURES_PER_LEVEL
            || snapshot.delay_factor() > PSK_POLICY_MAX_FACTOR
            || (snapshot.is_ready() && snapshot.0 != [0; PSK_POLICY_LEN])
            || (!snapshot.is_ready()
                && snapshot.delay_factor() == 0
                && snapshot.failed_attempts() == 0)
        {
            return Err(HsmError::InternalError);
        }
        Ok(snapshot)
    }

    /// Construct a snapshot, applying the same invariants as restoration.
    pub fn new(
        ready: bool,
        failed_attempts: u16,
        delay_factor: u16,
        deadline: u64,
    ) -> HsmResult<Self> {
        let mut bytes = [0; PSK_POLICY_LEN];
        bytes[0] = u8::from(!ready);
        bytes[2..4].copy_from_slice(&delay_factor.to_le_bytes());
        bytes[4..6].copy_from_slice(&failed_attempts.to_le_bytes());
        bytes[6..14].copy_from_slice(&deadline.to_le_bytes());
        Self::from_bytes(&bytes)
    }

    /// Bytes suitable for the partition property or retained storage slot.
    pub fn as_bytes(&self) -> &[u8; PSK_POLICY_LEN] {
        &self.0
    }

    /// Whether this snapshot represents Ready.
    pub fn is_ready(&self) -> bool {
        self.0[0] == 0
    }

    /// Failures counted at the current delay level.
    pub fn failed_attempts(&self) -> u16 {
        u16::from_le_bytes([self.0[4], self.0[5]])
    }

    /// Delay factor.
    pub fn delay_factor(&self) -> u16 {
        u16::from_le_bytes([self.0[2], self.0[3]])
    }

    /// Deadline in the current boot's monotonic clock domain.
    pub fn lockout_deadline(&self) -> u64 {
        u64::from_le_bytes([
            self.0[6], self.0[7], self.0[8], self.0[9], self.0[10], self.0[11], self.0[12],
            self.0[13],
        ])
    }

    /// Restart the full current delay in a new boot's clock domain.
    ///
    /// Warm/crash/IDFU recovery preserves counters but cannot compare the old
    /// deadline against a restarted SysTick. Even a previously expired delay
    /// is restarted conservatively. Ready remains the canonical all-zero state.
    pub fn restart_delay(&self, now_ticks: u64, ticks_per_second: NonZeroU64) -> HsmResult<Self> {
        if self.is_ready() {
            return Ok(*self);
        }
        let delay = u64::from(self.delay_factor())
            .checked_mul(PSK_POLICY_DELAY_SECS)
            .and_then(|seconds| seconds.checked_mul(ticks_per_second.get()))
            .ok_or(HsmError::InternalError)?;
        let deadline = now_ticks
            .checked_add(delay)
            .ok_or(HsmError::InternalError)?;
        Self::new(false, self.failed_attempts(), self.delay_factor(), deadline)
    }
}
