// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Storage-independent PSK authentication policy engine.
//!
//! The caller owns one policy context per authentication-policy scope, supplies
//! a monotonic clock, and serializes access to the context. This module does not
//! select partition/role scope, persist state, or authenticate credentials.
//! Session handlers must consult [`PskPolicy::can_authenticate`] before doing
//! authentication work and record the eventual credential-validation outcome.
//!
//! Clock values and deadlines use ticks in the same non-wrapping clock domain;
//! the clock frequency must remain constant for that domain's lifetime.

use core::num::NonZeroU64;

use tsra_fw_hsm_pal_traits::HsmError;
use tsra_fw_hsm_pal_traits::HsmResult;
use tsra_fw_hsm_pal_traits::PskPolicySnapshot;
/// Duration in seconds represented by one delay-factor increment.
pub use tsra_fw_hsm_pal_traits::PSK_POLICY_DELAY_SECS as DELAY_INCREMENT_SECS;
/// Counted authentication failures before increasing the delay factor.
pub use tsra_fw_hsm_pal_traits::PSK_POLICY_FAILURES_PER_LEVEL as FAILURES_PER_DELAY_LEVEL;
/// Largest delay factor; the next increment rolls the policy back to Ready.
pub use tsra_fw_hsm_pal_traits::PSK_POLICY_MAX_FACTOR as MAX_DELAY_FACTOR;

/// PSK authentication-policy state.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PskPolicyState {
    /// No accumulated authentication failures.
    #[default]
    Ready,
    /// Failures have accumulated; authentication must respect the deadline.
    Lockout,
}

/// Authentication-failure accounting and progressive lockout context.
///
/// The failure count stays below [`FAILURES_PER_DELAY_LEVEL`], and the delay
/// factor stays at or below [`MAX_DELAY_FACTOR`]. Private fields keep callers
/// from constructing a context that violates these invariants.
///
/// `Lockout` does not always mean authentication is currently blocked: factor
/// zero has no delay, and an expired deadline permits the next attempt without
/// clearing the accumulated penalties.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PskPolicy {
    state: PskPolicyState,
    failed_attempts: u16,
    delay_factor: u16,
    lockout_deadline: u64,
}

impl PskPolicy {
    /// Restore a validated snapshot from the current boot's clock domain.
    /// Boot recovery must rebase a retained snapshot before it is used.
    pub fn from_snapshot(snapshot: PskPolicySnapshot) -> Self {
        Self {
            state: if snapshot.is_ready() {
                PskPolicyState::Ready
            } else {
                PskPolicyState::Lockout
            },
            failed_attempts: snapshot.failed_attempts(),
            delay_factor: snapshot.delay_factor(),
            lockout_deadline: snapshot.lockout_deadline(),
        }
    }

    /// Encode the context without exposing mutable policy fields.
    pub fn snapshot(&self) -> HsmResult<PskPolicySnapshot> {
        PskPolicySnapshot::new(
            self.state == PskPolicyState::Ready,
            self.failed_attempts,
            self.delay_factor,
            self.lockout_deadline,
        )
    }

    /// Current policy state.
    pub fn state(&self) -> PskPolicyState {
        self.state
    }

    /// Failures counted since the last delay-factor increment or reset.
    pub fn failed_attempts(&self) -> u16 {
        self.failed_attempts
    }

    /// Current delay in multiples of [`DELAY_INCREMENT_SECS`].
    pub fn delay_factor(&self) -> u16 {
        self.delay_factor
    }

    /// Earliest permitted authentication time in the caller's clock ticks.
    pub fn lockout_deadline(&self) -> u64 {
        self.lockout_deadline
    }

    /// Whether authentication is permitted at `now_ticks`.
    ///
    /// Expiry permits another attempt but does not reset counters or state.
    /// The exact deadline is inclusive: an attempt at the deadline is allowed.
    pub fn can_authenticate(&self, now_ticks: u64) -> bool {
        self.state == PskPolicyState::Ready || now_ticks >= self.lockout_deadline
    }

    /// Record a credential-validation failure and calculate the next deadline.
    ///
    /// Every permitted failure incurs the current delay, including failures
    /// between threshold crossings. The threshold-crossing failure uses the
    /// newly increased factor. Incrementing beyond [`MAX_DELAY_FACTOR`] resets
    /// the complete context to Ready instead.
    ///
    /// `Ok(())` means the failure was recorded, not that authentication passed.
    /// The caller must still return its credential-validation error.
    ///
    /// Returns [`HsmError::LoginFailed`] while the existing deadline is active.
    /// Returns [`HsmError::InternalError`] if the delay or deadline cannot be
    /// represented in `u64` ticks. On either error the context is unchanged;
    /// the caller must propagate the error rather than admitting the session.
    pub fn record_failure(
        &mut self,
        now_ticks: u64,
        ticks_per_second: NonZeroU64,
    ) -> HsmResult<()> {
        if !self.can_authenticate(now_ticks) {
            return Err(HsmError::LoginFailed);
        }

        let mut next = *self;
        next.state = PskPolicyState::Lockout;
        next.failed_attempts += 1;

        if next.failed_attempts == FAILURES_PER_DELAY_LEVEL {
            next.failed_attempts = 0;
            next.delay_factor += 1;
            if next.delay_factor > MAX_DELAY_FACTOR {
                *self = Self::default();
                return Ok(());
            }
        }

        let delay_ticks = u64::from(next.delay_factor)
            .checked_mul(DELAY_INCREMENT_SECS)
            .and_then(|seconds| seconds.checked_mul(ticks_per_second.get()))
            .ok_or(HsmError::InternalError)?;
        next.lockout_deadline = now_ticks
            .checked_add(delay_ticks)
            .ok_or(HsmError::InternalError)?;

        *self = next;
        Ok(())
    }

    /// Record successful credential validation and clear all penalties.
    ///
    /// Returns [`HsmError::LoginFailed`] without changing the context if the
    /// deadline is still active. Correct credentials do not bypass lockout.
    pub fn record_success(&mut self, now_ticks: u64) -> HsmResult<()> {
        if !self.can_authenticate(now_ticks) {
            return Err(HsmError::LoginFailed);
        }

        *self = Self::default();
        Ok(())
    }
}
