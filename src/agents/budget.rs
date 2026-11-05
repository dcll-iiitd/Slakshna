//! Budget enforcement (A1): tokens, GPU-seconds and wall clock, with a hard stop.
//!
//! A `Budget` is charged before work is committed to; once any dimension would
//! be exceeded the charge is refused and the tracker latches into the stopped
//! state, so concurrent users of the same budget cannot keep spending.

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    pub tokens: u64,
    pub gpu_seconds: u64,
    pub wall_clock_secs: u64,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BudgetError {
    #[error("token budget exceeded (requested {requested}, remaining {remaining})")]
    Tokens { requested: u64, remaining: u64 },
    #[error("GPU-seconds budget exceeded (requested {requested}, remaining {remaining})")]
    GpuSeconds { requested: u64, remaining: u64 },
    #[error("wall-clock budget exhausted")]
    WallClock,
    #[error("budget stopped")]
    Stopped,
}

#[derive(Debug)]
struct Inner {
    tokens_left: u64,
    gpu_left: u64,
    stopped: bool,
}

#[derive(Debug)]
pub struct BudgetTracker {
    inner: Mutex<Inner>,
    deadline: Instant,
}

impl BudgetTracker {
    pub fn new(budget: Budget) -> Self {
        BudgetTracker {
            inner: Mutex::new(Inner {
                tokens_left: budget.tokens,
                gpu_left: budget.gpu_seconds,
                stopped: false,
            }),
            deadline: Instant::now() + Duration::from_secs(budget.wall_clock_secs),
        }
    }

    fn guard(&self) -> Result<std::sync::MutexGuard<'_, Inner>, BudgetError> {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if Instant::now() >= self.deadline {
            g.stopped = true;
            return Err(BudgetError::WallClock);
        }
        if g.stopped {
            return Err(BudgetError::Stopped);
        }
        Ok(g)
    }

    pub fn charge_tokens(&self, n: u64) -> Result<(), BudgetError> {
        let mut g = self.guard()?;
        if n > g.tokens_left {
            let remaining = g.tokens_left;
            g.stopped = true;
            return Err(BudgetError::Tokens { requested: n, remaining });
        }
        g.tokens_left -= n;
        Ok(())
    }

    pub fn charge_gpu_seconds(&self, n: u64) -> Result<(), BudgetError> {
        let mut g = self.guard()?;
        if n > g.gpu_left {
            let remaining = g.gpu_left;
            g.stopped = true;
            return Err(BudgetError::GpuSeconds { requested: n, remaining });
        }
        g.gpu_left -= n;
        Ok(())
    }

    /// Fails once the wall clock is spent or the budget has been stopped.
    pub fn check(&self) -> Result<(), BudgetError> {
        self.guard().map(|_| ())
    }

    /// Time left on the wall clock; callers use it as a hard process timeout.
    pub fn wall_remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }

    /// Emergency / manual stop (A7 kill switch).
    pub fn stop(&self) {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).stopped = true;
    }

    pub fn remaining(&self) -> (u64, u64) {
        let g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        (g.tokens_left, g.gpu_left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b() -> Budget { Budget { tokens: 100, gpu_seconds: 10, wall_clock_secs: 60 } }

    #[test]
    fn charges_until_exhausted_then_latches() {
        let t = BudgetTracker::new(b());
        assert!(t.charge_tokens(60).is_ok());
        assert_eq!(t.charge_tokens(50), Err(BudgetError::Tokens { requested: 50, remaining: 40 }));
        // A hard stop: even a charge that would fit is now refused.
        assert_eq!(t.charge_tokens(1), Err(BudgetError::Stopped));
        assert_eq!(t.charge_gpu_seconds(1), Err(BudgetError::Stopped));
    }

    #[test]
    fn gpu_budget_enforced_independently() {
        let t = BudgetTracker::new(b());
        assert!(t.charge_gpu_seconds(10).is_ok());
        assert!(matches!(t.charge_gpu_seconds(1), Err(BudgetError::GpuSeconds { .. })));
    }

    #[test]
    fn wall_clock_expires() {
        let t = BudgetTracker::new(Budget { tokens: 1, gpu_seconds: 1, wall_clock_secs: 0 });
        assert_eq!(t.check(), Err(BudgetError::WallClock));
        assert_eq!(t.wall_remaining(), Duration::ZERO);
    }

    #[test]
    fn manual_stop_blocks_everything() {
        let t = BudgetTracker::new(b());
        t.stop();
        assert_eq!(t.check(), Err(BudgetError::Stopped));
    }

    #[test]
    fn exact_spend_is_allowed() {
        let t = BudgetTracker::new(b());
        assert!(t.charge_tokens(100).is_ok());
        assert_eq!(t.remaining(), (0, 10));
    }
}
