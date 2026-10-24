use crate::{
    ComputeError, ExecutionTarget, OptionRow, Precision, ResidentOptions, Session,
    budget::{MemoryBudget, MemoryPlan},
};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::Instant;

// Budget estimates are per job. Serialize this API's jobs within one process so
// several individually admitted jobs do not multiply the estimated live set.
// Raw sessions and other processes remain outside this coordination boundary.
static BUDGETED_EXECUTION: Mutex<()> = Mutex::new(());

pub(crate) fn execution_guard() -> Result<std::sync::MutexGuard<'static, ()>, ComputeError> {
    BUDGETED_EXECUTION.lock().map_err(|_| {
        ComputeError::Execution("budgeted execution lock poisoned by an earlier failure".into())
    })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkedStats {
    pub completed_chunks: usize,
    pub packing_and_upload_seconds: f64,
    pub evaluation_seconds: f64,
    pub readback_seconds: f64,
    pub max_sampled_allocator_live_bytes: Option<u64>,
    pub max_sampled_allocator_reserved_bytes: Option<u64>,
}

pub struct ChunkedValuation {
    /// Same order as the input rows; no rounding to ledger money units.
    pub prices: Vec<f64>,
    pub portfolio_value: f64,
    /// Diagnostic only: f32 device reduction grouping can change with chunk size.
    pub device_reduced_value: Option<f64>,
    pub plan: MemoryPlan,
    pub stats: ChunkedStats,
}

/// Bounded sequential dispatch for the non-autodiff Black–Scholes kernel.
/// Every chunk is synchronized, read back, and dropped before the next upload.
/// Callers retain their own inputs; result storage is admitted by the host budget.
pub fn value_rows(
    rows: &[OptionRow],
    target: ExecutionTarget,
    precision: Precision,
    budget: &MemoryBudget,
) -> Result<ChunkedValuation, ComputeError> {
    let plan = plan_rows(rows, target, precision, budget)?;
    let _execution_guard = execution_guard()?;
    let session = Session::new(target, precision, plan.chunk_rows)?;
    let mut prices = Vec::with_capacity(rows.len());
    let mut sum = CompensatedSum::default();
    let mut device_sum = CompensatedSum::default();
    let mut finite_device_reductions = true;
    let mut stats = ChunkedStats::default();
    for chunk in rows.chunks(plan.chunk_rows) {
        let started = Instant::now();
        let resident = ResidentOptions::upload(&session, chunk)?;
        session.synchronize()?;
        stats.packing_and_upload_seconds += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let marks = resident.prices();
        // Materialize marks independently of the optional weighted diagnostic.
        // A lazy backend must not choose price arithmetic based on its consumers.
        #[cfg(feature = "fusion")]
        session.synchronize()?;
        let total = resident.portfolio_value(marks.clone());
        session.synchronize()?;
        stats.evaluation_seconds += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let values = session.read_vector(&marks)?;
        let aggregate = session.read_vector(&total)?;
        stats.readback_seconds += started.elapsed().as_secs_f64();
        if values.len() != chunk.len()
            || aggregate.len() != 1
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(ComputeError::Execution(format!(
                "invalid numerical result in chunk {}",
                stats.completed_chunks
            )));
        }
        if let Some(usage) = session.device().memory_pool_usage() {
            stats.max_sampled_allocator_live_bytes = Some(
                stats
                    .max_sampled_allocator_live_bytes
                    .unwrap_or(0)
                    .max(usage.bytes_in_use),
            );
            stats.max_sampled_allocator_reserved_bytes = Some(
                stats
                    .max_sampled_allocator_reserved_bytes
                    .unwrap_or(0)
                    .max(usage.bytes_reserved),
            );
            if usage.bytes_reserved > budget.compute_bytes {
                drop((resident, marks, total));
                session.device().memory_cleanup();
                return Err(ComputeError::BudgetExceeded(format!(
                    "device allocator reserved {} bytes, above configured {} bytes; lower the row cap or increase the budget",
                    usage.bytes_reserved, budget.compute_bytes
                )));
            }
        }
        // Aggregate marks in original input order with f64 arithmetic, so the
        // scheduling budget does not choose a different reduction tree for P&L.
        for (row, price) in chunk.iter().zip(&values) {
            let value = price * row.quantity as f64 * row.multiplier as f64;
            if !value.is_finite() {
                return Err(ComputeError::Execution(format!(
                    "{}: weighted value overflow",
                    row.row_id
                )));
            }
            sum.add(value);
        }
        if aggregate[0].is_finite() {
            device_sum.add(aggregate[0]);
        } else {
            finite_device_reductions = false;
        }
        prices.extend(values);
        stats.completed_chunks += 1;
        // Explicitly release the last chunk's handles before the next upload.
        drop((resident, marks, total));
    }
    session.synchronize()?;
    // Allocator pages may be cached. Cleanup is a best-effort release, not an OS
    // memory limit and not a promise that another session's pages will be freed.
    session.device().memory_cleanup();
    let portfolio_value = sum.value();
    if !portfolio_value.is_finite() {
        return Err(ComputeError::Execution(
            "portfolio aggregate overflow".into(),
        ));
    }
    Ok(ChunkedValuation {
        prices,
        portfolio_value,
        device_reduced_value: if finite_device_reductions {
            let value = device_sum.value();
            value.is_finite().then_some(value)
        } else {
            None
        },
        plan,
        stats,
    })
}

/// Validate the complete job and its admission plan without opening a device.
pub fn plan_rows(
    rows: &[OptionRow],
    target: ExecutionTarget,
    precision: Precision,
    budget: &MemoryBudget,
) -> Result<MemoryPlan, ComputeError> {
    let plan = budget.plan(rows, target, precision)?;
    validate_job(rows, precision)?;
    Ok(plan)
}

pub(crate) fn validate_job(rows: &[OptionRow], precision: Precision) -> Result<(), ComputeError> {
    // Validate before any GPU allocation, including duplicates straddling chunks.
    let mut identities = Vec::with_capacity(rows.len());
    for row in rows {
        row.validate_for_precision(precision)?;
        identities.push(row.row_id.as_str());
    }
    identities.sort_unstable();
    if identities.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ComputeError::InvalidInput(
            "duplicate row identity across the valuation job".into(),
        ));
    }
    Ok(())
}

#[derive(Default)]
pub(crate) struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }
    pub(crate) fn value(self) -> f64 {
        self.sum + self.correction
    }
}
