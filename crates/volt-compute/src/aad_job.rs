//! Budgeted reverse-mode Greeks. All sensitivities are per unit of the quoted
//! annual fraction (vega/rho/dividend) or per year of calendar passage (theta).
use crate::{
    ComputeError, ExecutionTarget, OptionRow, Precision, Session,
    aad::{AadResult, differentiate_rows},
    budget::{MemoryBudget, MemoryPlan, WorkloadMemoryModel},
    chunked::{CompensatedSum, execution_guard, validate_job},
};
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Serialize)]
pub struct ParallelSensitivities {
    pub option_position_value: f64,
    /// Simultaneous +1 currency-unit spot change to every row's underlying.
    pub delta: f64,
    pub vega: f64,
    pub rho: f64,
    pub theta: f64,
    pub dividend_sensitivity: f64,
}

#[derive(Debug, Serialize)]
pub struct AadJobResult {
    pub values: AadResult,
    /// Optional d²P/dS² from a second-order forward jet, in input order.
    pub gamma: Option<Vec<f64>>,
    /// Weighted sum of row gammas under simultaneous independent spot bumps.
    /// This does not include cross-gamma between different underlyings.
    pub parallel_gamma: Option<f64>,
    pub parallel_sensitivities: ParallelSensitivities,
    pub plan: MemoryPlan,
    pub completed_chunks: usize,
    /// Includes planning, validation, device setup, upload, forward, backward,
    /// synchronization, all readback, and stable host aggregation.
    pub job_seconds: f64,
}

pub fn plan_aad_rows(
    rows: &[OptionRow],
    target: ExecutionTarget,
    precision: Precision,
    budget: &MemoryBudget,
) -> Result<MemoryPlan, ComputeError> {
    plan_aad_rows_with_gamma(rows, target, precision, budget, false)
}

pub fn plan_aad_rows_with_gamma(
    rows: &[OptionRow],
    target: ExecutionTarget,
    precision: Precision,
    budget: &MemoryBudget,
    include_gamma: bool,
) -> Result<MemoryPlan, ComputeError> {
    let model = if include_gamma {
        WorkloadMemoryModel::BLACK_SCHOLES_AAD_GAMMA
    } else {
        WorkloadMemoryModel::BLACK_SCHOLES_AAD
    };
    let plan = budget.plan_for_workload(rows, target, precision, model)?;
    validate_job(rows, precision)?;
    Ok(plan)
}

pub fn differentiate_job(
    rows: &[OptionRow],
    target: ExecutionTarget,
    precision: Precision,
    budget: &MemoryBudget,
) -> Result<AadJobResult, ComputeError> {
    differentiate_job_with_gamma(rows, target, precision, budget, false)
}

pub fn differentiate_job_with_gamma(
    rows: &[OptionRow],
    target: ExecutionTarget,
    precision: Precision,
    budget: &MemoryBudget,
    include_gamma: bool,
) -> Result<AadJobResult, ComputeError> {
    let started = Instant::now();
    let plan = plan_aad_rows_with_gamma(rows, target, precision, budget, include_gamma)?;
    let _guard = execution_guard()?;
    let session = Session::new(target, precision, plan.chunk_rows)?;
    let mut result = AadResult {
        prices: Vec::with_capacity(rows.len()),
        delta: Vec::with_capacity(rows.len()),
        vega: Vec::with_capacity(rows.len()),
        rho: Vec::with_capacity(rows.len()),
        theta: Vec::with_capacity(rows.len()),
        dividend_sensitivity: Vec::with_capacity(rows.len()),
        max_sampled_allocator_live_bytes: None,
        max_sampled_allocator_reserved_bytes: None,
    };
    let mut sums: [CompensatedSum; 6] = std::array::from_fn(|_| CompensatedSum::default());
    let mut gamma = include_gamma.then(|| Vec::with_capacity(rows.len()));
    let mut gamma_sum = CompensatedSum::default();
    let mut completed_chunks = 0;
    for chunk in rows.chunks(plan.chunk_rows) {
        let values = differentiate_rows(&session, chunk)?;
        for (i, row) in chunk.iter().enumerate() {
            let metrics = [
                values.prices[i],
                values.delta[i],
                values.vega[i],
                values.rho[i],
                values.theta[i],
                values.dividend_sensitivity[i],
            ];
            for (sum, value) in sums.iter_mut().zip(metrics) {
                let weighted = value * row.quantity as f64 * row.multiplier as f64;
                if !weighted.is_finite() {
                    return Err(ComputeError::Execution(format!(
                        "{}: weighted Greek overflow",
                        row.row_id
                    )));
                }
                sum.add(weighted);
            }
        }
        result.max_sampled_allocator_live_bytes = max_usage(
            result.max_sampled_allocator_live_bytes,
            values.max_sampled_allocator_live_bytes,
        );
        result.max_sampled_allocator_reserved_bytes = max_usage(
            result.max_sampled_allocator_reserved_bytes,
            values.max_sampled_allocator_reserved_bytes,
        );
        if let Some(output) = &mut gamma {
            let jet = crate::second_order::spot_second_order(&session, chunk)?;
            for (row, value) in chunk.iter().zip(&jet.gamma) {
                let weighted = value * row.quantity as f64 * row.multiplier as f64;
                if !weighted.is_finite() {
                    return Err(ComputeError::Execution(format!(
                        "{}: weighted gamma overflow",
                        row.row_id
                    )));
                }
                gamma_sum.add(weighted);
            }
            result.max_sampled_allocator_live_bytes = max_usage(
                result.max_sampled_allocator_live_bytes,
                jet.max_sampled_allocator_live_bytes,
            );
            result.max_sampled_allocator_reserved_bytes = max_usage(
                result.max_sampled_allocator_reserved_bytes,
                jet.max_sampled_allocator_reserved_bytes,
            );
            output.extend(jet.gamma);
        }
        if result
            .max_sampled_allocator_reserved_bytes
            .is_some_and(|n| n > budget.compute_bytes)
        {
            session.device().memory_cleanup();
            return Err(ComputeError::BudgetExceeded("AAD allocator reservation exceeds compute budget; lower the row cap or increase the budget".into()));
        }
        result.prices.extend(values.prices);
        result.delta.extend(values.delta);
        result.vega.extend(values.vega);
        result.rho.extend(values.rho);
        result.theta.extend(values.theta);
        result
            .dividend_sensitivity
            .extend(values.dividend_sensitivity);
        completed_chunks += 1;
    }
    session.synchronize()?;
    session.device().memory_cleanup();
    let [price, delta, vega, rho, theta, dividend] = sums.map(CompensatedSum::value);
    if [price, delta, vega, rho, theta, dividend]
        .iter()
        .any(|n| !n.is_finite())
    {
        return Err(ComputeError::Execution(
            "portfolio Greek aggregate overflow".into(),
        ));
    }
    let parallel_gamma = if include_gamma {
        let value = gamma_sum.value();
        if !value.is_finite() {
            return Err(ComputeError::Execution(
                "portfolio gamma aggregate overflow".into(),
            ));
        }
        Some(value)
    } else {
        None
    };
    Ok(AadJobResult {
        values: result,
        gamma,
        parallel_gamma,
        parallel_sensitivities: ParallelSensitivities {
            option_position_value: price,
            delta,
            vega,
            rho,
            theta,
            dividend_sensitivity: dividend,
        },
        plan,
        completed_chunks,
        job_seconds: started.elapsed().as_secs_f64(),
    })
}

fn max_usage(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}
