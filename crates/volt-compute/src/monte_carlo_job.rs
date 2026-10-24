//! Bounded endpoint Monte Carlo risk. Common random numbers couple numerical
//! estimators across rows; their portfolio SE is estimator uncertainty, not a
//! model of the joint market P&L distribution across different underlyings.
use crate::{
    ComputeError, ExecutionTarget, OptionRow, Precision, Session,
    budget::MemoryBudget,
    chunked::{execution_guard, validate_job},
    monte_carlo::{GreekMoments, MonteCarloResult, MonteCarloRow, monte_carlo_rows, normal_at},
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    pub target: ExecutionTarget,
    pub precision: Precision,
    pub seed: u64,
    /// Independent antithetic pairs, each containing two terminal outcomes.
    pub antithetic_pairs: usize,
    pub budget: MemoryBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloPlan {
    pub schema: String,
    pub option_rows: usize,
    pub antithetic_pairs: usize,
    pub pairs_per_batch: usize,
    pub batches: usize,
    pub expanded_outcomes_per_batch: usize,
    pub total_option_outcomes: u64,
    pub estimated_compute_bytes: u64,
    pub estimated_host_working_bytes: u64,
    pub estimated_compute_bytes_per_outcome: u64,
    pub compute_admission_limit_bytes: u64,
    pub host_admission_limit_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct MonteCarloJobResult {
    pub result: MonteCarloResult,
    pub plan: MonteCarloPlan,
    pub completed_batches: usize,
    /// Complete admission, device initialization, random generation, upload,
    /// differentiation, readback and pair-moment aggregation. No analytic checks.
    pub job_seconds: f64,
}

pub fn plan_monte_carlo(
    rows: &[OptionRow],
    config: &MonteCarloConfig,
) -> Result<MonteCarloPlan, ComputeError> {
    let budget = &config.budget;
    if config.antithetic_pairs < 2
        || rows.is_empty()
        || budget.headroom_percent >= 100
        || budget.compute_bytes == 0
        || budget.host_working_bytes == 0
        || budget.max_batch_rows == 0
    {
        return Err(ComputeError::InvalidInput("Monte Carlo needs nonempty rows, at least two independent pairs, positive budgets/cap and headroom below 100%".into()));
    }
    if config.target == ExecutionTarget::Metal && config.precision == Precision::F64 {
        return Err(ComputeError::Unsupported(
            "Metal does not provide native f64 arithmetic".into(),
        ));
    }
    let row_count = rows.len() as u64;
    let outcomes_per_pair = row_count.checked_mul(2).ok_or_else(overflow)?;
    let total_option_outcomes = outcomes_per_pair
        .checked_mul(config.antithetic_pairs as u64)
        .ok_or_else(overflow)?;
    let scalar_bytes = if config.precision == Precision::F64 {
        8
    } else {
        4
    };
    // Terminal GBM graph, five leaf adjoints, LR gamma and forward/backward
    // workspaces. This model does not admit a time-stepped simulation graph.
    let compute_per_outcome = 256 * scalar_bytes;
    let compute_limit = usable(budget.compute_bytes, budget.headroom_percent);
    let host_limit = usable(budget.host_working_bytes, budget.headroom_percent);
    let compute_outcomes = compute_limit
        .checked_sub(budget.compute_reserve_bytes)
        .ok_or_else(|| {
            ComputeError::BudgetExceeded(
                "Monte Carlo reserve consumes compute admission limit".into(),
            )
        })?
        / compute_per_outcome;
    let max_id_bytes = rows
        .iter()
        .map(|row| row.row_id.len() as u64)
        .max()
        .unwrap_or(0);
    // Global row moments/result, validation keys, merge temporaries and IDs.
    // Caller-owned source rows and report serialization are excluded.
    let global_host = row_count
        .checked_mul(2048_u64.checked_add(max_id_bytes).ok_or_else(overflow)?)
        .and_then(|n| n.checked_add(8192))
        .ok_or_else(overflow)?;
    let shared = matches!(config.target, ExecutionTarget::Cpu | ExecutionTarget::Metal);
    let host_fixed = global_host
        .checked_add(if shared {
            budget.compute_reserve_bytes
        } else {
            0
        })
        .ok_or_else(overflow)?;
    let host_per_outcome = 256 + if shared { compute_per_outcome } else { 0 };
    let host_outcomes = host_limit.checked_sub(host_fixed).ok_or_else(|| {
        ComputeError::BudgetExceeded(
            "Monte Carlo row moments/shared reserve exceed host budget".into(),
        )
    })? / host_per_outcome;
    let pairs = (config.antithetic_pairs as u64)
        .min(compute_outcomes / outcomes_per_pair)
        .min(host_outcomes / outcomes_per_pair)
        .min(budget.max_batch_rows as u64 / outcomes_per_pair);
    if pairs == 0 {
        return Err(ComputeError::BudgetExceeded("cannot admit one antithetic pair across all option rows; increase budget/cap or reduce the portfolio".into()));
    }
    let expanded = pairs.checked_mul(outcomes_per_pair).ok_or_else(overflow)?;
    let plan = MonteCarloPlan {
        schema: "volt.compute.monte_carlo_plan.v1".into(),
        option_rows: rows.len(),
        antithetic_pairs: config.antithetic_pairs,
        pairs_per_batch: pairs as usize,
        batches: config.antithetic_pairs.div_ceil(pairs as usize),
        expanded_outcomes_per_batch: expanded as usize,
        total_option_outcomes,
        estimated_compute_bytes: budget.compute_reserve_bytes + expanded * compute_per_outcome,
        estimated_host_working_bytes: host_fixed + expanded * host_per_outcome,
        estimated_compute_bytes_per_outcome: compute_per_outcome,
        compute_admission_limit_bytes: compute_limit,
        host_admission_limit_bytes: host_limit,
    };
    validate_job(rows, config.precision)?;
    Ok(plan)
}

pub fn run_monte_carlo(
    rows: &[OptionRow],
    config: &MonteCarloConfig,
) -> Result<MonteCarloJobResult, ComputeError> {
    let started = Instant::now();
    let plan = plan_monte_carlo(rows, config)?;
    let _guard = execution_guard()?;
    let session = Session::new(
        config.target,
        config.precision,
        plan.expanded_outcomes_per_batch,
    )?;
    let cleanup = crate::runtime::DeviceCleanupGuard(&session);
    let mut result = MonteCarloResult {
        rows: rows
            .iter()
            .map(|row| MonteCarloRow {
                row_id: row.row_id.clone(),
                moments: GreekMoments::default(),
            })
            .collect(),
        portfolio: GreekMoments::default(),
        pair_count: config.antithetic_pairs,
        max_sampled_allocator_live_bytes: None,
        max_sampled_allocator_reserved_bytes: None,
    };
    let mut first_pair = 0;
    let mut completed_batches = 0;
    while first_pair < config.antithetic_pairs {
        let pairs = plan
            .pairs_per_batch
            .min(config.antithetic_pairs - first_pair);
        let normals: Vec<_> = (0..pairs)
            .map(|i| normal_at(config.seed, (first_pair + i) as u64))
            .collect();
        let batch = monte_carlo_rows(&session, rows, &normals)?;
        if batch.rows.len() != rows.len() || batch.pair_count != pairs {
            return Err(ComputeError::Execution(
                "invalid Monte Carlo batch result shape".into(),
            ));
        }
        for (output, input) in result.rows.iter_mut().zip(batch.rows) {
            if output.row_id != input.row_id {
                return Err(ComputeError::Execution(
                    "Monte Carlo result identity mismatch".into(),
                ));
            }
            output.moments.merge(&input.moments);
        }
        result.portfolio.merge(&batch.portfolio);
        let expected_count = (first_pair + pairs) as u64;
        for row in &result.rows {
            validate_moments(&row.moments, expected_count)?;
        }
        validate_moments(&result.portfolio, expected_count)?;
        result.max_sampled_allocator_live_bytes = max_usage(
            result.max_sampled_allocator_live_bytes,
            batch.max_sampled_allocator_live_bytes,
        );
        result.max_sampled_allocator_reserved_bytes = max_usage(
            result.max_sampled_allocator_reserved_bytes,
            batch.max_sampled_allocator_reserved_bytes,
        );
        if result
            .max_sampled_allocator_reserved_bytes
            .is_some_and(|n| n > config.budget.compute_bytes)
        {
            session.device().memory_cleanup();
            return Err(ComputeError::BudgetExceeded(
                "Monte Carlo allocator reservation exceeds compute budget".into(),
            ));
        }
        first_pair += pairs;
        completed_batches += 1;
    }
    session.synchronize()?;
    drop(cleanup);
    Ok(MonteCarloJobResult {
        result,
        plan,
        completed_batches,
        job_seconds: started.elapsed().as_secs_f64(),
    })
}
fn max_usage(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}
fn usable(bytes: u64, headroom: u8) -> u64 {
    let pct = 100 - u64::from(headroom);
    (bytes / 100) * pct + (bytes % 100) * pct / 100
}
fn overflow() -> ComputeError {
    ComputeError::InvalidInput("Monte Carlo memory/count estimate overflow".into())
}

fn validate_moments(moments: &GreekMoments, expected_count: u64) -> Result<(), ComputeError> {
    for metric in [
        &moments.price,
        &moments.delta,
        &moments.vega,
        &moments.rho,
        &moments.theta,
        &moments.dividend_sensitivity,
        &moments.gamma,
    ] {
        if metric.count != expected_count
            || !metric.mean.is_finite()
            || !metric.m2.is_finite()
            || metric.m2 < 0.0
        {
            return Err(ComputeError::Execution(
                "Monte Carlo merged moments overflow or count mismatch".into(),
            ));
        }
    }
    Ok(())
}
