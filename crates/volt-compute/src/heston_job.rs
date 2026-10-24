//! Budgeted European Heston risk. Numeric requests preserve position identities
//! and weights; they do not validate legal contracts or a canonical Volt ledger.
use crate::{
    ComputeError, ExecutionTarget, Precision, Session,
    budget::MemoryBudget,
    chunked::{CompensatedSum, execution_guard},
    heston::{self, HestonOption, HestonParams, HestonResult, QuadratureConfig},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HestonPosition {
    pub row_id: String,
    pub contract_id: String,
    pub quantity: i64,
    pub multiplier: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HestonJobRequest {
    pub schema: String,
    pub currency: String,
    pub options: Vec<HestonOption>,
    pub params: Vec<HestonParams>,
    pub positions: Vec<HestonPosition>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HestonJobConfig {
    pub target: ExecutionTarget,
    pub precision: Precision,
    pub quadrature: QuadratureConfig,
    pub budget: MemoryBudget,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HestonMemoryPlan {
    pub schema: String,
    pub option_rows: usize,
    pub chunk_rows: usize,
    pub chunks: usize,
    pub quadrature_nodes: usize,
    pub expanded_cells_per_chunk: usize,
    pub total_quadrature_cells: u64,
    pub estimated_compute_bytes: u64,
    pub estimated_host_working_bytes: u64,
    pub estimated_compute_bytes_per_cell: u64,
    pub compute_admission_limit_bytes: u64,
    pub host_admission_limit_bytes: u64,
}
#[derive(Debug, Serialize)]
pub struct HestonJobResult {
    pub values: HestonResult,
    /// Signed quantity/multiplier sums for parallel changes in row inputs.
    /// Variance risk is per unit VARIANCE, not annual volatility points.
    pub parallel_sensitivities: Vec<NamedSensitivity>,
    pub plan: HestonMemoryPlan,
    pub completed_chunks: usize,
    pub evaluated_unique_rows: usize,
    pub device_batches: usize,
    pub job_seconds: f64,
}
#[derive(Debug, Serialize)]
pub struct NamedSensitivity {
    pub metric: &'static str,
    pub value: f64,
}

pub fn plan_heston(
    request: &HestonJobRequest,
    config: &HestonJobConfig,
) -> Result<HestonMemoryPlan, ComputeError> {
    let n = request.options.len();
    let budget = &config.budget;
    if request.schema != "volt.compute.heston_request.v1"
        || n == 0
        || request.params.len() != n
        || request.positions.len() != n
        || request.currency.len() != 3
        || !request.currency.bytes().all(|c| c.is_ascii_uppercase())
    {
        return Err(ComputeError::InvalidInput("Heston request needs matching nonempty options/params/positions, v1 schema and uppercase three-letter currency".into()));
    }
    if budget.headroom_percent >= 100
        || budget.compute_bytes == 0
        || budget.host_working_bytes == 0
        || budget.max_batch_rows == 0
    {
        return Err(ComputeError::InvalidInput(
            "positive Heston budgets/cap and headroom below 100% required".into(),
        ));
    }
    if config.target == ExecutionTarget::Metal && config.precision == Precision::F64 {
        return Err(ComputeError::Unsupported(
            "Metal does not provide native f64 arithmetic".into(),
        ));
    }
    let nodes = config.quadrature.nodes;
    if !(16..=2048).contains(&nodes) {
        return Err(ComputeError::InvalidInput(
            "Heston requires 16..=2048 quadrature nodes".into(),
        ));
    }
    let total_cells = (n as u64).checked_mul(nodes as u64).ok_or_else(overflow)?;
    let scalar = if config.precision == Precision::F64 {
        8
    } else {
        4
    };
    // Complex characteristic-function forward/backward graph, nine leaves and
    // reductions. This is not a model for retained time-stepped Heston paths.
    let compute_per_cell = 1024 * scalar;
    let compute_limit = usable(budget.compute_bytes, budget.headroom_percent);
    let host_limit = usable(budget.host_working_bytes, budget.headroom_percent);
    let max_id = request
        .options
        .iter()
        .map(|o| o.row_id.len() as u64)
        .max()
        .unwrap_or(0);
    // Includes ordered risk output, identical-input result cache, identity index,
    // comparison records and cloned per-batch source metadata.
    let global_host = (n as u64)
        .checked_mul(
            2048_u64
                .checked_add(max_id.checked_mul(2).ok_or_else(overflow)?)
                .ok_or_else(overflow)?,
        )
        .and_then(|v| v.checked_add(8192))
        .ok_or_else(overflow)?;
    let shared = matches!(config.target, ExecutionTarget::Cpu | ExecutionTarget::Metal);
    let host_fixed = global_host
        .checked_add(if shared {
            budget.compute_reserve_bytes
        } else {
            0
        })
        .ok_or_else(overflow)?;
    let host_per_cell = 768 + if shared { compute_per_cell } else { 0 };
    let compute_cells = compute_limit
        .checked_sub(budget.compute_reserve_bytes)
        .ok_or_else(|| {
            ComputeError::BudgetExceeded("Heston compute reserve exceeds admission limit".into())
        })?
        / compute_per_cell;
    let host_cells = host_limit.checked_sub(host_fixed).ok_or_else(|| {
        ComputeError::BudgetExceeded(
            "Heston global output/shared reserve exceeds host admission limit".into(),
        )
    })? / host_per_cell;
    let chunk_rows = (n as u64)
        .min(compute_cells / nodes as u64)
        .min(host_cells / nodes as u64)
        .min(budget.max_batch_rows as u64 / nodes as u64)
        .min(heston::MAX_QUADRATURE_CELLS as u64 / nodes as u64) as usize;
    if chunk_rows == 0 {
        return Err(ComputeError::BudgetExceeded(
            "cannot admit one Heston row with its complete quadrature and backward graph".into(),
        ));
    }
    // Check the entire request before device initialization, including identities
    // crossing chunks; validate a single row at a time to avoid the batch cell cap.
    let mut ids = Vec::with_capacity(n);
    for ((o, p), position) in request
        .options
        .iter()
        .zip(&request.params)
        .zip(&request.positions)
    {
        heston::validate_inputs(
            std::slice::from_ref(o),
            std::slice::from_ref(p),
            config.quadrature,
            config.precision,
        )?;
        if position.row_id != o.row_id
            || position.contract_id.trim().is_empty()
            || position.multiplier == 0
        {
            return Err(ComputeError::InvalidInput(
                "Heston position identity mismatch, empty contract ID or zero multiplier".into(),
            ));
        }
        ids.push(o.row_id.as_str());
    }
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ComputeError::InvalidInput(
            "duplicate Heston row across job chunks".into(),
        ));
    }
    let cells = chunk_rows * nodes;
    Ok(HestonMemoryPlan {
        schema: "volt.compute.heston_memory_plan.v1".into(),
        option_rows: n,
        chunk_rows,
        chunks: n.div_ceil(chunk_rows),
        quadrature_nodes: nodes,
        expanded_cells_per_chunk: cells,
        total_quadrature_cells: total_cells,
        estimated_compute_bytes: budget.compute_reserve_bytes + cells as u64 * compute_per_cell,
        estimated_host_working_bytes: host_fixed + cells as u64 * host_per_cell,
        estimated_compute_bytes_per_cell: compute_per_cell,
        compute_admission_limit_bytes: compute_limit,
        host_admission_limit_bytes: host_limit,
    })
}

pub fn run_heston(
    request: &HestonJobRequest,
    config: &HestonJobConfig,
) -> Result<HestonJobResult, ComputeError> {
    let started = Instant::now();
    let plan = plan_heston(request, config)?;
    let _guard = execution_guard()?;
    let session = Session::new(
        config.target,
        config.precision,
        plan.expanded_cells_per_chunk,
    )?;
    let cleanup = crate::runtime::DeviceCleanupGuard(&session);
    let mut output = empty_result(request.options.len());
    let mut sums: [CompensatedSum; 11] = std::array::from_fn(|_| CompensatedSum::default());
    let mut completed_chunks = 0;
    let mut device_batches = 0;
    // Financially identical inputs must share one numerical result, including
    // across tail batches. Shape-dependent f32 backend rounding must not turn
    // equal long/short exposures into a spurious residual. IDs and weights are
    // preserved separately. Cache storage is included in global host allowance.
    let mut cache: BTreeMap<[u64; 11], [f64; 11]> = BTreeMap::new();
    for start in (0..request.options.len()).step_by(plan.chunk_rows) {
        let end = (start + plan.chunk_rows).min(request.options.len());
        let mut seen = BTreeSet::new();
        let missing: Vec<_> = (start..end)
            .filter(|&i| {
                let key = economic_key(&request.options[i], &request.params[i]);
                !cache.contains_key(&key) && seen.insert(key)
            })
            .collect();
        if !missing.is_empty() {
            let options: Vec<_> = missing
                .iter()
                .map(|&i| request.options[i].clone())
                .collect();
            let params: Vec<_> = missing.iter().map(|&i| request.params[i]).collect();
            let batch = heston::differentiate_rows(&session, &options, &params, config.quadrature)?;
            let arrays = metrics(&batch);
            if arrays.iter().any(|(_, v)| v.len() != missing.len()) {
                return Err(ComputeError::Execution(
                    "Heston output length mismatch".into(),
                ));
            }
            for (j, &source) in missing.iter().enumerate() {
                let key = economic_key(&request.options[source], &request.params[source]);
                cache.insert(key, std::array::from_fn(|k| arrays[k].1[j]));
            }
            output.max_sampled_allocator_live_bytes = max_usage(
                output.max_sampled_allocator_live_bytes,
                batch.max_sampled_allocator_live_bytes,
            );
            output.max_sampled_allocator_reserved_bytes = max_usage(
                output.max_sampled_allocator_reserved_bytes,
                batch.max_sampled_allocator_reserved_bytes,
            );
            if output
                .max_sampled_allocator_reserved_bytes
                .is_some_and(|n| n > config.budget.compute_bytes)
            {
                return Err(ComputeError::BudgetExceeded(
                    "Heston allocator reservation exceeds compute budget".into(),
                ));
            }
            device_batches += 1;
        }
        for source in start..end {
            let values = *cache
                .get(&economic_key(
                    &request.options[source],
                    &request.params[source],
                ))
                .ok_or_else(|| ComputeError::Execution("missing Heston numerical result".into()))?;
            let position = &request.positions[source];
            for (sum, value) in sums.iter_mut().zip(values) {
                let weighted = value * position.quantity as f64 * position.multiplier as f64;
                if !weighted.is_finite() {
                    return Err(ComputeError::Execution(
                        "Heston weighted sensitivity overflow".into(),
                    ));
                }
                sum.add(weighted);
            }
            push_values(&mut output, values);
        }
        completed_chunks += 1;
    }
    let mut parallel = Vec::with_capacity(11);
    for ((name, _), sum) in metrics(&output).into_iter().zip(sums) {
        let value = sum.value();
        if !value.is_finite() {
            return Err(ComputeError::Execution(
                "Heston portfolio aggregate overflow".into(),
            ));
        }
        parallel.push(NamedSensitivity {
            metric: name,
            value,
        });
    }
    session.synchronize()?;
    drop(cleanup);
    Ok(HestonJobResult {
        values: output,
        parallel_sensitivities: parallel,
        plan,
        completed_chunks,
        evaluated_unique_rows: cache.len(),
        device_batches,
        job_seconds: started.elapsed().as_secs_f64(),
    })
}

pub fn metrics(values: &HestonResult) -> [(&'static str, &Vec<f64>); 11] {
    [
        ("price", &values.prices),
        ("delta", &values.delta),
        ("gamma", &values.gamma),
        ("initial_variance", &values.initial_variance_sensitivity),
        ("kappa", &values.kappa_sensitivity),
        ("long_run_variance", &values.long_run_variance_sensitivity),
        ("vol_of_variance", &values.vol_of_variance_sensitivity),
        ("correlation", &values.correlation_sensitivity),
        ("rho", &values.rho),
        ("theta", &values.theta),
        ("dividend", &values.dividend_sensitivity),
    ]
}
fn empty_result(n: usize) -> HestonResult {
    HestonResult {
        prices: Vec::with_capacity(n),
        delta: Vec::with_capacity(n),
        gamma: Vec::with_capacity(n),
        initial_variance_sensitivity: Vec::with_capacity(n),
        kappa_sensitivity: Vec::with_capacity(n),
        long_run_variance_sensitivity: Vec::with_capacity(n),
        vol_of_variance_sensitivity: Vec::with_capacity(n),
        correlation_sensitivity: Vec::with_capacity(n),
        rho: Vec::with_capacity(n),
        theta: Vec::with_capacity(n),
        dividend_sensitivity: Vec::with_capacity(n),
        max_sampled_allocator_live_bytes: None,
        max_sampled_allocator_reserved_bytes: None,
    }
}
fn economic_key(o: &HestonOption, p: &HestonParams) -> [u64; 11] {
    [
        o.spot.to_bits(),
        o.strike.to_bits(),
        o.maturity_years.to_bits(),
        o.rate.to_bits(),
        o.dividend_yield.to_bits(),
        p.initial_variance.to_bits(),
        p.kappa.to_bits(),
        p.long_run_variance.to_bits(),
        p.vol_of_variance.to_bits(),
        p.correlation.to_bits(),
        u64::from(matches!(o.side, crate::OptionSide::Put)),
    ]
}
fn push_values(out: &mut HestonResult, values: [f64; 11]) {
    let [
        price,
        delta,
        gamma,
        v0,
        kappa,
        long_run,
        xi,
        corr,
        rho,
        theta,
        dividend,
    ] = values;
    out.prices.push(price);
    out.delta.push(delta);
    out.gamma.push(gamma);
    out.initial_variance_sensitivity.push(v0);
    out.kappa_sensitivity.push(kappa);
    out.long_run_variance_sensitivity.push(long_run);
    out.vol_of_variance_sensitivity.push(xi);
    out.correlation_sensitivity.push(corr);
    out.rho.push(rho);
    out.theta.push(theta);
    out.dividend_sensitivity.push(dividend);
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
    ComputeError::InvalidInput("Heston memory/count estimate overflow".into())
}
