//! Genuine analytic BS / differentiated-Fourier Heston model hedges on supplied
//! market paths. This is a model baseline, not a learned policy or BS proxy for
//! Heston. Supported application accounting is currently r=q=0, no market impact.
//! Heston Greeks are checked by separate node and cutoff refinements. Failure is
//! an error; no clipping, nearest-grid lookup or alternate model is substituted.
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use serde::{Deserialize, Serialize};

use crate::{
    ComputeError, ExecutionTarget, OptionRow, OptionSide, Precision,
    aad::analytic_greeks,
    budget::MemoryBudget,
    chunked::CompensatedSum,
    hedge_application::{HedgeCostConfig, HedgeLiability, HedgePathBatch},
    heston::{HestonOption, HestonParams, QuadratureConfig},
    heston_adaptive::{
        AdaptiveHestonConfig, AdaptiveHestonResult, AdaptiveTailMethod, adaptive_greeks,
    },
    heston_job::{HestonJobConfig, HestonJobRequest, HestonPosition, run_heston},
    market_paths::{MarketModel, MarketPaths, STOCK, VARIANCE},
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HestonBaselineMethod {
    #[default]
    DeviceFourier,
    /// Explicit native Rust CPU f64 reference, regardless of tensor target/dtype.
    AdaptiveScalar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BaselineExecutionPolicy {
    AnalyticBlackScholesCpuF64,
    DeviceFourier {
        target: ExecutionTarget,
        precision: Precision,
    },
    AdaptiveScalarCpuF64,
}
const ADAPTIVE_THREAD_STACK_BYTES: usize = 1024 * 1024;
fn one_thread() -> usize {
    1
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineTolerances {
    pub price_absolute: f64,
    pub price_relative: f64,
    pub delta_absolute: f64,
    pub delta_relative: f64,
    pub variance_absolute: f64,
    pub variance_relative: f64,
}
impl Default for BaselineTolerances {
    fn default() -> Self {
        Self {
            price_absolute: 5e-5,
            price_relative: 5e-5,
            delta_absolute: 5e-5,
            delta_relative: 5e-4,
            variance_absolute: 5e-3,
            variance_relative: 5e-4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelHedgeConfig {
    /// DeviceFourier target. AdaptiveScalar and BS explicitly use CPU f64.
    pub target: ExecutionTarget,
    /// DeviceFourier arithmetic; scalar references always retain f64.
    pub precision: Precision,
    #[serde(default)]
    pub heston_method: HestonBaselineMethod,
    #[serde(default)]
    pub adaptive: AdaptiveHestonConfig,
    #[serde(default = "one_thread")]
    pub adaptive_threads: usize,
    pub budget: MemoryBudget,
    /// DeviceFourier base nodes <=1024, cutoff <=5million. Check twice the nodes at this
    /// cutoff, then twice the cutoff at the refined node count. Use last result.
    pub quadrature: QuadratureConfig,
    pub tolerances: BaselineTolerances,
    /// 1..=256 distinct pricing queries per outer job; inner job admission may
    /// further chunk by expanded quadrature cells and backward workspace.
    pub max_queries_per_job: usize,
    /// Exact economic-state memoization only. Once full, new states are not cached.
    pub max_cached_states: usize,
}
impl Default for ModelHedgeConfig {
    fn default() -> Self {
        Self {
            target: ExecutionTarget::Cpu,
            precision: Precision::F64,
            heston_method: HestonBaselineMethod::DeviceFourier,
            adaptive: AdaptiveHestonConfig::default(),
            adaptive_threads: 1,
            budget: MemoryBudget::default(),
            quadrature: QuadratureConfig {
                nodes: 1024,
                upper_bound: 40_000.0,
            },
            tolerances: BaselineTolerances::default(),
            max_queries_per_job: 256,
            max_cached_states: 10_000,
        }
    }
}
impl ModelHedgeConfig {
    /// Pure configuration validation; creates no device/session or path buffers.
    pub fn validate(&self) -> Result<(), ComputeError> {
        validate_config(self)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaselineValidation {
    pub requested_call_queries: usize,
    pub priced_call_queries: usize,
    pub exact_cache_hits: usize,
    pub coalesced_queries: usize,
    pub pricing_jobs: usize,
    pub device_batches: usize,
    pub adaptive_converged_queries: usize,
    /// Successful two-successive-shell and envelope checks. These are numerical
    /// tail estimates, not universal rigorous analytic certificates.
    pub adaptive_tail_checked_queries: usize,
    /// Subset using the restricted, real-arithmetic characteristic-modulus bound.
    #[serde(default)]
    pub adaptive_analytic_tail_queries: usize,
    pub adaptive_function_evaluations: u64,
    pub adaptive_panels: u64,
    pub adaptive_threads_used: usize,
    pub adaptive_max_cutoff: f64,
    /// Financial units, ordered price, delta, current-variance sensitivity.
    pub adaptive_max_estimated_errors: [f64; 3],
    pub adaptive_max_tail_estimates: [f64; 3],
    /// Propagated absolute bounds for a cached linear component composition.
    #[serde(default)]
    pub composed_error_bounds: [f64; 3],
    pub max_price_node_difference: f64,
    pub max_price_cutoff_difference: f64,
    pub max_delta_node_difference: f64,
    pub max_delta_cutoff_difference: f64,
    pub max_variance_node_difference: f64,
    pub max_variance_cutoff_difference: f64,
    pub estimated_additional_host_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHedgeResult {
    pub schema: String,
    pub paths: usize,
    pub steps: usize,
    pub hedge_count: usize,
    /// Row-major `[path, decision, instrument]`; no terminal decision or division.
    pub holdings: Vec<f64>,
    /// Sum of model liability values at inception for each path; premium is not
    /// added to any P&L implicitly. A caller can compare common-premium hedges.
    pub initial_model_prices: Vec<f64>,
    pub market_source_hashes: Vec<String>,
    pub liability: HedgeLiability,
    pub config: ModelHedgeConfig,
    pub execution_policy: BaselineExecutionPolicy,
    pub validation: BaselineValidation,
}

#[derive(Debug, Clone, Copy)]
struct CallRisk {
    price: f64,
    delta: f64,
    variance: f64,
}
type StateKey = [u64; 9];
#[derive(Clone)]
struct Destination {
    stock: usize,
    variance: usize,
    weight: f64,
    denominator: f64,
    initial_path: Option<usize>,
}
struct Query {
    option: HestonOption,
    params: HestonParams,
    destinations: Vec<Destination>,
}

/// dH/d(current variance) for a funded claim paying integrated variance at T.
/// This is not the derivative of an annualized or zero-value variance swap.
pub fn funded_variance_claim_variance_derivative(
    kappa: f64,
    remaining_years: f64,
    rate: f64,
) -> Result<f64, ComputeError> {
    if !kappa.is_finite()
        || kappa <= 0.0
        || !remaining_years.is_finite()
        || remaining_years < 0.0
        || !rate.is_finite()
    {
        return Err(invalid(
            "funded variance derivative requires positive kappa, nonnegative remaining time and finite rate",
        ));
    }
    finite(
        (-rate * remaining_years).exp() * (-(-kappa * remaining_years).exp_m1()) / kappa,
        "funded claim variance derivative",
    )
}

type LiabilityLegs = Vec<(f64, f64)>;
fn validate_model_inputs(
    markets: &[MarketPaths],
    liability: HedgeLiability,
    config: &ModelHedgeConfig,
) -> Result<(bool, LiabilityLegs), ComputeError> {
    let first = markets
        .first()
        .ok_or_else(|| invalid("model hedge needs at least one market"))?;
    validate_config(config)?;
    let heston = matches!(first.config.model, MarketModel::Heston { .. });
    for market in markets {
        market.validate()?;
        if market.config.paths != first.config.paths
            || market.config.steps != first.config.steps
            || market.config.first_path_id != first.config.first_path_id
            || market.times != first.times
            || matches!(market.config.model, MarketModel::Heston { .. }) != heston
        {
            return Err(invalid(
                "model hedge markets must have aligned path identities, grids and model families",
            ));
        }
        if market.config.rate != 0.0 || market.config.dividend_yield != 0.0 {
            return Err(ComputeError::Unsupported(
                "model hedge accounting currently supports r=q=0 only".into(),
            ));
        }
    }
    Ok((heston, liability_legs(liability)?))
}

pub fn model_hedge_holdings(
    markets: &[MarketPaths],
    liability: HedgeLiability,
    config: &ModelHedgeConfig,
) -> Result<ModelHedgeResult, ComputeError> {
    let (heston, legs) = validate_model_inputs(markets, liability, config)?;
    let first = &markets[0];
    let hedge_count = markets
        .len()
        .checked_mul(if heston { 2 } else { 1 })
        .ok_or_else(|| invalid("model hedge shape overflow"))?;
    let length = first
        .config
        .paths
        .checked_mul(first.config.steps)
        .and_then(|x| x.checked_mul(hedge_count))
        .ok_or_else(|| invalid("model hedge shape overflow"))?;
    let cache_bytes = (config.max_cached_states as u64)
        .checked_mul(512)
        .ok_or_else(|| invalid("model hedge cache size overflow"))?;
    let adaptive = heston && config.heston_method == HestonBaselineMethod::AdaptiveScalar;
    let workers = config.adaptive_threads.min(config.max_queries_per_job);
    let adaptive_workspace = if adaptive {
        (workers as u64)
            .checked_mul(config.adaptive.modeled_workspace_bytes())
            .ok_or_else(|| invalid("adaptive workspace overflow"))?
    } else {
        0
    };
    let adaptive_host = if adaptive {
        adaptive_workspace
            .checked_add(
                (workers as u64)
                    .checked_mul(ADAPTIVE_THREAD_STACK_BYTES as u64)
                    .ok_or_else(|| invalid("adaptive stack size overflow"))?,
            )
            .and_then(|x| x.checked_add(config.budget.compute_reserve_bytes))
            .ok_or_else(|| invalid("adaptive host size overflow"))?
    } else {
        0
    };
    if adaptive
        && adaptive_workspace
            .checked_add(config.budget.compute_reserve_bytes)
            .is_none_or(|x| {
                x > usable_host(config.budget.compute_bytes, config.budget.headroom_percent)
            })
    {
        return Err(ComputeError::BudgetExceeded(
            "adaptive CPU f64 work/reserve exceed compute allowance".into(),
        ));
    }
    let fixed = (length as u64)
        .checked_mul(8)
        .and_then(|x| x.checked_add((first.config.paths as u64).checked_mul(8)?))
        .and_then(|x| x.checked_add(if heston { cache_bytes } else { 0 }))
        .and_then(|x| x.checked_add((config.max_queries_per_job as u64).checked_mul(8192)?))
        .and_then(|x| x.checked_add(32768))
        .and_then(|x| x.checked_add(adaptive_host))
        .ok_or_else(|| invalid("model hedge memory size overflow"))?;
    let available = usable_host(
        config.budget.host_working_bytes,
        config.budget.headroom_percent,
    );
    if fixed >= available {
        return Err(ComputeError::BudgetExceeded("model hedge holdings/cache exceed additional host allowance; stream fewer paths or reduce cache".into()));
    }
    let mut job_budget = config.budget.clone();
    // Leave the caller's declared headroom in place for the nested Heston job.
    job_budget.host_working_bytes = (available - fixed)
        .checked_mul(100)
        .ok_or_else(|| invalid("nested host budget overflow"))?
        / u64::from(100 - config.budget.headroom_percent);
    let mut result = ModelHedgeResult {
        schema: "volt.compute.model_hedge.v1".into(),
        paths: first.config.paths,
        steps: first.config.steps,
        hedge_count,
        holdings: vec![0.0; length],
        initial_model_prices: vec![0.0; first.config.paths],
        market_source_hashes: markets.iter().map(|x| x.content_hash.clone()).collect(),
        liability,
        config: config.clone(),
        execution_policy: if !heston {
            BaselineExecutionPolicy::AnalyticBlackScholesCpuF64
        } else if adaptive {
            BaselineExecutionPolicy::AdaptiveScalarCpuF64
        } else {
            BaselineExecutionPolicy::DeviceFourier {
                target: config.target,
                precision: config.precision,
            }
        },
        validation: BaselineValidation {
            estimated_additional_host_bytes: fixed,
            ..BaselineValidation::default()
        },
    };
    let mut cache: BTreeMap<StateKey, CallRisk> = BTreeMap::new();
    let mut pending = Vec::<Query>::new();
    let mut pending_index = BTreeMap::<StateKey, usize>::new();
    for path in 0..result.paths {
        for decision in 0..result.steps {
            let tau = first.config.horizon() - first.times[decision];
            if tau <= 0.0 {
                return Err(invalid(
                    "model baseline decision must precede terminal time",
                ));
            }
            for (system, market) in markets.iter().enumerate() {
                let spot = market
                    .value(path, decision, STOCK)
                    .ok_or_else(|| invalid("missing model hedge spot"))?;
                let offset = (path * result.steps + decision) * hedge_count
                    + system * if heston { 2 } else { 1 };
                for &(strike, weight) in &legs {
                    result.validation.requested_call_queries += 1;
                    match market.config.model {
                        MarketModel::BlackScholes { volatility } => {
                            let (price, delta) = bs_call(spot, strike, tau, volatility)?;
                            result.holdings[offset] =
                                finite(result.holdings[offset] + weight * delta, "BS hedge sum")?;
                            if decision == 0 {
                                result.initial_model_prices[path] = finite(
                                    result.initial_model_prices[path] + weight * price,
                                    "BS initial price sum",
                                )?;
                            }
                        }
                        MarketModel::Heston { mut params } => {
                            params.initial_variance = market
                                .value(path, decision, VARIANCE)
                                .ok_or_else(|| invalid("missing model current variance"))?;
                            let option = HestonOption {
                                row_id: format!("model-{path}-{decision}-{system}-{strike}"),
                                side: OptionSide::Call,
                                spot,
                                strike,
                                maturity_years: tau,
                                rate: 0.0,
                                dividend_yield: 0.0,
                            };
                            let denominator =
                                funded_variance_claim_variance_derivative(params.kappa, tau, 0.0)?;
                            if denominator <= 0.0 {
                                return Err(invalid(
                                    "funded claim sensitivity is zero at a decision",
                                ));
                            }
                            let destination = Destination {
                                stock: offset,
                                variance: offset + 1,
                                weight,
                                denominator,
                                initial_path: (decision == 0).then_some(path),
                            };
                            let key = key(&option, &params);
                            if let Some(risk) = cache.get(&key) {
                                apply_risk(&mut result, *risk, &destination)?;
                                result.validation.exact_cache_hits += 1;
                            } else if let Some(index) = pending_index.get(&key) {
                                pending[*index].destinations.push(destination);
                                result.validation.coalesced_queries += 1;
                            } else {
                                pending_index.insert(key, pending.len());
                                pending.push(Query {
                                    option,
                                    params,
                                    destinations: vec![destination],
                                });
                            }
                            if pending.iter().map(|q| q.destinations.len()).sum::<usize>()
                                >= config.max_queries_per_job
                            {
                                flush(
                                    &mut pending,
                                    &mut pending_index,
                                    &mut cache,
                                    &mut result,
                                    &job_budget,
                                )?;
                            }
                        }
                    }
                }
            }
        }
    }
    flush(
        &mut pending,
        &mut pending_index,
        &mut cache,
        &mut result,
        &job_budget,
    )?;
    validate_parity_composition(markets.len(), &mut result)?;
    apply_parity_adjustment(markets, &mut result)?;
    Ok(result)
}

fn flush(
    pending: &mut Vec<Query>,
    index: &mut BTreeMap<StateKey, usize>,
    cache: &mut BTreeMap<StateKey, CallRisk>,
    result: &mut ModelHedgeResult,
    budget: &MemoryBudget,
) -> Result<(), ComputeError> {
    if pending.is_empty() {
        return Ok(());
    }
    if result.config.heston_method == HestonBaselineMethod::AdaptiveScalar {
        return flush_adaptive(pending, index, cache, result);
    }
    let request = HestonJobRequest {
        schema: "volt.compute.heston_request.v1".into(),
        currency: "USD".into(),
        options: pending.iter().map(|q| q.option.clone()).collect(),
        params: pending.iter().map(|q| q.params).collect(),
        positions: pending
            .iter()
            .map(|q| HestonPosition {
                row_id: q.option.row_id.clone(),
                contract_id: "synthetic-baseline-liability".into(),
                quantity: 1,
                multiplier: 1,
            })
            .collect(),
    };
    let q = result.config.quadrature;
    let run = |quad| {
        run_heston(
            &request,
            &HestonJobConfig {
                target: result.config.target,
                precision: result.config.precision,
                quadrature: quad,
                budget: budget.clone(),
            },
        )
    };
    let base = run(q)?;
    let node = run(QuadratureConfig {
        nodes: q.nodes * 2,
        upper_bound: q.upper_bound,
    })?;
    let tail = run(QuadratureConfig {
        nodes: q.nodes * 2,
        upper_bound: q.upper_bound * 2.0,
    })?;
    result.validation.pricing_jobs += 3;
    result.validation.device_batches +=
        base.device_batches + node.device_batches + tail.device_batches;
    result.validation.priced_call_queries += pending.len();
    for (i, query) in pending.iter().enumerate() {
        let triples = [
            (
                "price",
                base.values.prices[i],
                node.values.prices[i],
                tail.values.prices[i],
                result.config.tolerances.price_absolute,
                result.config.tolerances.price_relative,
            ),
            (
                "delta",
                base.values.delta[i],
                node.values.delta[i],
                tail.values.delta[i],
                result.config.tolerances.delta_absolute,
                result.config.tolerances.delta_relative,
            ),
            (
                "variance",
                base.values.initial_variance_sensitivity[i],
                node.values.initial_variance_sensitivity[i],
                tail.values.initial_variance_sensitivity[i],
                result.config.tolerances.variance_absolute,
                result.config.tolerances.variance_relative,
            ),
        ];
        for (name, a, b, c, absolute, relative) in triples {
            let nodes = (a - b).abs();
            let cutoff = (b - c).abs();
            match name {
                "price" => {
                    result.validation.max_price_node_difference =
                        result.validation.max_price_node_difference.max(nodes);
                    result.validation.max_price_cutoff_difference =
                        result.validation.max_price_cutoff_difference.max(cutoff);
                }
                "delta" => {
                    result.validation.max_delta_node_difference =
                        result.validation.max_delta_node_difference.max(nodes);
                    result.validation.max_delta_cutoff_difference =
                        result.validation.max_delta_cutoff_difference.max(cutoff);
                }
                _ => {
                    result.validation.max_variance_node_difference =
                        result.validation.max_variance_node_difference.max(nodes);
                    result.validation.max_variance_cutoff_difference =
                        result.validation.max_variance_cutoff_difference.max(cutoff);
                }
            }
            if ![a, b, c].iter().all(|x| x.is_finite())
                || nodes > absolute + relative * b.abs()
                || cutoff > absolute + relative * c.abs()
            {
                return Err(ComputeError::Execution(format!(
                    "model hedge quadrature failed {name} refinement for {}: node_difference={nodes:e}, cutoff_difference={cutoff:e}; S={},K={},tau={},V={}; increase declared resolution or restrict evaluated cohort",
                    query.option.row_id,
                    query.option.spot,
                    query.option.strike,
                    query.option.maturity_years,
                    query.params.initial_variance
                )));
            }
        }
        let risk = CallRisk {
            price: tail.values.prices[i],
            delta: tail.values.delta[i],
            variance: tail.values.initial_variance_sensitivity[i],
        };
        let slack = result.config.tolerances.price_absolute
            + result.config.tolerances.price_relative * query.option.spot;
        if risk.price < (query.option.spot - query.option.strike).max(0.0) - slack
            || risk.price > query.option.spot + slack
            || risk.delta < -result.config.tolerances.delta_absolute
            || risk.delta > 1.0 + result.config.tolerances.delta_absolute
        {
            return Err(ComputeError::Execution(
                "model hedge price/delta violates call bounds".into(),
            ));
        }
        for destination in &query.destinations {
            apply_risk(result, risk, destination)?;
        }
        if cache.len() < result.config.max_cached_states {
            cache.insert(key(&query.option, &query.params), risk);
        }
    }
    pending.clear();
    index.clear();
    Ok(())
}
fn apply_risk(
    result: &mut ModelHedgeResult,
    risk: CallRisk,
    d: &Destination,
) -> Result<(), ComputeError> {
    result.holdings[d.stock] = finite(
        result.holdings[d.stock] + d.weight * risk.delta,
        "stock hedge sum",
    )?;
    result.holdings[d.variance] = finite(
        result.holdings[d.variance] + d.weight * risk.variance / d.denominator,
        "variance claim hedge sum",
    )?;
    if let Some(path) = d.initial_path {
        result.initial_model_prices[path] = finite(
            result.initial_model_prices[path] + d.weight * risk.price,
            "initial model price sum",
        )?;
    }
    Ok(())
}

fn flush_adaptive(
    pending: &mut Vec<Query>,
    index: &mut BTreeMap<StateKey, usize>,
    cache: &mut BTreeMap<StateKey, CallRisk>,
    result: &mut ModelHedgeResult,
) -> Result<(), ComputeError> {
    let workers = result.config.adaptive_threads.min(pending.len());
    let config = result.config.adaptive;
    let compute = |queries: &[Query]| -> Result<Vec<AdaptiveHestonResult>, ComputeError> {
        queries
            .iter()
            .map(|q| adaptive_greeks(&q.option, &q.params, config))
            .collect()
    };
    let evaluated = if workers == 1 {
        compute(pending)?
    } else {
        // Native scalar calculations share no tensors, mutation or training
        // execution guard. Merge deterministic contiguous slices in query order.
        std::thread::scope(|scope| -> Result<Vec<AdaptiveHestonResult>, ComputeError> {
            let mut handles = Vec::new();
            for worker in 0..workers {
                let start = worker * pending.len() / workers;
                let end = (worker + 1) * pending.len() / workers;
                let queries = &pending[start..end];
                handles.push(
                    std::thread::Builder::new()
                        .name(format!("heston-reference-{worker}"))
                        .stack_size(ADAPTIVE_THREAD_STACK_BYTES)
                        .spawn_scoped(scope, move || compute(queries))
                        .map_err(|e| {
                            ComputeError::Execution(format!(
                                "cannot start adaptive CPU worker: {e}"
                            ))
                        })?,
                );
            }
            let mut output = Vec::with_capacity(pending.len());
            let mut failure = None;
            for handle in handles {
                match handle.join() {
                    Ok(Ok(mut values)) => output.append(&mut values),
                    Ok(Err(error)) => {
                        if failure.is_none() {
                            failure = Some(error);
                        }
                    }
                    Err(_) => {
                        if failure.is_none() {
                            failure = Some(ComputeError::Execution(
                                "adaptive CPU worker panicked".into(),
                            ));
                        }
                    }
                }
            }
            if let Some(error) = failure {
                Err(error)
            } else {
                Ok(output)
            }
        })?
    };
    if evaluated.len() != pending.len() {
        return Err(ComputeError::Execution(
            "adaptive baseline result shape mismatch".into(),
        ));
    }
    result.validation.pricing_jobs += 1;
    result.validation.priced_call_queries += pending.len();
    result.validation.adaptive_threads_used = result.validation.adaptive_threads_used.max(workers);
    for (query, value) in pending.iter().zip(evaluated) {
        let errors = [
            value.error_estimates.price,
            value.error_estimates.delta,
            value.error_estimates.initial_variance_sensitivity,
        ];
        let tolerances = [
            config.price_absolute_tolerance,
            config.delta_absolute_tolerance,
            config.variance_absolute_tolerance,
        ];
        if !value.converged
            || errors
                .iter()
                .zip(tolerances)
                .any(|(e, t)| !e.is_finite() || *e < 0.0 || *e > t)
            || value
                .error_estimates
                .tail
                .iter()
                .any(|e| !e.is_finite() || *e < 0.0)
        {
            return Err(ComputeError::Execution(format!(
                "adaptive baseline did not certify its declared numerical estimates for {}",
                query.option.row_id
            )));
        }
        let risk = CallRisk {
            price: finite(value.price, "adaptive price")?,
            delta: finite(value.delta, "adaptive delta")?,
            variance: finite(
                value.initial_variance_sensitivity,
                "adaptive variance sensitivity",
            )?,
        };
        let slack = config.price_absolute_tolerance;
        if risk.price < (query.option.spot - query.option.strike).max(0.0) - slack
            || risk.price > query.option.spot + slack
            || risk.delta < -config.delta_absolute_tolerance
            || risk.delta > 1.0 + config.delta_absolute_tolerance
        {
            return Err(ComputeError::Execution(format!(
                "adaptive model hedge violates call bounds for {}",
                query.option.row_id
            )));
        }
        result.validation.adaptive_converged_queries += 1;
        result.validation.adaptive_tail_checked_queries += 1;
        if value.tail_method == AdaptiveTailMethod::AnalyticCharacteristicBound {
            result.validation.adaptive_analytic_tail_queries += 1;
        }
        result.validation.adaptive_function_evaluations = result
            .validation
            .adaptive_function_evaluations
            .checked_add(value.evaluations as u64)
            .ok_or_else(|| invalid("adaptive evaluation counter overflow"))?;
        result.validation.adaptive_panels = result
            .validation
            .adaptive_panels
            .checked_add(value.panels as u64)
            .ok_or_else(|| invalid("adaptive panel counter overflow"))?;
        result.validation.adaptive_max_cutoff =
            result.validation.adaptive_max_cutoff.max(value.cutoff);
        for (i, error) in errors.iter().enumerate() {
            result.validation.adaptive_max_estimated_errors[i] =
                result.validation.adaptive_max_estimated_errors[i].max(*error);
            result.validation.adaptive_max_tail_estimates[i] =
                result.validation.adaptive_max_tail_estimates[i].max(value.error_estimates.tail[i]);
        }
        for destination in &query.destinations {
            apply_risk(result, risk, destination)?;
        }
        if cache.len() < result.config.max_cached_states {
            cache.insert(key(&query.option, &query.params), risk);
        }
    }
    pending.clear();
    index.clear();
    Ok(())
}
fn key(o: &HestonOption, p: &HestonParams) -> StateKey {
    [
        o.spot.to_bits(),
        o.strike.to_bits(),
        o.maturity_years.to_bits(),
        p.initial_variance.to_bits(),
        p.kappa.to_bits(),
        p.long_run_variance.to_bits(),
        p.vol_of_variance.to_bits(),
        p.correlation.to_bits(),
        0,
    ]
}
fn liability_legs(liability: HedgeLiability) -> Result<Vec<(f64, f64)>, ComputeError> {
    liability.payoff(100.0)?;
    Ok(match liability {
        HedgeLiability::Call { strike } | HedgeLiability::Put { strike } => vec![(strike, 1.0)],
        HedgeLiability::Straddle { strike } => vec![(strike, 2.0)],
        HedgeLiability::NormalizedCallSpread { lower, upper } => vec![
            (lower, 1.0 / (upper - lower)),
            (upper, -1.0 / (upper - lower)),
        ],
        HedgeLiability::None => vec![],
    })
}
/// Exact European parity at the application boundary's required r=q=0.
/// Put=C-S+K; Straddle=2C-S+K. The K term is an inception mark, not
/// an extra terminal cashflow: the liability payoff already includes the put.
/// Call absolute error is inherited (doubled for a straddle); relative error
/// of a nearly worthless put can consequently be large. No clipping is used.
fn parity_strike(liability: HedgeLiability) -> Option<f64> {
    match liability {
        HedgeLiability::Put { strike } | HedgeLiability::Straddle { strike } => Some(strike),
        _ => None,
    }
}
/// Apply absolute component-error admission only for new parity liabilities.
/// Direct multi-system estimates use a conservative global maximum per system;
/// cached composition sums each component estimate. Variance error means C_v,
/// not the funded-claim holding error C_v/a(tau). Floating rounding is excluded.
/// Fixed Fourier refinement
/// differences are numerical diagnostics, not rigorous integration bounds.
fn validate_parity_composition(
    systems: usize,
    result: &mut ModelHedgeResult,
) -> Result<(), ComputeError> {
    let weight = match result.liability {
        HedgeLiability::Put { .. } => 1.0,
        HedgeLiability::Straddle { .. } => 2.0,
        _ => return Ok(()),
    };
    let v = &mut result.validation;
    let errors = match result.execution_policy {
        BaselineExecutionPolicy::AnalyticBlackScholesCpuF64 => [0.0; 3],
        BaselineExecutionPolicy::AdaptiveScalarCpuF64 => v.adaptive_max_estimated_errors,
        BaselineExecutionPolicy::DeviceFourier { .. } => [
            v.max_price_node_difference + v.max_price_cutoff_difference,
            v.max_delta_node_difference + v.max_delta_cutoff_difference,
            v.max_variance_node_difference + v.max_variance_cutoff_difference,
        ],
    };
    v.composed_error_bounds = [
        weight * systems as f64 * errors[0],
        weight * errors[1],
        weight * errors[2],
    ];
    let t = result.config.tolerances;
    if v.composed_error_bounds
        .iter()
        .zip([t.price_absolute, t.delta_absolute, t.variance_absolute])
        .any(|(e, t)| !e.is_finite() || *e > t)
    {
        return Err(ComputeError::Execution("parity liability composition exceeds absolute model error allowance; tighten component resolution/tolerances".into()));
    }
    Ok(())
}
fn apply_parity_adjustment(
    markets: &[MarketPaths],
    result: &mut ModelHedgeResult,
) -> Result<(), ComputeError> {
    let Some(strike) = parity_strike(result.liability) else {
        return Ok(());
    };
    let per = result.hedge_count / markets.len();
    for path in 0..result.paths {
        for (system, market) in markets.iter().enumerate() {
            for step in 0..result.steps {
                let offset = (path * result.steps + step) * result.hedge_count + system * per;
                result.holdings[offset] =
                    finite(result.holdings[offset] - 1.0, "parity stock holding")?;
            }
            let spot = market
                .value(path, 0, STOCK)
                .ok_or_else(|| invalid("missing parity spot"))?;
            result.initial_model_prices[path] = finite(
                result.initial_model_prices[path] + (strike - spot),
                "parity initial price",
            )?;
        }
    }
    Ok(())
}
fn bs_call(spot: f64, strike: f64, time: f64, volatility: f64) -> Result<(f64, f64), ComputeError> {
    if volatility == 0.0 {
        return if spot == strike {
            Err(ComputeError::Unsupported(
                "zero-volatility call at strike has no unique model delta".into(),
            ))
        } else {
            Ok((
                (spot - strike).max(0.0),
                if spot > strike { 1.0 } else { 0.0 },
            ))
        };
    }
    let g = analytic_greeks(&OptionRow {
        row_id: "model-bs".into(),
        contract_id: "baseline".into(),
        side: OptionSide::Call,
        spot,
        strike,
        maturity_years: time,
        volatility,
        rate: 0.0,
        dividend_yield: 0.0,
        quantity: 1,
        multiplier: 1,
    })?;
    Ok((g.price, g.delta))
}
pub fn validate_config(c: &ModelHedgeConfig) -> Result<(), ComputeError> {
    if c.max_queries_per_job == 0
        || c.max_queries_per_job > 256
        || !(16..=1024).contains(&c.quadrature.nodes)
        || !c.quadrature.upper_bound.is_finite()
        || !(1.0..=5_000_000.0).contains(&c.quadrature.upper_bound)
        || !(1..=64).contains(&c.adaptive_threads)
        || c.budget.headroom_percent >= 100
        || c.budget.host_working_bytes == 0
        || c.budget.compute_bytes == 0
    {
        return Err(invalid(
            "baseline requires1..=256 outer queries,16..=1024 base nodes,cutoff1..=5000000,1..=64 adaptive workers and positive budgets/headroom<100",
        ));
    }
    if c.heston_method == HestonBaselineMethod::AdaptiveScalar {
        c.adaptive.validate()?;
    }
    let t = c.tolerances;
    if [
        t.price_absolute,
        t.price_relative,
        t.delta_absolute,
        t.delta_relative,
        t.variance_absolute,
        t.variance_relative,
    ]
    .iter()
    .any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(invalid("baseline tolerances must be finite nonnegative"));
    }
    Ok(())
}
fn usable_host(bytes: u64, headroom: u8) -> u64 {
    ((bytes as u128) * u128::from(100 - headroom) / 100) as u64
}
fn invalid(message: &str) -> ComputeError {
    ComputeError::InvalidInput(message.into())
}
fn finite(value: f64, what: &str) -> Result<f64, ComputeError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ComputeError::Execution(format!("nonfinite {what}")))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarHedgeRollout {
    pub pnl: Vec<f64>,
    pub costs: Vec<f64>,
    /// Absolute notional traded at decisions plus terminal liquidation, even when its fee is zero.
    pub turnover: Vec<f64>,
    pub terminal_turnover: Vec<f64>,
}
/// Independent compensated scalar P&L, matching the tensor path-accounting
/// contract. This is floating-point analytical accounting, not the integer ledger.
pub fn scalar_hedge_pnl(
    batch: &HedgePathBatch,
    holdings: &[f64],
    cost: HedgeCostConfig,
    premium: f64,
) -> Result<ScalarHedgeRollout, ComputeError> {
    batch.validate()?;
    let expected = batch
        .paths
        .checked_mul(batch.steps)
        .and_then(|x| x.checked_mul(batch.hedge_count))
        .ok_or_else(|| invalid("hedge holdings shape overflow"))?;
    if holdings.len() != expected
        || holdings.iter().any(|x| !x.is_finite())
        || !premium.is_finite()
        || !cost.proportional.is_finite()
        || cost.proportional < 0.0
        || !cost.terminal_proportional.is_finite()
        || cost.terminal_proportional < 0.0
    {
        return Err(invalid("scalar hedge holdings/cost/premium mismatch"));
    }
    let mut output = ScalarHedgeRollout {
        pnl: Vec::with_capacity(batch.paths),
        costs: Vec::with_capacity(batch.paths),
        turnover: Vec::with_capacity(batch.paths),
        terminal_turnover: Vec::with_capacity(batch.paths),
    };
    for path in 0..batch.paths {
        let (mut pnl, mut fees, mut turns, mut terminal_turns) = (
            CompensatedSum::default(),
            CompensatedSum::default(),
            CompensatedSum::default(),
            CompensatedSum::default(),
        );
        pnl.add(premium);
        pnl.add(-batch.payoffs[path]);
        for time in 0..batch.steps {
            for instrument in 0..batch.hedge_count {
                let i = (path * batch.steps + time) * batch.hedge_count + instrument;
                let now = (path * (batch.steps + 1) + time) * batch.hedge_count + instrument;
                let next = now + batch.hedge_count;
                let previous = if time == 0 {
                    0.0
                } else {
                    holdings[i - batch.hedge_count]
                };
                let traded = finite(
                    (holdings[i] - previous).abs() * batch.marks[now],
                    "turnover",
                )?;
                let charge = finite(traded * cost.proportional, "transaction fee")?;
                turns.add(traded);
                fees.add(charge);
                pnl.add(-charge);
                pnl.add(finite(
                    holdings[i] * (batch.marks[next] - batch.marks[now]),
                    "hedge gain",
                )?);
                if time + 1 == batch.steps {
                    let terminal =
                        finite(holdings[i].abs() * batch.marks[next], "terminal turnover")?;
                    let charge = finite(terminal * cost.terminal_proportional, "terminal fee")?;
                    terminal_turns.add(terminal);
                    turns.add(terminal);
                    fees.add(charge);
                    pnl.add(-charge);
                }
            }
        }
        output.pnl.push(finite(pnl.value(), "total P&L")?);
        output.costs.push(finite(fees.value(), "total cost")?);
        output
            .turnover
            .push(finite(turns.value(), "total turnover")?);
        output
            .terminal_turnover
            .push(finite(terminal_turns.value(), "total terminal turnover")?);
    }
    Ok(output)
}
pub fn no_hedge_pnl(
    batch: &HedgePathBatch,
    premium: f64,
) -> Result<ScalarHedgeRollout, ComputeError> {
    batch.validate()?;
    if !premium.is_finite() {
        return Err(invalid("premium must be finite"));
    }
    Ok(ScalarHedgeRollout {
        pnl: batch
            .payoffs
            .iter()
            .map(|x| finite(premium - x, "unhedged P&L"))
            .collect::<Result<_, _>>()?,
        costs: vec![0.0; batch.paths],
        turnover: vec![0.0; batch.paths],
        terminal_turnover: vec![0.0; batch.paths],
    })
}

/// Content-addressed cache for genuine scalar-f64 baselines. Entries are local
/// immutable numerical artifacts, not an authentication scheme for third-party
/// data. Hashes check compatibility/integrity; no policy outcomes enter the key.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineCacheConfig {
    pub directory: PathBuf,
    #[serde(default)]
    pub mode: BaselineCacheMode,
    #[serde(default = "default_cache_entry_bytes")]
    pub max_entry_bytes: u64,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineCacheMode {
    #[default]
    ReadWrite,
    ReadOnly,
}
fn default_cache_entry_bytes() -> u64 {
    64 * 1024 * 1024
}
impl BaselineCacheConfig {
    pub fn validate(&self) -> Result<(), ComputeError> {
        if self.directory.as_os_str().is_empty()
            || !(16384..=1024 * 1024 * 1024).contains(&self.max_entry_bytes)
        {
            return Err(invalid(
                "baseline cache needs a directory and16KiB..=1GiB entry cap",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct BaselineCacheAccess {
    pub key: Option<String>,
    pub hit: bool,
    pub written: bool,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub payload_sha256: Option<String>,
    pub actual_function_evaluations: u64,
    pub entry_hits: usize,
    pub entry_misses: usize,
    pub entries_written: usize,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheHeader {
    schema: String,
    key: String,
    engine_sha256: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheFooter {
    payload_sha256: String,
}

/// Stable numerical policy identity deliberately excludes worker counts,
/// admission budgets and policy architecture. Scalar CPU execution is the same
/// mathematical computation when those operational settings differ.
pub fn baseline_numeric_policy_hash(
    config: &ModelHedgeConfig,
    heston: bool,
) -> Result<String, ComputeError> {
    config.validate()?;
    if !heston {
        return cache_hash(&("analytic_bs_cpu_f64", baseline_engine_hash()));
    }
    if config.heston_method != HestonBaselineMethod::AdaptiveScalar {
        return cache_hash(&("device_fourier", baseline_engine_hash(), config));
    }
    cache_hash(&(
        "adaptive_heston_cpu_f64",
        baseline_engine_hash(),
        config.adaptive,
        config.tolerances,
    ))
}
fn baseline_engine_hash() -> &'static str {
    static HASH: OnceLock<String> = OnceLock::new();
    HASH.get_or_init(|| {
        let mut hash = Sha256::new();
        hash.update(b"volt.baseline.engine.v1\0");
        for bytes in [
            include_bytes!("model_hedge.rs").as_slice(),
            include_bytes!("heston_adaptive.rs").as_slice(),
            include_bytes!("market_paths.rs").as_slice(),
            include_bytes!("aad.rs").as_slice(),
            include_bytes!("pricing.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
            include_bytes!("../../../rust-toolchain.toml").as_slice(),
        ] {
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
        hash.update(std::env::consts::OS);
        hash.update(std::env::consts::ARCH);
        format!("{:x}", hash.finalize())
    })
    .as_str()
}
fn cache_hash<T: Serialize>(value: &T) -> Result<String, ComputeError> {
    let bytes = serde_json::to_vec(value).map_err(cache_error)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
fn cache_entry_byte_limit(
    paths: usize,
    steps: usize,
    hedges: usize,
    sources: usize,
    cache: &BaselineCacheConfig,
) -> Result<u64, ComputeError> {
    cache.validate()?;
    let floats = paths
        .checked_mul(steps)
        .and_then(|n| n.checked_mul(hedges))
        .and_then(|n| n.checked_add(paths))
        .ok_or_else(|| invalid("cache shape overflow"))?;
    Ok((floats as u64)
        .checked_mul(32)
        .and_then(|n| n.checked_add((sources as u64).checked_mul(128)?))
        .and_then(|n| n.checked_add(32768))
        .ok_or_else(|| invalid("cache byte limit overflow"))?
        .min(cache.max_entry_bytes))
}
/// Serialized read buffer and conservative JSON parser/vector-growth headroom.
/// Expected decoded outputs are additionally in the normal baseline plan.
/// Streaming writes need only an8KiB buffer. This is admission, not an OS quota.
pub fn baseline_cache_working_bytes(
    paths: usize,
    steps: usize,
    hedges: usize,
    sources: usize,
    cache: &BaselineCacheConfig,
) -> Result<u64, ComputeError> {
    cache_entry_byte_limit(paths, steps, hedges, sources, cache)?
        .checked_mul(24)
        .and_then(|n| n.checked_add(32768))
        .ok_or_else(|| invalid("cache parser working estimate overflow"))
}

/// Cache independent scalar call components. Multiple market systems and
/// normalized spreads are assembled linearly from the same immutable call
/// artifacts, so a one-system cohort is reusable inside a larger system set.
/// Puts and straddles reuse the call artifacts with exact r=q=0 parity; cash
/// offsets enter model value and stock offsets enter holdings, never fees.
/// Access hashes for composed results identify an ordered component manifest.
pub fn model_hedge_holdings_cached(
    markets: &[MarketPaths],
    liability: HedgeLiability,
    config: &ModelHedgeConfig,
    cache: Option<&BaselineCacheConfig>,
) -> Result<(ModelHedgeResult, BaselineCacheAccess), ComputeError> {
    let Some(cache) = cache else {
        return single_model_hedge_holdings_cached(markets, liability, config, None);
    };
    let (heston, legs) = validate_model_inputs(markets, liability, config)?;
    cache.validate()?;
    if heston && config.heston_method != HestonBaselineMethod::AdaptiveScalar {
        return Err(ComputeError::Unsupported(
            "persistent cache supports scalar-f64 BS/adaptive Heston only".into(),
        ));
    }
    if legs.is_empty() {
        // The exact empty liability has zero model holdings and needs no oracle
        // artifact. Ordinary model input and memory admission still apply.
        return single_model_hedge_holdings_cached(markets, liability, config, None);
    }
    if markets.len() == 1 && matches!(liability, HedgeLiability::Call { .. }) {
        let (r, a) = single_model_hedge_holdings_cached(markets, liability, config, Some(cache))?;
        return Ok((r, entry_counts(a)));
    }
    if legs.iter().any(|(_, w)| !w.is_finite()) {
        return Err(invalid("nonfinite cached liability weight"));
    }
    let first = &markets[0];
    let per = if heston { 2 } else { 1 };
    let hedges = markets
        .len()
        .checked_mul(per)
        .ok_or_else(|| invalid("composed cache shape overflow"))?;
    let length = first
        .config
        .paths
        .checked_mul(first.config.steps)
        .and_then(|n| n.checked_mul(hedges))
        .ok_or_else(|| invalid("composed cache shape overflow"))?;
    let extra = (length as u64)
        .checked_add(first.config.paths as u64)
        .and_then(|n| n.checked_mul(24))
        .ok_or_else(|| invalid("composed cache workspace overflow"))?;
    let available = usable_host(
        config.budget.host_working_bytes,
        config.budget.headroom_percent,
    );
    let cache_work = baseline_cache_working_bytes(
        first.config.paths,
        first.config.steps,
        hedges,
        markets.len(),
        cache,
    )?;
    if extra.checked_add(cache_work).is_none_or(|n| n >= available) {
        return Err(ComputeError::BudgetExceeded(
            "composed cache accumulators and read workspace exceed host allowance".into(),
        ));
    }
    let mut inner = config.clone();
    inner.budget.host_working_bytes = (((available - extra) as u128) * 100
        / u128::from(100 - config.budget.headroom_percent))
    .min(u128::from(u64::MAX)) as u64;
    let mut holdings: Vec<_> = std::iter::repeat_with(CompensatedSum::default)
        .take(length)
        .collect();
    let mut prices: Vec<_> = std::iter::repeat_with(CompensatedSum::default)
        .take(first.config.paths)
        .collect();
    let mut validation = BaselineValidation::default();
    let mut access = BaselineCacheAccess::default();
    let mut manifest = Sha256::new();
    if parity_strike(liability).is_some() {
        manifest.update(b"volt.baseline.parity_component_manifest.v1\0");
        manifest.update(cache_hash(&liability)?.as_bytes());
    } else {
        manifest.update(b"volt.baseline.component_manifest.v1\0");
    }
    let mut component_bounds = vec![[0.0f64; 3]; markets.len()];
    for (component, market) in markets.iter().enumerate() {
        for (strike, weight) in &legs {
            let (part, part_access) = single_model_hedge_holdings_cached(
                std::slice::from_ref(market),
                HedgeLiability::Call { strike: *strike },
                &inner,
                Some(cache),
            )?;
            let part_access = entry_counts(part_access);
            for (path, price) in prices.iter_mut().enumerate() {
                price.add(weight * part.initial_model_prices[path]);
                for step in 0..first.config.steps {
                    for instrument in 0..per {
                        let to = (path * first.config.steps + step) * hedges
                            + component * per
                            + instrument;
                        let from = (path * first.config.steps + step) * per + instrument;
                        holdings[to].add(weight * part.holdings[from]);
                    }
                }
            }
            for (i, bound) in component_bounds[component].iter_mut().enumerate() {
                *bound = finite(
                    *bound + weight.abs() * part.validation.adaptive_max_estimated_errors[i],
                    "composed cache error",
                )?;
            }
            merge_validation(&mut validation, &part.validation)?;
            access.entry_hits += part_access.entry_hits;
            access.entry_misses += part_access.entry_misses;
            access.entries_written += part_access.entries_written;
            access.bytes_read = access
                .bytes_read
                .checked_add(part_access.bytes_read)
                .ok_or_else(|| invalid("cache byte counter overflow"))?;
            access.bytes_written = access
                .bytes_written
                .checked_add(part_access.bytes_written)
                .ok_or_else(|| invalid("cache byte counter overflow"))?;
            access.actual_function_evaluations = access
                .actual_function_evaluations
                .checked_add(part_access.actual_function_evaluations)
                .ok_or_else(|| invalid("cache work counter overflow"))?;
            manifest.update((component as u64).to_le_bytes());
            manifest.update(strike.to_le_bytes());
            manifest.update(weight.to_le_bytes());
            manifest.update(part_access.key.unwrap().as_bytes());
            manifest.update(part_access.payload_sha256.unwrap().as_bytes());
        }
    }
    validation.composed_error_bounds = [
        component_bounds.iter().map(|e| e[0]).sum(),
        component_bounds.iter().map(|e| e[1]).fold(0.0, f64::max),
        component_bounds.iter().map(|e| e[2]).fold(0.0, f64::max),
    ];
    let budgets = [
        config.tolerances.price_absolute,
        config.tolerances.delta_absolute,
        config.tolerances.variance_absolute,
    ];
    if validation
        .composed_error_bounds
        .iter()
        .zip(budgets)
        .any(|(e, t)| !e.is_finite() || *e > t)
    {
        return Err(ComputeError::Execution("cached component composition exceeds absolute model error allowance; tighten component tolerances".into()));
    }
    validation.estimated_additional_host_bytes = validation
        .estimated_additional_host_bytes
        .checked_add(extra)
        .ok_or_else(|| invalid("composed host counter overflow"))?;
    let hash = format!("{:x}", manifest.finalize());
    access.key = Some(hash.clone());
    access.payload_sha256 = Some(hash);
    access.hit = access.entry_misses == 0;
    access.written = access.entries_written > 0;
    let mut result = ModelHedgeResult {
        schema: "volt.compute.model_hedge.v1".into(),
        paths: first.config.paths,
        steps: first.config.steps,
        hedge_count: hedges,
        holdings: holdings
            .into_iter()
            .map(CompensatedSum::value)
            .map(|v| finite(v, "composed holdings"))
            .collect::<Result<_, _>>()?,
        initial_model_prices: prices
            .into_iter()
            .map(CompensatedSum::value)
            .map(|v| finite(v, "composed prices"))
            .collect::<Result<_, _>>()?,
        market_source_hashes: markets.iter().map(|m| m.content_hash.clone()).collect(),
        liability,
        config: config.clone(),
        execution_policy: if heston {
            BaselineExecutionPolicy::AdaptiveScalarCpuF64
        } else {
            BaselineExecutionPolicy::AnalyticBlackScholesCpuF64
        },
        validation,
    };
    apply_parity_adjustment(markets, &mut result)?;
    Ok((result, access))
}
fn entry_counts(mut a: BaselineCacheAccess) -> BaselineCacheAccess {
    if a.key.is_some() {
        a.entry_hits = usize::from(a.hit);
        a.entry_misses = usize::from(!a.hit);
        a.entries_written = usize::from(a.written);
    }
    a
}
fn merge_validation(
    to: &mut BaselineValidation,
    from: &BaselineValidation,
) -> Result<(), ComputeError> {
    macro_rules! add{($($field:ident),*)=>{$(to.$field=to.$field.checked_add(from.$field).ok_or_else(||invalid("composed validation counter overflow"))?;)*};}
    add!(
        requested_call_queries,
        priced_call_queries,
        exact_cache_hits,
        coalesced_queries,
        pricing_jobs,
        device_batches,
        adaptive_converged_queries,
        adaptive_tail_checked_queries,
        adaptive_analytic_tail_queries,
        adaptive_function_evaluations,
        adaptive_panels
    );
    macro_rules! maximum{($($field:ident),*)=>{$(to.$field=to.$field.max(from.$field);)*};}
    maximum!(
        adaptive_threads_used,
        adaptive_max_cutoff,
        max_price_node_difference,
        max_price_cutoff_difference,
        max_delta_node_difference,
        max_delta_cutoff_difference,
        max_variance_node_difference,
        max_variance_cutoff_difference,
        estimated_additional_host_bytes
    );
    for i in 0..3 {
        to.adaptive_max_estimated_errors[i] =
            to.adaptive_max_estimated_errors[i].max(from.adaptive_max_estimated_errors[i]);
        to.adaptive_max_tail_estimates[i] =
            to.adaptive_max_tail_estimates[i].max(from.adaptive_max_tail_estimates[i]);
    }
    Ok(())
}

fn single_model_hedge_holdings_cached(
    markets: &[MarketPaths],
    liability: HedgeLiability,
    config: &ModelHedgeConfig,
    cache: Option<&BaselineCacheConfig>,
) -> Result<(ModelHedgeResult, BaselineCacheAccess), ComputeError> {
    let Some(cache) = cache else {
        let result = model_hedge_holdings(markets, liability, config)?;
        let access = BaselineCacheAccess {
            key: None,
            hit: false,
            written: false,
            bytes_read: 0,
            bytes_written: 0,
            payload_sha256: None,
            actual_function_evaluations: result.validation.adaptive_function_evaluations,
            ..BaselineCacheAccess::default()
        };
        return Ok((result, access));
    };
    cache.validate()?;
    let (heston, _) = validate_model_inputs(markets, liability, config)?;
    if heston && config.heston_method != HestonBaselineMethod::AdaptiveScalar {
        return Err(ComputeError::Unsupported(
            "persistent cache supports scalar-f64 BS/adaptive Heston only".into(),
        ));
    }
    let first = &markets[0];
    let hedges = markets
        .len()
        .checked_mul(if heston { 2 } else { 1 })
        .ok_or_else(|| invalid("cache shape overflow"))?;
    let policy = baseline_numeric_policy_hash(config, heston)?;
    let sources: Vec<_> = markets.iter().map(|m| m.content_hash.clone()).collect();
    let key = cache_hash(&("volt.baseline.cache.v1", &policy, liability, &sources))?;
    let path = cache.directory.join(format!("{key}.baseline-v1"));
    let limit = cache_entry_byte_limit(
        first.config.paths,
        first.config.steps,
        hedges,
        markets.len(),
        cache,
    )?;
    let output = (first.config.paths as u64)
        .checked_mul(first.config.steps as u64)
        .and_then(|n| n.checked_mul(hedges as u64))
        .and_then(|n| n.checked_add(first.config.paths as u64))
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| invalid("cache output size overflow"))?;
    let cache_working = baseline_cache_working_bytes(
        first.config.paths,
        first.config.steps,
        hedges,
        markets.len(),
        cache,
    )?;
    if cache_working.checked_add(output).is_none_or(|n| {
        n > usable_host(
            config.budget.host_working_bytes,
            config.budget.headroom_percent,
        )
    }) {
        return Err(ComputeError::BudgetExceeded(
            "baseline cache read/result working allowance exceeded".into(),
        ));
    }
    match File::open(&path) {
        Ok(file) => {
            let (result, bytes, checksum) =
                read_cache(file, limit, &key, markets, liability, config)?;
            return Ok((
                result,
                BaselineCacheAccess {
                    key: Some(key),
                    hit: true,
                    written: false,
                    bytes_read: bytes,
                    bytes_written: 0,
                    payload_sha256: Some(checksum),
                    actual_function_evaluations: 0,
                    ..BaselineCacheAccess::default()
                },
            ));
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(cache_error(e)),
    }
    if cache.mode == BaselineCacheMode::ReadOnly {
        return Err(invalid("required baseline cache entry is missing"));
    }
    if cache.directory.exists() && !cache.directory.is_dir() {
        return Err(invalid("baseline cache location is not a directory"));
    }
    let result = model_hedge_holdings(markets, liability, config)?;
    let actual = result.validation.adaptive_function_evaluations;
    std::fs::create_dir_all(&cache.directory).map_err(cache_error)?;
    let (bytes, checksum, published) = write_cache(&path, &key, &result, limit)?;
    if !published {
        let (result, read_bytes, checksum) = read_cache(
            File::open(&path).map_err(cache_error)?,
            limit,
            &key,
            markets,
            liability,
            config,
        )?;
        return Ok((
            result,
            BaselineCacheAccess {
                key: Some(key),
                hit: false,
                written: false,
                bytes_read: read_bytes,
                bytes_written: 0,
                payload_sha256: Some(checksum),
                actual_function_evaluations: actual,
                ..BaselineCacheAccess::default()
            },
        ));
    }
    Ok((
        result,
        BaselineCacheAccess {
            key: Some(key),
            hit: false,
            written: true,
            bytes_read: 0,
            bytes_written: bytes,
            payload_sha256: Some(checksum),
            actual_function_evaluations: actual,
            ..BaselineCacheAccess::default()
        },
    ))
}
fn read_cache(
    file: File,
    limit: u64,
    key: &str,
    markets: &[MarketPaths],
    liability: HedgeLiability,
    config: &ModelHedgeConfig,
) -> Result<(ModelHedgeResult, u64, String), ComputeError> {
    let length = file.metadata().map_err(cache_error)?.len();
    if length > limit {
        return Err(ComputeError::BudgetExceeded(
            "baseline cache entry exceeds declared/shape byte cap".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(cache_error)?;
    if bytes.len() as u64 > limit {
        return Err(ComputeError::BudgetExceeded(
            "baseline cache entry grew beyond byte cap".into(),
        ));
    }
    let lines: Vec<_> = bytes.splitn(4, |b| *b == b'\n').collect();
    if lines.len() != 4 || !lines[3].is_empty() || lines[0].len() > 4096 || lines[2].len() > 512 {
        return Err(invalid("invalid baseline cache framing"));
    }
    let header: CacheHeader = serde_json::from_slice(lines[0]).map_err(cache_error)?;
    if header.schema != "volt.compute.baseline_cache.v1"
        || header.key != key
        || header.engine_sha256 != baseline_engine_hash()
    {
        return Err(invalid("baseline cache schema/key/engine mismatch"));
    }
    let footer: CacheFooter = serde_json::from_slice(lines[2]).map_err(cache_error)?;
    let checksum = format!("{:x}", Sha256::digest(lines[1]));
    if checksum != footer.payload_sha256 {
        return Err(invalid("baseline cache payload checksum mismatch"));
    }
    let result: ModelHedgeResult = serde_json::from_slice(lines[1]).map_err(cache_error)?;
    validate_cached_result(&result, markets, liability, config)?;
    Ok((result, bytes.len() as u64, checksum))
}
fn validate_cached_result(
    r: &ModelHedgeResult,
    markets: &[MarketPaths],
    liability: HedgeLiability,
    c: &ModelHedgeConfig,
) -> Result<(), ComputeError> {
    let first = &markets[0];
    let heston = matches!(first.config.model, MarketModel::Heston { .. });
    let hedges = markets.len() * (if heston { 2 } else { 1 });
    let expected = first
        .config
        .paths
        .checked_mul(first.config.steps)
        .and_then(|n| n.checked_mul(hedges))
        .ok_or_else(|| invalid("cache shape overflow"))?;
    if r.schema != "volt.compute.model_hedge.v1"
        || r.paths != first.config.paths
        || r.steps != first.config.steps
        || r.hedge_count != hedges
        || r.holdings.len() != expected
        || r.initial_model_prices.len() != r.paths
        || r.holdings
            .iter()
            .chain(&r.initial_model_prices)
            .any(|v| !v.is_finite())
        || r.market_source_hashes
            != markets
                .iter()
                .map(|m| m.content_hash.clone())
                .collect::<Vec<_>>()
        || cache_hash(&r.liability)? != cache_hash(&liability)?
        || baseline_numeric_policy_hash(&r.config, heston)?
            != baseline_numeric_policy_hash(c, heston)?
    {
        return Err(invalid(
            "baseline cache payload shape/source/numerical-policy mismatch",
        ));
    }
    let execution = if heston {
        BaselineExecutionPolicy::AdaptiveScalarCpuF64
    } else {
        BaselineExecutionPolicy::AnalyticBlackScholesCpuF64
    };
    if cache_hash(&r.execution_policy)? != cache_hash(&execution)? {
        return Err(invalid("baseline cache producer execution mismatch"));
    }
    if heston {
        let v = &r.validation;
        let t = [
            c.adaptive.price_absolute_tolerance,
            c.adaptive.delta_absolute_tolerance,
            c.adaptive.variance_absolute_tolerance,
        ];
        if v.priced_call_queries != v.adaptive_converged_queries
            || v.adaptive_converged_queries != v.adaptive_tail_checked_queries
            || (0..3).any(|i| {
                !v.adaptive_max_estimated_errors[i].is_finite()
                    || v.adaptive_max_estimated_errors[i] < 0.0
                    || v.adaptive_max_estimated_errors[i] > t[i]
                    || !v.adaptive_max_tail_estimates[i].is_finite()
                    || v.adaptive_max_tail_estimates[i] < 0.0
                    || v.adaptive_max_tail_estimates[i] > v.adaptive_max_estimated_errors[i]
            })
        {
            return Err(invalid(
                "baseline cache producer numerical validation mismatch",
            ));
        }
    }
    Ok(())
}
struct CachePayloadWriter<'a> {
    inner: &'a mut BufWriter<File>,
    hash: Sha256,
    bytes: u64,
    limit: u64,
}
impl Write for CachePayloadWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .bytes
            .checked_add(bytes.len() as u64)
            .is_none_or(|n| n > self.limit)
        {
            return Err(std::io::Error::other(
                "baseline cache serialized payload exceeds byte cap",
            ));
        }
        let n = self.inner.write(bytes)?;
        self.hash.update(&bytes[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
fn write_cache(
    path: &Path,
    key: &str,
    result: &ModelHedgeResult,
    limit: u64,
) -> Result<(u64, String, bool), ComputeError> {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let temporary = path.with_extension(format!(
        "partial-{}-{}",
        std::process::id(),
        NONCE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut created = false;
    let outcome = (|| {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(cache_error)?;
        created = true;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(
            &mut writer,
            &CacheHeader {
                schema: "volt.compute.baseline_cache.v1".into(),
                key: key.into(),
                engine_sha256: baseline_engine_hash().into(),
            },
        )
        .map_err(cache_error)?;
        writer.write_all(b"\n").map_err(cache_error)?;
        let mut payload = CachePayloadWriter {
            inner: &mut writer,
            hash: Sha256::new(),
            bytes: 0,
            limit: limit.saturating_sub(8192),
        };
        serde_json::to_writer(&mut payload, result).map_err(cache_error)?;
        let checksum = format!("{:x}", payload.hash.finalize());
        writer.write_all(b"\n").map_err(cache_error)?;
        serde_json::to_writer(
            &mut writer,
            &CacheFooter {
                payload_sha256: checksum.clone(),
            },
        )
        .map_err(cache_error)?;
        writer.write_all(b"\n").map_err(cache_error)?;
        writer.flush().map_err(cache_error)?;
        writer.get_ref().sync_all().map_err(cache_error)?;
        let bytes = writer.get_ref().metadata().map_err(cache_error)?.len();
        drop(writer);
        match std::fs::hard_link(&temporary, path) {
            Ok(()) => Ok((bytes, checksum, true)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok((0, checksum, false)),
            Err(e) => Err(cache_error(e)),
        }
    })();
    if created {
        let _ = std::fs::remove_file(&temporary);
    }
    outcome
}
fn cache_error(e: impl std::fmt::Display) -> ComputeError {
    ComputeError::Execution(format!("baseline cache: {e}"))
}
