//! Frozen-checkpoint comparisons on identical held-out market paths. Candidate,
//! genuine model hedge and no-hedge P&Ls use independent scalar f64 accounting;
//! device-accounting differences are reported separately. This runner never
//! trains, calibrates BatchNorm, selects a checkpoint, or substitutes a baseline.
use crate::{
    ComputeError, ExecutionTarget, Session,
    chunked::{CompensatedSum, execution_guard},
    hedge_application::{self, HedgeMemoryPlan, HedgePathBatch, PreparedHedgePaths},
    hedge_policy::HedgePolicy,
    hedge_reconciliation::{AuditModelFamily, ReconciliationAudit, reconcile_path},
    hedge_training::{
        HedgeTrainingConfig, TrainingCheckpoint, generate_markets_with_substeps, validate_resume,
    },
    market_paths::MarketModel,
    model_hedge::{
        BaselineCacheConfig, BaselineExecutionPolicy, BaselineValidation, ModelHedgeConfig,
        ScalarHedgeRollout, baseline_cache_working_bytes, baseline_numeric_policy_hash,
        model_hedge_holdings_cached, no_hedge_pnl, scalar_hedge_pnl,
    },
    risk_objective::{self, BootstrapConfig, PairedPathComparison, RiskEvaluation, RiskObjective},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const SYSTEM_SEED_STRIDE: u64 = 0x9e3779b97f4a7c15;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkConfig {
    pub target: ExecutionTarget,
    pub paths: usize,
    pub seed: u64,
    pub batch_paths: usize,
    pub model: ModelHedgeConfig,
    pub bootstrap_paths: usize,
    pub bootstrap_replicates: usize,
    pub audit_paths: usize,
    pub output_samples: Option<PathBuf>,
    #[serde(default)]
    pub baseline_cache: Option<BaselineCacheConfig>,
    #[serde(default)]
    pub terminal_pnl_output: Option<PathBuf>,
    #[serde(default = "default_substeps")]
    pub simulation_substeps: usize,
}
fn default_substeps() -> usize {
    1
}
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkPlan {
    pub policy: HedgeMemoryPlan,
    pub batches: usize,
    pub global_results_risk_bootstrap_audits_bytes: u64,
    pub retained_market_and_rollout_bytes: u64,
    pub baseline_fixed_host_bytes: u64,
    pub cache_working_bytes: u64,
    pub terminal_artifact_working_bytes: u64,
    pub resolution_working_bytes: u64,
    pub overall_host_admission_limit_bytes: u64,
    pub effective_model: ModelHedgeConfig,
    pub market_cohort_spec_sha256: String,
    pub model_numeric_policy_sha256: String,
    pub selected_policy_sha256: String,
    pub estimated_peak_host_bytes: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct StrategyMetrics {
    pub mean_cost: f64,
    pub mean_turnover: f64,
    pub mean_terminal_turnover: f64,
    pub max_absolute_holding: f64,
}
#[derive(Debug, Clone, Serialize)]
pub struct ArithmeticGap {
    pub convention: String,
    pub paths: usize,
    pub mean_signed_difference: f64,
    pub root_mean_square_difference: f64,
    pub max_absolute_difference: f64,
}
#[derive(Debug, Serialize)]
pub struct BenchmarkPathAudit {
    pub path_id: u64,
    pub candidate_device_pnl: f64,
    pub candidate_f64_identity_pnl: f64,
    pub model_f64_pnl: f64,
    pub candidate: ReconciliationAudit,
    pub model: ReconciliationAudit,
}
#[derive(Debug, Clone, Serialize)]
pub struct SampleArtifact {
    pub path: PathBuf,
    pub rows: usize,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct BaselineCacheSummary {
    pub enabled: bool,
    pub hits: usize,
    pub misses: usize,
    pub entries_written: usize,
    pub bytes_read: u64,
    pub bytes_written: u64,
    /// Current run work, excluding numerical work recorded in cached producers.
    pub actual_function_evaluations: u64,
    pub manifest_sha256: String,
}
#[derive(Debug, Serialize)]
pub struct BenchmarkReport {
    pub schema: String,
    pub config: BenchmarkConfig,
    pub training_config: HedgeTrainingConfig,
    pub plan: BenchmarkPlan,
    pub selected_update: u64,
    pub selected_validation_risk: f64,
    pub checkpoint_sha256: String,
    pub checkpoint_hash_convention: String,
    pub selected_policy_sha256: String,
    pub inference_pure: bool,
    pub paths: usize,
    pub first_path_id: u64,
    pub last_path_id: u64,
    pub derived_system_seeds: Vec<u64>,
    pub path_stream_hash: String,
    pub market_stream_hash: String,
    pub path_hash_convention: String,
    pub identical_paths_for_all_strategies: bool,
    pub candidate: RiskEvaluation,
    pub model: RiskEvaluation,
    pub no_hedge: RiskEvaluation,
    pub candidate_device: RiskEvaluation,
    pub candidate_device_accounting_gap: ArithmeticGap,
    pub candidate_metrics: StrategyMetrics,
    pub model_metrics: StrategyMetrics,
    pub no_hedge_metrics: StrategyMetrics,
    pub paired_candidate_vs_model: PairedPathComparison,
    pub paired_candidate_vs_no_hedge: PairedPathComparison,
    pub bootstrap_subset_paths: usize,
    pub bootstrap_subset_selection: String,
    pub model_execution_policy: BaselineExecutionPolicy,
    pub model_validation: Vec<BaselineValidation>,
    pub mean_initial_model_liability_price: f64,
    pub common_premium: f64,
    pub audits: Vec<BenchmarkPathAudit>,
    pub audits_exact_reconciled: bool,
    pub audits_quantization_within_bounds: bool,
    pub sample_artifact: Option<SampleArtifact>,
    pub terminal_pnl_artifact: Option<SampleArtifact>,
    pub baseline_cache: BaselineCacheSummary,
    pub completed_batches: usize,
    pub seconds: f64,
    pub interpretation: String,
}

/// Pure preflight: validates both snapshots/configs and reserved streams before
/// creating a session, allocating global result vectors, or touching an output.
pub fn plan_benchmark(
    checkpoint: &TrainingCheckpoint,
    config: &BenchmarkConfig,
) -> Result<BenchmarkPlan, ComputeError> {
    if !(2..=1_000_000).contains(&config.paths)
        || config.batch_paths == 0
        || config.bootstrap_paths < 2
        || config.bootstrap_paths > config.paths
        || !(2..=10_000).contains(&config.bootstrap_replicates)
        || config.audit_paths == 0
        || config.audit_paths > config.paths
        || config.audit_paths > 64
    {
        return Err(invalid(
            "benchmark needs2..=1m paths, positive batch,2..=paths bootstrap subset,2..=10000 replicates and1..=min(paths,64) audits",
        ));
    }
    checkpoint.config.validate()?;
    validate_resume(&checkpoint.config, checkpoint)?;
    config.model.validate()?;
    let training = &checkpoint.config;
    let mut reserved = BTreeSet::new();
    let mut bases = vec![training.market.seed, training.validation_seed];
    if training.calibration_paths > 0 && training.policy.batch_norm.enabled {
        bases.push(training.calibration_seed);
    }
    for base in bases {
        for system in 0..training.systems {
            reserved.insert(base.wrapping_add((system as u64).wrapping_mul(SYSTEM_SEED_STRIDE)));
        }
    }
    let mut evaluated = BTreeSet::new();
    for system in 0..training.systems {
        let seed = config
            .seed
            .wrapping_add((system as u64).wrapping_mul(SYSTEM_SEED_STRIDE));
        if reserved.contains(&seed) || !evaluated.insert(seed) {
            return Err(invalid(
                "held-out benchmark stream overlaps training, calibration or validation data",
            ));
        }
    }
    let policy = hedge_application::plan_hedging(
        &training.policy,
        config.batch_paths,
        training.precision,
        config.target,
        &training.budget,
        false,
    )?;
    let cache_working = if let Some(cache) = &config.baseline_cache {
        baseline_cache_working_bytes(
            config.batch_paths,
            training.policy.time_steps,
            training.policy.hedge_count,
            training.systems,
            cache,
        )?
    } else {
        0
    };
    let terminal_working = if config.terminal_pnl_output.is_some() {
        128 * 1024
    } else {
        0
    };
    for path in [&config.output_samples, &config.terminal_pnl_output]
        .into_iter()
        .flatten()
    {
        if path.file_name().is_none() {
            return Err(invalid("benchmark artifact path needs a file name"));
        }
    }
    if config.output_samples.is_some() && config.output_samples == config.terminal_pnl_output {
        return Err(invalid("JSONL and terminal-PnL artifact paths must differ"));
    }
    let decisions = training
        .policy
        .time_steps
        .checked_mul(training.policy.hedge_count)
        .ok_or_else(overflow)? as u64;
    let audits = (config.audit_paths as u64)
        .checked_mul(2)
        .and_then(|n| n.checked_mul(16384_u64.checked_add(decisions.checked_mul(4096)?)?))
        .ok_or_else(overflow)?;
    // Four global P&L vectors plus nonlinear-risk sort/weight workspaces. Paired
    // bootstrap buffers and both ledgers are independently reserved as well.
    let global = (config.paths as u64)
        .checked_mul(96)
        .and_then(|n| n.checked_add((config.bootstrap_paths as u64).checked_mul(96)?))
        .and_then(|n| n.checked_add((config.bootstrap_replicates as u64).checked_mul(32)?))
        .and_then(|n| n.checked_add(audits))
        .and_then(|n| n.checked_add(65536))
        .and_then(|n| {
            n.checked_add(if config.output_samples.is_some() {
                decisions.checked_mul(128)?.checked_add(8192)?
            } else {
                0
            })
        })
        .and_then(|n| n.checked_add(terminal_working))
        .ok_or_else(overflow)?;
    let mut market = training.market.clone();
    market.paths = config.batch_paths;
    market.first_path_id = 0;
    market.seed = config.seed;
    let source = market.validate()?;
    let resolution = crate::market_path_resolution::MarketPathResolutionConfig {
        market: market.clone(),
        substeps: config.simulation_substeps,
    }
    .validate()?;
    let resolution_working = resolution
        .working_bytes
        .checked_mul(training.systems as u64)
        .ok_or_else(overflow)?;
    let coarse_generation = source
        .output_bytes
        .checked_mul(training.systems as u64)
        .and_then(|n| n.checked_add(source.input_bytes))
        .ok_or_else(overflow)?;
    let resolution_extra = if config.simulation_substeps == 1 {
        0
    } else {
        resolution_working.saturating_sub(coarse_generation)
    };
    let retained = source
        .output_bytes
        .checked_mul(training.systems as u64)
        .and_then(|n| n.checked_add(source.input_bytes))
        .and_then(|n| {
            n.checked_add(
                (config.batch_paths as u64)
                    .checked_mul(decisions.checked_mul(16)?.checked_add(256)?)?,
            )
        })
        .ok_or_else(overflow)?;
    let retained = retained
        .checked_add(cache_working)
        .and_then(|n| n.checked_add(resolution_extra))
        .ok_or_else(overflow)?;
    let limit = usable(
        training.budget.host_working_bytes,
        training.budget.headroom_percent,
    );
    let candidate_peak = global
        .checked_add(retained)
        .and_then(|n| n.checked_add(policy.estimated_host_working_bytes))
        .ok_or_else(overflow)?;
    if candidate_peak > limit {
        return Err(ComputeError::BudgetExceeded(
            "benchmark global results/audits plus candidate workspace exceed host budget".into(),
        ));
    }
    let available = limit
        .checked_sub(global)
        .and_then(|n| n.checked_sub(retained))
        .ok_or_else(overflow)?;
    let mut effective = config.model.clone();
    let model_limit = usable(
        config.model.budget.host_working_bytes,
        config.model.budget.headroom_percent,
    )
    .min(available);
    let raw = ((model_limit as u128) * 100 / u128::from(100 - effective.budget.headroom_percent))
        .min(u128::from(u64::MAX)) as u64;
    effective.budget.host_working_bytes = raw.min(effective.budget.host_working_bytes);
    let is_heston = matches!(training.market.model, MarketModel::Heston { .. });
    let baseline_per_path = decisions
        .checked_mul(8)
        .and_then(|v| v.checked_add(8))
        .ok_or_else(overflow)?;
    let baseline_fixed = (config.batch_paths as u64)
        .checked_mul(baseline_per_path)
        .ok_or_else(overflow)?
        .checked_add(if is_heston {
            (effective.max_cached_states as u64)
                .checked_mul(512)
                .ok_or_else(overflow)?
        } else {
            0
        })
        .ok_or_else(overflow)?
        .checked_add(
            (effective.max_queries_per_job as u64)
                .checked_mul(8192)
                .ok_or_else(overflow)?,
        )
        .ok_or_else(overflow)?
        .checked_add(
            (effective.adaptive_threads as u64)
                .checked_mul(1024 * 1024)
                .ok_or_else(overflow)?,
        )
        .ok_or_else(overflow)?
        .checked_add(32768)
        .ok_or_else(overflow)?;
    if baseline_fixed
        >= usable(
            effective.budget.host_working_bytes,
            effective.budget.headroom_percent,
        )
    {
        return Err(ComputeError::BudgetExceeded("benchmark baseline outputs/cache/worker reserve cannot fit after global results/audits".into()));
    }
    if config
        .output_samples
        .as_ref()
        .is_some_and(|p| p.file_name().is_none())
    {
        return Err(invalid("benchmark samples path needs a file name"));
    }
    Ok(BenchmarkPlan {
        policy,
        batches: config.paths.div_ceil(config.batch_paths),
        global_results_risk_bootstrap_audits_bytes: global,
        retained_market_and_rollout_bytes: retained,
        baseline_fixed_host_bytes: baseline_fixed,
        cache_working_bytes: cache_working,
        terminal_artifact_working_bytes: terminal_working,
        resolution_working_bytes: resolution_working,
        overall_host_admission_limit_bytes: limit,
        estimated_peak_host_bytes: candidate_peak.max(global + retained + model_limit),
        market_cohort_spec_sha256: MarketCohortHasher::new(
            training,
            config.seed,
            config.paths,
            config.simulation_substeps,
        )?
        .spec_sha256()
        .to_owned(),
        model_numeric_policy_sha256: baseline_numeric_policy_hash(&effective, is_heston)?,
        selected_policy_sha256: hash_json(&checkpoint.selected_policy)?,
        effective_model: effective,
    })
}

pub fn benchmark_checkpoint(
    checkpoint: &TrainingCheckpoint,
    config: impl std::borrow::Borrow<BenchmarkConfig>,
) -> Result<BenchmarkReport, ComputeError> {
    let config = config.borrow();
    let started = Instant::now();
    let plan = plan_benchmark(checkpoint, config)?;
    let checkpoint_sha256 = hash_json(checkpoint)?;
    let selected_hash = hash_json(&checkpoint.selected_policy)?;
    let training = &checkpoint.config;
    let mut candidate = Vec::with_capacity(config.paths);
    let mut device = Vec::with_capacity(config.paths);
    let mut model = Vec::with_capacity(config.paths);
    let mut no_hedge = Vec::with_capacity(config.paths);
    let mut candidate_metrics = MetricTotals::default();
    let mut model_metrics = MetricTotals::default();
    let mut gap = GapTotals::default();
    let mut model_prices = CompensatedSum::default();
    let mut audits = Vec::with_capacity(config.audit_paths);
    let mut validations = Vec::with_capacity(plan.batches);
    let mut cache_summary = BaselineCacheSummary {
        enabled: config.baseline_cache.is_some(),
        hits: 0,
        misses: 0,
        entries_written: 0,
        bytes_read: 0,
        bytes_written: 0,
        actual_function_evaluations: 0,
        manifest_sha256: String::new(),
    };
    let mut cache_manifest = Sha256::new();
    cache_manifest.update(b"volt.baseline.cache.manifest.v1\0");
    let mut model_execution = None;
    let mut stream_hash = Sha256::new();
    stream_hash.update(b"volt.benchmark.shared_paths.v1\0");
    let cohort_spec_hash = hash_json(&(
        training.market.model,
        training.market.steps,
        training.market.dt,
        training.market.initial_spot,
        training.market.rate,
        training.market.dividend_yield,
        training.market.integrated_variance_rule,
        training.systems,
        training.liability,
        config.seed,
        config.paths,
        crate::market_paths::RNG_CONTRACT,
    ))?;
    stream_hash.update(cohort_spec_hash.as_bytes());
    if config.simulation_substeps > 1 {
        stream_hash.update(b"volt.market_resolution.substeps.v1\0");
        stream_hash.update((config.simulation_substeps as u64).to_le_bytes());
    }
    let mut market_fingerprint = MarketCohortHasher::new(
        training,
        config.seed,
        config.paths,
        config.simulation_substeps,
    )?;
    let mut samples = config
        .output_samples
        .as_ref()
        .map(|p| AtomicSamples::new(p))
        .transpose()?;
    let mut terminal = config
        .terminal_pnl_output
        .as_ref()
        .map(|path| {
            let header = TerminalPnlHeader {
                schema: "volt.compute.terminal_pnl.v2".into(),
                rows: config.paths,
                first_path_id: 0,
                columns: TERMINAL_PNL_COLUMNS.iter().map(|s| s.to_string()).collect(),
                cohort_spec_sha256: cohort_spec_hash.clone(),
                market_cohort_spec_sha256: market_fingerprint.spec_sha256().to_owned(),
                selected_policy_sha256: selected_hash.clone(),
                checkpoint_sha256: checkpoint_sha256.clone(),
                model_numeric_policy_sha256: baseline_numeric_policy_hash(
                    &plan.effective_model,
                    matches!(training.market.model, MarketModel::Heston { .. }),
                )?,
                common_premium: training.premium_for_comparison,
                cost: training.cost,
                objective: training.objective,
            };
            AtomicTerminalPnl::new(path, &header)
        })
        .transpose()?;
    let family = if matches!(training.market.model, MarketModel::Heston { .. }) {
        AuditModelFamily::Heston
    } else {
        AuditModelFamily::BlackScholes
    };
    let mut completed_batches = 0;
    for first in (0..config.paths).step_by(config.batch_paths) {
        let count = config.batch_paths.min(config.paths - first);
        let markets = generate_markets_with_substeps(
            training,
            config.seed,
            first as u64,
            count,
            config.simulation_substeps,
        )?;
        let batch = HedgePathBatch::from_markets(&markets, training.liability)?;
        market_fingerprint.update_batch(first as u64, &batch)?;
        // The complete guard and every policy tensor end HERE. Genuine model
        // pricing below may acquire its own device/Fourier execution guard.
        let action = policy_actions(checkpoint, config, &batch, &selected_hash)?;
        let candidate_scalar = scalar_hedge_pnl(
            &batch,
            &action.holdings,
            training.cost,
            training.premium_for_comparison,
        )?;
        let (baseline, cache_access) = model_hedge_holdings_cached(
            &markets,
            training.liability,
            &plan.effective_model,
            config.baseline_cache.as_ref(),
        )?;
        if config.baseline_cache.is_some() {
            cache_summary.hits += cache_access.entry_hits;
            cache_summary.misses += cache_access.entry_misses;
            cache_summary.entries_written += cache_access.entries_written;
        }
        cache_summary.bytes_read = cache_summary
            .bytes_read
            .checked_add(cache_access.bytes_read)
            .ok_or_else(overflow)?;
        cache_summary.bytes_written = cache_summary
            .bytes_written
            .checked_add(cache_access.bytes_written)
            .ok_or_else(overflow)?;
        cache_summary.actual_function_evaluations = cache_summary
            .actual_function_evaluations
            .checked_add(cache_access.actual_function_evaluations)
            .ok_or_else(overflow)?;
        if let (Some(key), Some(checksum)) = (&cache_access.key, &cache_access.payload_sha256) {
            cache_manifest.update(key.as_bytes());
            cache_manifest.update(checksum.as_bytes());
        }

        let expected_sources: Vec<_> = markets.iter().map(|m| m.content_hash.clone()).collect();
        if baseline.paths != count
            || baseline.steps != batch.steps
            || baseline.hedge_count != batch.hedge_count
            || baseline.market_source_hashes != expected_sources
        {
            return Err(execution(
                "model hedge did not preserve shared path identities/shape",
            ));
        }
        if let Some(previous) = &model_execution {
            if hash_json(previous)? != hash_json(&baseline.execution_policy)? {
                return Err(execution("model execution policy changed within benchmark"));
            }
        } else {
            model_execution = Some(baseline.execution_policy.clone());
        }
        let model_scalar = scalar_hedge_pnl(
            &batch,
            &baseline.holdings,
            training.cost,
            training.premium_for_comparison,
        )?;
        let unhedged = no_hedge_pnl(&batch, training.premium_for_comparison)?;
        candidate_metrics.add(&candidate_scalar, &action.holdings)?;
        model_metrics.add(&model_scalar, &baseline.holdings)?;
        for price in &baseline.initial_model_prices {
            model_prices.add(*price);
        }
        for local in 0..count {
            stream_hash.update(((first + local) as u64).to_le_bytes());
            let mark_stride = (batch.steps + 1) * batch.hedge_count;
            let feature_stride = batch.steps * batch.feature_count;
            hash_f64s(
                &mut stream_hash,
                &batch.marks[local * mark_stride..(local + 1) * mark_stride],
            );
            hash_f64s(
                &mut stream_hash,
                &batch.features[local * feature_stride..(local + 1) * feature_stride],
            );
            stream_hash.update(batch.payoffs[local].to_le_bytes());
            gap.add(action.pnl[local] - candidate_scalar.pnl[local])?;
            let per = batch.steps * batch.hedge_count;
            let c_hold = &action.holdings[local * per..(local + 1) * per];
            let m_hold = &baseline.holdings[local * per..(local + 1) * per];
            if first + local < config.audit_paths {
                let ca = reconcile_path(
                    &batch,
                    local,
                    c_hold,
                    family,
                    training.market.dt,
                    training.premium_for_comparison,
                    training.cost,
                )?;
                let ma = reconcile_path(
                    &batch,
                    local,
                    m_hold,
                    family,
                    training.market.dt,
                    training.premium_for_comparison,
                    training.cost,
                )?;
                if (ca.unquantized_float_pnl - candidate_scalar.pnl[local]).abs()
                    > 1e-10 * (1.0 + candidate_scalar.pnl[local].abs())
                    || (ma.unquantized_float_pnl - model_scalar.pnl[local]).abs()
                        > 1e-10 * (1.0 + model_scalar.pnl[local].abs())
                {
                    return Err(execution(
                        "audit and independent scalar path accounting disagree",
                    ));
                }
                audits.push(BenchmarkPathAudit {
                    path_id: (first + local) as u64,
                    candidate_device_pnl: action.pnl[local],
                    candidate_f64_identity_pnl: candidate_scalar.pnl[local],
                    model_f64_pnl: model_scalar.pnl[local],
                    candidate: ca,
                    model: ma,
                });
            }
            if let Some(output) = terminal.as_mut() {
                output.write_row([
                    candidate_scalar.pnl[local],
                    model_scalar.pnl[local],
                    unhedged.pnl[local],
                    action.pnl[local],
                    candidate_scalar.costs[local],
                    model_scalar.costs[local],
                    candidate_scalar.turnover[local],
                    model_scalar.turnover[local],
                ])?;
            }
            if let Some(output) = samples.as_mut() {
                let stride = (batch.steps + 1) * batch.hedge_count;
                let marks = &batch.marks[local * stride..(local + 1) * stride];
                let mut hash = Sha256::new();
                hash.update(b"volt.benchmark.path_marks.v1\0");
                for mark in marks {
                    hash.update(mark.to_le_bytes());
                }
                output.write_row(&SampleRow {
                    schema: "volt.compute.hedge_benchmark_sample.v1",
                    path_id: (first + local) as u64,
                    seed: config.seed,
                    batch_source_hash: &batch.source_hash,
                    path_marks_hash: format!("{:x}", hash.finalize()),
                    candidate_device_pnl: action.pnl[local],
                    candidate_f64_pnl: candidate_scalar.pnl[local],
                    model_f64_pnl: model_scalar.pnl[local],
                    no_hedge_pnl: unhedged.pnl[local],
                    candidate_cost: candidate_scalar.costs[local],
                    model_cost: model_scalar.costs[local],
                    candidate_turnover: candidate_scalar.turnover[local],
                    model_turnover: model_scalar.turnover[local],
                    candidate_holdings: c_hold,
                    model_holdings: m_hold,
                })?;
            }
        }
        candidate.extend(candidate_scalar.pnl);
        device.extend(action.pnl);
        model.extend(model_scalar.pnl);
        no_hedge.extend(unhedged.pnl);
        validations.push(baseline.validation);
        completed_batches += 1;
    }
    if candidate.len() != config.paths
        || model.len() != config.paths
        || device.len() != config.paths
        || no_hedge.len() != config.paths
        || audits.len() != config.audit_paths
    {
        return Err(execution("benchmark final path/audit count mismatch"));
    }
    let objective = training.objective;
    let candidate_risk =
        risk_objective::weighted_risk(&candidate, None, objective).map_err(risk_error)?;
    let model_risk = risk_objective::weighted_risk(&model, None, objective).map_err(risk_error)?;
    let no_hedge_risk =
        risk_objective::weighted_risk(&no_hedge, None, objective).map_err(risk_error)?;
    let candidate_device =
        risk_objective::weighted_risk(&device, None, objective).map_err(risk_error)?;
    let n = config.bootstrap_paths;
    let bootstrap = BootstrapConfig {
        replicates: config.bootstrap_replicates,
        confidence: 0.95,
        seed: config.seed ^ 0xb00757a9,
    };
    let paired_model =
        risk_objective::paired_path_comparison(&candidate[..n], &model[..n], objective, bootstrap)
            .map_err(risk_error)?;
    let paired_nohedge = risk_objective::paired_path_comparison(
        &candidate[..n],
        &no_hedge[..n],
        objective,
        bootstrap,
    )
    .map_err(risk_error)?;
    // Complete every fallible numeric check before atomically replacing an
    // existing output artifact.
    cache_summary.manifest_sha256 = format!("{:x}", cache_manifest.finalize());
    let candidate_metrics_report = candidate_metrics.report(config.paths)?;
    let model_metrics_report = model_metrics.report(config.paths)?;
    let gap_report = gap.report(config.paths)?;
    let mean_model_price = finite(
        model_prices.value() / config.paths as f64,
        "mean model premium",
    )?;
    let model_execution =
        model_execution.ok_or_else(|| execution("missing model execution metadata"))?;
    let exact = audits.iter().all(|a| {
        a.candidate.exact_reconciliation
            && a.model.exact_reconciliation
            && a.candidate.float_identity_within_bound
            && a.model.float_identity_within_bound
    });
    let bounded = audits
        .iter()
        .all(|a| a.candidate.quantization_within_bound && a.model.quantization_within_bound);
    let path_stream_hash = format!("{:x}", stream_hash.finalize());
    let market_stream_hash = market_fingerprint.finish()?;
    let terminal_pnl_artifact = terminal
        .map(|writer| writer.commit(&path_stream_hash, &market_stream_hash))
        .transpose()?;
    let sample_artifact = samples.map(AtomicSamples::commit).transpose()?;
    Ok(BenchmarkReport {schema:"volt.compute.hedge_benchmark.v1".into(),config:config.clone(),training_config:training.clone(),plan,
        selected_update:checkpoint.selected_update,selected_validation_risk:checkpoint.selected_validation_risk,checkpoint_sha256,
        checkpoint_hash_convention:"SHA256 of streamed serde serialization of checkpoint OBJECT, not original file bytes".into(),
        selected_policy_sha256:selected_hash,inference_pure:true,paths:config.paths,first_path_id:0,last_path_id:(config.paths-1) as u64,
        derived_system_seeds:(0..training.systems).map(|s|config.seed.wrapping_add((s as u64).wrapping_mul(SYSTEM_SEED_STRIDE))).collect(),
        path_stream_hash,market_stream_hash,identical_paths_for_all_strategies:true,
        path_hash_convention:"partition-independent definition/seed/count plus absolute path IDs and actual f64 marks/features/payoffs, in path-major order".into(),
        candidate:candidate_risk,model:model_risk,no_hedge:no_hedge_risk,candidate_device,
        candidate_device_accounting_gap:gap_report,candidate_metrics:candidate_metrics_report,model_metrics:model_metrics_report,
        no_hedge_metrics:StrategyMetrics {mean_cost:0.0,mean_turnover:0.0,mean_terminal_turnover:0.0,max_absolute_holding:0.0},
        paired_candidate_vs_model:paired_model,paired_candidate_vs_no_hedge:paired_nohedge,bootstrap_subset_paths:n,
        bootstrap_subset_selection:"first N independent held-out paths, selected before outcomes; same paired indices for both strategies; conditional on frozen policy and simulator".into(),
        model_execution_policy:model_execution,model_validation:validations,
        mean_initial_model_liability_price:mean_model_price,common_premium:training.premium_for_comparison,
        audits,audits_exact_reconciled:exact,audits_quantization_within_bounds:bounded,sample_artifact,terminal_pnl_artifact,baseline_cache:cache_summary,completed_batches,seconds:started.elapsed().as_secs_f64(),
        interpretation:"Lower risk is preferable. All primary strategies share independent f64 accounting and the same premium/marks/payoffs/path IDs. Device-minus-f64 differences include input downcasting and arithmetic. No improvement or paper replication is inferred merely from completed training, and poor candidates are reported unchanged. Selected weights/BatchNorm/Adam remain frozen; bootstrap does not measure training-seed or model uncertainty.".into()})
}

struct PolicyActions {
    pnl: Vec<f64>,
    holdings: Vec<f64>,
}
fn policy_actions(
    checkpoint: &TrainingCheckpoint,
    config: &BenchmarkConfig,
    batch: &HedgePathBatch,
    selected_hash: &str,
) -> Result<PolicyActions, ComputeError> {
    let _guard = execution_guard()?;
    let session = Session::new(
        config.target,
        checkpoint.config.precision,
        checkpoint.config.policy.max_batch_rows,
    )?;
    let _cleanup = crate::runtime::DeviceCleanupGuard(&session);
    let mut policy = HedgePolicy::from_snapshot(&session, checkpoint.selected_policy.clone())?;
    let prepared = PreparedHedgePaths::upload(&session, batch)?;
    if prepared.source_hash() != batch.source_hash {
        return Err(execution("policy upload changed path identity"));
    }
    let rollout = hedge_application::policy_rollout(
        &session,
        &prepared,
        &mut policy,
        checkpoint.config.cost,
        checkpoint.config.premium_for_comparison,
        false,
        true,
    )?;
    let pnl = rollout
        .pnl
        .without_autodiff()
        .try_to_vec_as::<f64>()
        .map_err(burn_error)?;
    if pnl.len() != batch.paths
        || pnl.iter().any(|v| !v.is_finite())
        || rollout.holdings.len() != batch.steps
    {
        return Err(execution("malformed policy benchmark output"));
    }
    let mut holdings = vec![0.0; batch.paths * batch.steps * batch.hedge_count];
    for (time, tensor) in rollout.holdings.into_iter().enumerate() {
        let values = tensor.try_to_vec_as::<f64>().map_err(burn_error)?;
        if values.len() != batch.paths * batch.hedge_count || values.iter().any(|v| !v.is_finite())
        {
            return Err(execution("malformed retained policy holdings"));
        }
        for path in 0..batch.paths {
            holdings[(path * batch.steps + time) * batch.hedge_count
                ..(path * batch.steps + time + 1) * batch.hedge_count]
                .copy_from_slice(&values[path * batch.hedge_count..(path + 1) * batch.hedge_count]);
        }
    }
    if hash_json(&policy.snapshot()?)? != selected_hash {
        return Err(execution(
            "held-out inference mutated policy/BatchNorm/Adam state",
        ));
    }
    if session
        .device()
        .memory_pool_usage()
        .is_some_and(|u| u.bytes_reserved > checkpoint.config.budget.compute_bytes)
    {
        return Err(ComputeError::BudgetExceeded(
            "benchmark candidate allocator exceeds compute budget".into(),
        ));
    }
    session.synchronize()?;
    Ok(PolicyActions { pnl, holdings })
}

#[derive(Default)]
struct MetricTotals {
    cost: CompensatedSum,
    turnover: CompensatedSum,
    terminal: CompensatedSum,
    max: f64,
}
impl MetricTotals {
    fn add(&mut self, rollout: &ScalarHedgeRollout, holdings: &[f64]) -> Result<(), ComputeError> {
        for x in &rollout.costs {
            self.cost.add(*x);
        }
        for x in &rollout.turnover {
            self.turnover.add(*x);
        }
        for x in &rollout.terminal_turnover {
            self.terminal.add(*x);
        }
        for h in holdings {
            self.max = self.max.max(h.abs());
        }
        Ok(())
    }
    fn report(self, paths: usize) -> Result<StrategyMetrics, ComputeError> {
        Ok(StrategyMetrics {
            mean_cost: finite(self.cost.value() / paths as f64, "mean fees")?,
            mean_turnover: finite(self.turnover.value() / paths as f64, "mean turnover")?,
            mean_terminal_turnover: finite(
                self.terminal.value() / paths as f64,
                "mean terminal turnover",
            )?,
            max_absolute_holding: finite(self.max, "maximum holding")?,
        })
    }
}
#[derive(Default)]
struct GapTotals {
    sum: CompensatedSum,
    square: CompensatedSum,
    max: f64,
}
impl GapTotals {
    fn add(&mut self, gap: f64) -> Result<(), ComputeError> {
        finite(gap, "device accounting gap")?;
        self.sum.add(gap);
        self.square.add(finite(gap * gap, "squared device gap")?);
        self.max = self.max.max(gap.abs());
        Ok(())
    }
    fn report(self, paths: usize) -> Result<ArithmeticGap, ComputeError> {
        Ok(ArithmeticGap {convention:"device P&L minus independent f64 identity using the same retained actions and original host marks; includes input dtype conversion".into(),paths,
        mean_signed_difference:finite(self.sum.value()/paths as f64,"mean device gap")?,root_mean_square_difference:finite((self.square.value()/paths as f64).sqrt(),"RMS device gap")?,max_absolute_difference:self.max})
    }
}

#[derive(Serialize)]
struct SampleRow<'a> {
    schema: &'static str,
    path_id: u64,
    seed: u64,
    batch_source_hash: &'a str,
    path_marks_hash: String,
    candidate_device_pnl: f64,
    candidate_f64_pnl: f64,
    model_f64_pnl: f64,
    no_hedge_pnl: f64,
    candidate_cost: f64,
    model_cost: f64,
    candidate_turnover: f64,
    model_turnover: f64,
    candidate_holdings: &'a [f64],
    model_holdings: &'a [f64],
}
struct AtomicSamples {
    target: PathBuf,
    temporary: PathBuf,
    writer: Option<BufWriter<File>>,
    hash: Sha256,
    rows: usize,
    bytes: u64,
    committed: bool,
}
impl AtomicSamples {
    fn new(path: &Path) -> Result<Self, ComputeError> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(io_error)?;
        }
        let name = path
            .file_name()
            .ok_or_else(|| invalid("samples path missing filename"))?
            .to_string_lossy();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(burn_error)?
            .as_nanos();
        let temporary = path.with_file_name(format!(
            ".{name}.volt-benchmark-{}-{stamp}.tmp",
            std::process::id()
        ));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(io_error)?;
        Ok(Self {
            target: path.into(),
            temporary,
            writer: Some(BufWriter::new(file)),
            hash: Sha256::new(),
            rows: 0,
            bytes: 0,
            committed: false,
        })
    }
    fn write_row(&mut self, row: &SampleRow<'_>) -> Result<(), ComputeError> {
        let mut bytes = serde_json::to_vec(row).map_err(burn_error)?;
        bytes.push(b'\n');
        self.writer
            .as_mut()
            .expect("uncommitted writer")
            .write_all(&bytes)
            .map_err(io_error)?;
        self.hash.update(&bytes);
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(overflow)?;
        self.rows += 1;
        Ok(())
    }
    fn commit(mut self) -> Result<SampleArtifact, ComputeError> {
        let mut writer = self.writer.take().expect("uncommitted writer");
        writer.flush().map_err(io_error)?;
        writer.get_ref().sync_all().map_err(io_error)?;
        drop(writer);
        std::fs::rename(&self.temporary, &self.target).map_err(io_error)?;
        self.committed = true;
        let hash = std::mem::take(&mut self.hash);
        Ok(SampleArtifact {
            path: self.target.clone(),
            rows: self.rows,
            bytes: self.bytes,
            sha256: format!("{:x}", hash.finalize()),
        })
    }
}
impl Drop for AtomicSamples {
    fn drop(&mut self) {
        if !self.committed {
            self.writer.take();
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}
struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn hash_json<T: Serialize>(value: &T) -> Result<String, ComputeError> {
    let mut writer = HashWriter(Sha256::new());
    serde_json::to_writer(&mut writer, value).map_err(burn_error)?;
    Ok(format!("{:x}", writer.0.finalize()))
}
fn hash_f64s(hash: &mut Sha256, values: &[f64]) {
    let mut buffer = [0_u8; 8192];
    for chunk in values.chunks(1024) {
        for (value, slot) in chunk.iter().zip(buffer.as_chunks_mut::<8>().0.iter_mut()) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        hash.update(&buffer[..chunk.len() * 8]);
    }
}
fn usable(bytes: u64, headroom: u8) -> u64 {
    ((bytes as u128) * u128::from(100 - headroom) / 100) as u64
}
fn overflow() -> ComputeError {
    invalid("benchmark memory/count estimate overflow")
}
fn invalid(message: &str) -> ComputeError {
    ComputeError::InvalidInput(message.into())
}
fn execution(message: &str) -> ComputeError {
    ComputeError::Execution(message.into())
}
fn finite(value: f64, name: &str) -> Result<f64, ComputeError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(execution(&format!("nonfinite {name}")))
    }
}
fn risk_error(error: risk_objective::RiskError) -> ComputeError {
    execution(&error.to_string())
}
fn burn_error(error: impl std::fmt::Display) -> ComputeError {
    execution(&error.to_string())
}
fn io_error(error: std::io::Error) -> ComputeError {
    execution(&format!("benchmark sample artifact: {error}"))
}

/// Binary format: 16-byte magic, u32 little-endian JSON-header length, bounded
/// JSON header, rows of eight IEEE754 little-endian f64 values (64 bytes/row),
/// 8-byte footer magic, 32-byte financial path digest, 32-byte market-only path
/// digest, then 32-byte SHA256 of
/// all preceding bytes. Artifact report SHA256 covers the complete file. Path
/// identity is implicit and ordered; no path is sampled or omitted.
pub const TERMINAL_PNL_MAGIC: [u8; 16] = *b"VOLT.PNL.F64.V2\0";
const TERMINAL_FOOTER_MAGIC: [u8; 8] = *b"VOLTPN2!";
pub const TERMINAL_PNL_ROW_BYTES: u64 = 64;
pub const TERMINAL_PNL_COLUMNS: [&str; 8] = [
    "candidate_pnl",
    "model_pnl",
    "no_hedge_pnl",
    "device_pnl",
    "candidate_cost",
    "model_cost",
    "candidate_turnover",
    "model_turnover",
];
const MAX_TERMINAL_HEADER: usize = 16384;
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalPnlHeader {
    pub schema: String,
    pub rows: usize,
    pub first_path_id: u64,
    pub columns: Vec<String>,
    pub cohort_spec_sha256: String,
    pub market_cohort_spec_sha256: String,
    pub selected_policy_sha256: String,
    pub checkpoint_sha256: String,
    pub model_numeric_policy_sha256: String,
    pub common_premium: f64,
    pub cost: crate::hedge_application::HedgeCostConfig,
    pub objective: RiskObjective,
}
#[derive(Debug, Clone, Serialize)]
pub struct TerminalPnlVerification {
    pub header: TerminalPnlHeader,
    pub data_offset: u64,
    pub path_stream_sha256: String,
    pub market_stream_sha256: String,
    pub artifact: SampleArtifact,
}
struct AtomicTerminalPnl {
    target: PathBuf,
    temporary: PathBuf,
    writer: Option<BufWriter<File>>,
    hash: Sha256,
    expected_rows: usize,
    rows: usize,
    bytes: u64,
    max_bytes: u64,
    committed: bool,
}
impl AtomicTerminalPnl {
    fn new(path: &Path, header: &TerminalPnlHeader) -> Result<Self, ComputeError> {
        validate_terminal_header(header)?;
        let json = serde_json::to_vec(header).map_err(burn_error)?;
        if json.len() > MAX_TERMINAL_HEADER {
            return Err(invalid("terminal-PnL header exceeds size cap"));
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(io_error)?;
        }
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(burn_error)?
            .as_nanos();
        let temporary = path.with_extension(format!("pnl-partial-{}-{stamp}", std::process::id()));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(io_error)?;
        let max_bytes = terminal_file_bytes(header.rows, json.len())?;
        let mut output = Self {
            target: path.into(),
            temporary,
            writer: Some(BufWriter::new(file)),
            hash: Sha256::new(),
            expected_rows: header.rows,
            rows: 0,
            bytes: 0,
            max_bytes,
            committed: false,
        };
        output.write_bytes(&TERMINAL_PNL_MAGIC)?;
        output.write_bytes(&(json.len() as u32).to_le_bytes())?;
        output.write_bytes(&json)?;
        Ok(output)
    }
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), ComputeError> {
        let count = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(overflow)?;
        if count > self.max_bytes {
            return Err(invalid("terminal-PnL writer exceeded declared file shape"));
        }
        self.writer
            .as_mut()
            .expect("uncommitted writer")
            .write_all(bytes)
            .map_err(io_error)?;
        self.hash.update(bytes);
        self.bytes = count;
        Ok(())
    }
    fn write_row(&mut self, values: [f64; 8]) -> Result<(), ComputeError> {
        if self.rows >= self.expected_rows
            || values.iter().any(|x| !x.is_finite())
            || values[4..].iter().any(|x| *x < 0.0)
        {
            return Err(invalid("terminal-PnL row count/value mismatch"));
        }
        let mut bytes = [0u8; 64];
        for (value, slot) in values.iter().zip(bytes.as_chunks_mut::<8>().0.iter_mut()) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        self.write_bytes(&bytes)?;
        self.rows += 1;
        Ok(())
    }
    fn commit(
        mut self,
        path_stream_hash: &str,
        market_stream_hash: &str,
    ) -> Result<SampleArtifact, ComputeError> {
        if self.rows != self.expected_rows {
            return Err(invalid("terminal-PnL output is incomplete"));
        }
        let digest = decode_sha256(path_stream_hash)?;
        self.write_bytes(&TERMINAL_FOOTER_MAGIC)?;
        self.write_bytes(&digest)?;
        self.write_bytes(&decode_sha256(market_stream_hash)?)?;
        let checksum = self.hash.clone().finalize();
        self.write_bytes(&checksum)?;
        if self.bytes != self.max_bytes {
            return Err(invalid("terminal-PnL final byte count mismatch"));
        }
        let mut writer = self.writer.take().expect("uncommitted writer");
        writer.flush().map_err(io_error)?;
        writer.get_ref().sync_all().map_err(io_error)?;
        drop(writer);
        std::fs::rename(&self.temporary, &self.target).map_err(io_error)?;
        self.committed = true;
        Ok(SampleArtifact {
            path: self.target.clone(),
            rows: self.rows,
            bytes: self.bytes,
            sha256: format!("{:x}", std::mem::take(&mut self.hash).finalize()),
        })
    }
}
impl Drop for AtomicTerminalPnl {
    fn drop(&mut self) {
        if !self.committed {
            self.writer.take();
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}
fn terminal_file_bytes(rows: usize, header: usize) -> Result<u64, ComputeError> {
    (rows as u64)
        .checked_mul(TERMINAL_PNL_ROW_BYTES)
        .and_then(|n| n.checked_add(header as u64))
        .and_then(|n| n.checked_add(124))
        .ok_or_else(overflow)
}
fn validate_terminal_header(header: &TerminalPnlHeader) -> Result<(), ComputeError> {
    if header.schema != "volt.compute.terminal_pnl.v2"
        || !(2..=1_000_000).contains(&header.rows)
        || header.first_path_id != 0
        || !header
            .columns
            .iter()
            .map(String::as_str)
            .eq(TERMINAL_PNL_COLUMNS)
        || !header.common_premium.is_finite()
        || !header.cost.proportional.is_finite()
        || header.cost.proportional < 0.0
        || !header.cost.terminal_proportional.is_finite()
        || header.cost.terminal_proportional < 0.0
    {
        return Err(invalid(
            "invalid terminal-PnL header/schema/financial terms",
        ));
    }
    for hash in [
        &header.cohort_spec_sha256,
        &header.market_cohort_spec_sha256,
        &header.selected_policy_sha256,
        &header.checkpoint_sha256,
        &header.model_numeric_policy_sha256,
    ] {
        decode_sha256(hash)?;
    }
    risk_objective::weighted_risk(&[0.0], None, header.objective).map_err(risk_error)?;
    Ok(())
}
fn decode_sha256(value: &str) -> Result<[u8; 32], ComputeError> {
    if value.len() != 64 || !value.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(invalid("expected a64-digit SHA256 digest"));
    }
    let mut bytes = [0u8; 32];
    for (i, pair) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let digit = |c: u8| {
            if c.is_ascii_digit() {
                c - b'0'
            } else {
                c.to_ascii_lowercase() - b'a' + 10
            }
        };
        bytes[i] = (digit(pair[0]) << 4) | digit(pair[1]);
    }
    Ok(bytes)
}
/// Streaming integrity/shape/numeric verification with bounded memory. The data
/// offset and public column/row layout allow independent offline risk analysis.
/// Hashes check integrity and identity metadata, not third-party authenticity.
pub fn verify_terminal_pnl_artifact(path: &Path) -> Result<TerminalPnlVerification, ComputeError> {
    let file = File::open(path).map_err(io_error)?;
    let length = file.metadata().map_err(io_error)?.len();
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 16];
    reader.read_exact(&mut magic).map_err(io_error)?;
    if magic != TERMINAL_PNL_MAGIC {
        return Err(invalid("terminal-PnL magic mismatch"));
    }
    let mut size = [0u8; 4];
    reader.read_exact(&mut size).map_err(io_error)?;
    let size_value = u32::from_le_bytes(size) as usize;
    if size_value == 0 || size_value > MAX_TERMINAL_HEADER {
        return Err(invalid("terminal-PnL header size invalid"));
    }
    let mut json = vec![0u8; size_value];
    reader.read_exact(&mut json).map_err(io_error)?;
    let header: TerminalPnlHeader = serde_json::from_slice(&json).map_err(burn_error)?;
    validate_terminal_header(&header)?;
    if length != terminal_file_bytes(header.rows, size_value)? {
        return Err(invalid("terminal-PnL length/count mismatch"));
    }
    let mut hash = Sha256::new();
    hash.update(magic);
    hash.update(size);
    hash.update(&json);
    let mut remaining = (header.rows as u64) * TERMINAL_PNL_ROW_BYTES;
    let mut buffer = [0u8; 8192];
    while remaining > 0 {
        let n = remaining.min(buffer.len() as u64) as usize;
        reader.read_exact(&mut buffer[..n]).map_err(io_error)?;
        for (row_index, row) in buffer[..n].as_chunks::<64>().0.iter().enumerate() {
            for (column, raw) in row.as_chunks::<8>().0.iter().enumerate() {
                let value = f64::from_le_bytes(*raw);
                if !value.is_finite() || (column >= 4 && value < 0.0) {
                    return Err(invalid(&format!(
                        "terminal-PnL invalid numeric row{row_index}"
                    )));
                }
            }
        }
        hash.update(&buffer[..n]);
        remaining -= n as u64;
    }
    let mut footer = [0u8; 8];
    reader.read_exact(&mut footer).map_err(io_error)?;
    if footer != TERMINAL_FOOTER_MAGIC {
        return Err(invalid("terminal-PnL footer mismatch"));
    }
    let mut path_digest = [0u8; 32];
    reader.read_exact(&mut path_digest).map_err(io_error)?;
    hash.update(footer);
    hash.update(path_digest);
    let mut market_digest = [0u8; 32];
    reader.read_exact(&mut market_digest).map_err(io_error)?;
    hash.update(market_digest);
    let mut checksum = [0u8; 32];
    reader.read_exact(&mut checksum).map_err(io_error)?;
    let expected_checksum: [u8; 32] = hash.clone().finalize().into();
    if expected_checksum != checksum {
        return Err(invalid("terminal-PnL checksum mismatch"));
    }
    hash.update(checksum);
    let mut extra = [0u8; 1];
    if reader.read(&mut extra).map_err(io_error)? != 0 {
        return Err(invalid("trailing terminal-PnL bytes"));
    }
    let path_stream_sha256 = path_digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok(TerminalPnlVerification {
        data_offset: 20 + size_value as u64,
        artifact: SampleArtifact {
            path: path.into(),
            rows: header.rows,
            bytes: length,
            sha256: format!("{:x}", hash.finalize()),
        },
        header,
        path_stream_sha256,
        market_stream_sha256: market_digest.iter().map(|b| format!("{b:02x}")).collect(),
    })
}

/// Reusable market-only cohort fingerprint. Liability, premium, cost, objective,
/// policy architecture, policy precision and training seeds are excluded.
/// Every path contributes its ID plus actual marks/features in fixed LE f64
/// order, permitting paired liability/no-liability or band/policy comparisons.
pub struct MarketCohortHasher {
    spec: String,
    hash: Sha256,
    next: u64,
    paths: usize,
    steps: usize,
    hedges: usize,
}
impl MarketCohortHasher {
    pub fn new(
        training: &HedgeTrainingConfig,
        seed: u64,
        paths: usize,
        simulation_substeps: usize,
    ) -> Result<Self, ComputeError> {
        if paths == 0 || simulation_substeps == 0 || training.systems == 0 {
            return Err(invalid("invalid market fingerprint shape"));
        }
        let spec = hash_json(&(
            "volt.market_cohort.spec.v1",
            training.market.model,
            training.market.steps,
            training.market.dt,
            training.market.initial_spot,
            training.market.rate,
            training.market.dividend_yield,
            training.market.integrated_variance_rule,
            training.systems,
            seed,
            paths,
            simulation_substeps,
            crate::market_paths::RNG_CONTRACT,
        ))?;
        let mut hash = Sha256::new();
        hash.update(b"volt.market_cohort.rows.v1\0");
        hash.update(spec.as_bytes());
        let per = if matches!(training.market.model, MarketModel::Heston { .. }) {
            2
        } else {
            1
        };
        let hedges = training.systems.checked_mul(per).ok_or_else(overflow)?;
        Ok(Self {
            spec,
            hash,
            next: 0,
            paths,
            steps: training.market.steps,
            hedges,
        })
    }
    pub fn spec_sha256(&self) -> &str {
        &self.spec
    }
    pub fn update_batch(
        &mut self,
        first_path_id: u64,
        batch: &HedgePathBatch,
    ) -> Result<(), ComputeError> {
        batch.validate()?;
        let next = self
            .next
            .checked_add(batch.paths as u64)
            .ok_or_else(overflow)?;
        if first_path_id != self.next
            || next > self.paths as u64
            || batch.steps != self.steps
            || batch.hedge_count != self.hedges
            || batch.feature_count != self.hedges
        {
            return Err(invalid("market fingerprint rows/grid are not aligned"));
        }
        let marks = (batch.steps + 1) * batch.hedge_count;
        let features = batch.steps * batch.feature_count;
        for path in 0..batch.paths {
            self.hash
                .update((first_path_id + path as u64).to_le_bytes());
            hash_f64s(
                &mut self.hash,
                &batch.marks[path * marks..(path + 1) * marks],
            );
            hash_f64s(
                &mut self.hash,
                &batch.features[path * features..(path + 1) * features],
            );
        }
        self.next = next;
        Ok(())
    }
    pub fn finish(self) -> Result<String, ComputeError> {
        if self.next != self.paths as u64 {
            return Err(invalid("incomplete market fingerprint cohort"));
        }
        Ok(format!("{:x}", self.hash.finalize()))
    }
}
