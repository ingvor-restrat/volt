//! Complete bounded training/checkpoint/inference runner for exogenous markets.
//! Paper-style architecture is distinct from quantitative paper replication:
//! reports state held-out counts and matching deviations, never infer success
//! merely from a decreasing training loss.
use crate::{
    ComputeError, ExecutionTarget, Precision, Session,
    budget::MemoryBudget,
    chunked::execution_guard,
    hedge_application::{
        self, HedgeCostConfig, HedgeLiability, HedgeMemoryPlan, HedgePathBatch, PreparedHedgePaths,
    },
    hedge_policy::{HedgePolicy, HedgePolicyConfig, PolicySnapshot},
    market_paths::{self, MarketModel, MarketPathConfig},
    risk_objective::{self, BootstrapConfig, PairedPathComparison, RiskEvaluation, RiskObjective},
};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs::OpenOptions, io::Write, path::Path, time::Instant};

/// Validation-only stopping, evaluated at declared validation boundaries.
/// No test outcomes affect this decision. A selected zero policy counts as an
/// improvement at update zero; patience persists across checkpoint resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EarlyStoppingConfig {
    pub minimum_updates: u64,
    pub patience_updates: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStopReason {
    MaximumUpdates,
    ValidationPatience,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HedgeTrainingConfig {
    pub target: ExecutionTarget,
    pub precision: Precision,
    pub market: MarketPathConfig,
    pub systems: usize,
    pub liability: HedgeLiability,
    pub policy: HedgePolicyConfig,
    pub objective: RiskObjective,
    pub cost: HedgeCostConfig,
    pub policy_seed: u64,
    pub batch_paths: usize,
    pub max_updates: u64,
    #[serde(default)]
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub learning_rate: f64,
    pub validation_interval: u64,
    pub validation_paths: usize,
    pub validation_seed: u64,
    /// Zero preserves legacy EMA-only checkpoints. New profiles refresh frozen
    /// BN statistics from an independent training-only stream before validation.
    #[serde(default)]
    pub calibration_paths: usize,
    #[serde(default = "default_calibration_seed")]
    pub calibration_seed: u64,
    pub test_paths: usize,
    pub test_seed: u64,
    pub evaluation_batch_paths: usize,
    pub premium_for_comparison: f64,
    pub bootstrap_paths: usize,
    pub bootstrap_replicates: usize,
    pub budget: MemoryBudget,
}
impl HedgeTrainingConfig {
    pub fn paper_heston(target: ExecutionTarget) -> Self {
        Self {
            target,
            precision: Precision::F32,
            market: MarketPathConfig::paper_heston(256, 11),
            systems: 1,
            liability: HedgeLiability::Call { strike: 100.0 },
            policy: HedgePolicyConfig::paper_heston(4096),
            objective: RiskObjective::Cvar { alpha: 0.5 },
            cost: HedgeCostConfig::default(),
            policy_seed: 23,
            batch_paths: 256,
            max_updates: 1000,
            early_stopping: None,
            learning_rate: 0.005,
            validation_interval: 100,
            validation_paths: 4096,
            validation_seed: 1009,
            calibration_paths: 4096,
            calibration_seed: 5011,
            test_paths: 16384,
            test_seed: 2003,
            evaluation_batch_paths: 1024,
            premium_for_comparison: 1.691833925110386,
            bootstrap_paths: 4096,
            bootstrap_replicates: 200,
            budget: MemoryBudget {
                compute_bytes: 1024 * 1024 * 1024,
                host_working_bytes: 2 * 1024 * 1024 * 1024,
                max_batch_rows: 4096,
                ..MemoryBudget::default()
            },
        }
    }
    pub fn black_scholes(target: ExecutionTarget) -> Self {
        let mut c = Self::paper_heston(target);
        c.market.model = MarketModel::BlackScholes { volatility: 0.2 };
        c.policy = HedgePolicyConfig::paper_black_scholes(4096);
        c.objective = RiskObjective::SquaredLoss;
        let row = crate::OptionRow {
            row_id: "bs-control".into(),
            contract_id: "synthetic-call".into(),
            side: crate::OptionSide::Call,
            spot: 100.0,
            strike: 100.0,
            maturity_years: 30.0 / 365.0,
            volatility: 0.2,
            rate: 0.0,
            dividend_yield: 0.0,
            quantity: 1,
            multiplier: 1,
        };
        c.premium_for_comparison = crate::reference_price(&row).expect("fixed valid BS fixture");
        c
    }
    pub fn validate(&self) -> Result<(HedgeMemoryPlan, HedgeMemoryPlan), ComputeError> {
        if self.systems == 0
            || self.systems > 5
            || self.batch_paths < 2
            || self.evaluation_batch_paths == 0
            || self.max_updates == 0
            || self.validation_interval == 0
            || self.validation_paths < 2
            || self.test_paths < 2
            || self.bootstrap_paths < 2
            || self.bootstrap_replicates < 2
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.premium_for_comparison.is_finite()
            || !self.cost.proportional.is_finite()
            || self.cost.proportional < 0.0
            || !self.cost.terminal_proportional.is_finite()
            || self.cost.terminal_proportional < 0.0
            || self.market.rate != 0.0
            || self.market.dividend_yield != 0.0
            || self.market.seed == self.validation_seed
            || self.market.seed == self.test_seed
            || self.validation_seed == self.test_seed
            || self.market.first_path_id != 0
        {
            return Err(invalid(
                "invalid training sizes, rates, independent seeds, or learning rate",
            ));
        }
        if let Some(stopping) = &self.early_stopping
            && (stopping.minimum_updates == 0 || stopping.patience_updates == 0)
        {
            return Err(invalid("early stopping update counts must be positive"));
        }
        self.liability.payoff(self.market.initial_spot)?;
        risk_objective::weighted_risk(&[0.0], None, self.objective).map_err(execution)?;
        crate::hedge_policy::validate_policy_config(&self.policy, self.precision)?;
        if self.precision == Precision::F32 {
            let constants = [
                self.learning_rate,
                self.cost.proportional,
                self.cost.terminal_proportional,
                self.premium_for_comparison,
            ];
            if constants.iter().any(|v| !(*v as f32).is_finite())
                || (self.learning_rate as f32) <= 0.0
            {
                return Err(invalid("training constants not representable in f32"));
            }
            if let RiskObjective::Entropic { risk_aversion: l } = self.objective
                && (!l.is_finite() || !(l as f32).is_normal() || !((1.0 / l) as f32).is_finite())
            {
                return Err(invalid("entropic scale not representable in f32"));
            }
        }
        let mut seeds = std::collections::BTreeSet::new();
        let mut stream_seeds = vec![self.market.seed, self.validation_seed, self.test_seed];
        if self.calibration_paths > 0 && self.policy.batch_norm.enabled {
            if self.calibration_paths < 2 {
                return Err(invalid("BatchNorm calibration needs at least two paths"));
            }
            stream_seeds.push(self.calibration_seed);
        }
        for base in stream_seeds {
            for system in 0..self.systems {
                if !seeds
                    .insert(base.wrapping_add((system as u64).wrapping_mul(0x9e3779b97f4a7c15)))
                {
                    return Err(invalid(
                        "derived training/validation/test system streams overlap",
                    ));
                }
            }
        }
        let per = if matches!(self.market.model, MarketModel::Heston { .. }) {
            2
        } else {
            1
        };
        if self.policy.time_steps != self.market.steps
            || self.policy.hedge_count != per * self.systems
            || self.policy.feature_count != per * self.systems
        {
            return Err(invalid("policy dimensions differ from market systems"));
        }
        let mut market = self.market.clone();
        market.paths = self.batch_paths;
        market.validate()?;
        market.paths = self.evaluation_batch_paths;
        market.validate()?;
        self.max_updates
            .checked_mul(self.batch_paths as u64)
            .ok_or_else(|| invalid("training path-ID range overflow"))?;
        let train = hedge_application::plan_hedging(
            &self.policy,
            self.batch_paths,
            self.precision,
            self.target,
            &self.budget,
            true,
        )?;
        if self.calibration_paths > 0 && self.policy.batch_norm.enabled {
            hedge_application::plan_hedging(
                &self.policy,
                self.calibration_paths,
                self.precision,
                self.target,
                &self.budget,
                false,
            )?;
        }
        let evaluation = hedge_application::plan_hedging(
            &self.policy,
            self.evaluation_batch_paths,
            self.precision,
            self.target,
            &self.budget,
            false,
        )?;
        let history_bytes = self
            .max_updates
            .div_ceil(self.validation_interval)
            .checked_mul(160)
            .and_then(|n| {
                (self.bootstrap_replicates as u64)
                    .checked_mul(32)
                    .and_then(|b| n.checked_add(b))
            })
            .ok_or_else(|| invalid("training history/bootstrap storage overflow"))?;
        let global = (self.test_paths as u64)
            .checked_mul(64)
            .and_then(|n| {
                (self.validation_paths as u64)
                    .checked_mul(32)
                    .and_then(|v| n.checked_add(v))
            })
            .and_then(|n| n.checked_add(history_bytes))
            .ok_or_else(|| invalid("held-out storage estimate overflow"))?;
        if global
            .checked_add(evaluation.estimated_host_working_bytes)
            .is_none_or(|n| n > evaluation.host_admission_limit_bytes)
        {
            return Err(ComputeError::BudgetExceeded(
                "held-out results and evaluation workspace exceed host allowance".into(),
            ));
        }
        Ok((train, evaluation))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EtaSnapshot {
    pub value: f64,
    pub first_moment: f64,
    pub second_moment: f64,
    pub step: u64,
}
struct EtaState {
    state: EtaSnapshot,
    leaf: Tensor<1>,
}
impl EtaState {
    fn from_state(session: &Session, state: EtaSnapshot) -> Result<Self, ComputeError> {
        if [state.value, state.first_moment, state.second_moment]
            .iter()
            .any(|v| !v.is_finite())
            || state.second_moment < 0.0
        {
            return Err(invalid("invalid CVaR eta checkpoint"));
        }
        let leaf = session.vector(vec![state.value]).autodiff().require_grad();
        Ok(Self { state, leaf })
    }
    fn candidate(
        &self,
        gradient: f64,
        config: &HedgeTrainingConfig,
    ) -> Result<EtaSnapshot, ComputeError> {
        let a = config.policy.adam;
        let step = self
            .state
            .step
            .checked_add(1)
            .ok_or_else(|| invalid("eta step overflow"))?;
        let first = a.beta1 * self.state.first_moment + (1.0 - a.beta1) * gradient;
        let second = a.beta2 * self.state.second_moment + (1.0 - a.beta2) * gradient * gradient;
        let value = self.state.value
            - config.learning_rate * (first / (1.0 - a.beta1.powf(step as f64)))
                / ((second / (1.0 - a.beta2.powf(step as f64))).sqrt() + a.epsilon);
        if [gradient, first, second, value]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err(ComputeError::Execution("nonfinite CVaR eta update".into()));
        }
        Ok(EtaSnapshot {
            value,
            first_moment: first,
            second_moment: second,
            step,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpoint {
    pub schema: String,
    pub config: HedgeTrainingConfig,
    pub completed_updates: u64,
    pub current_policy: PolicySnapshot,
    pub current_eta: Option<EtaSnapshot>,
    pub selected_policy: PolicySnapshot,
    pub selected_update: u64,
    pub selected_validation_risk: f64,
}
#[derive(Debug, Serialize)]
pub struct TrainingObservation {
    pub update: u64,
    pub training_loss: f64,
    pub gradient_norm: f64,
    pub update_seconds: f64,
    pub validation_risk: f64,
    pub selected: bool,
}
#[derive(Debug, Serialize)]
pub struct HedgeEvaluationReport {
    pub candidate: RiskEvaluation,
    pub no_hedge: RiskEvaluation,
    pub paired_subset: PairedPathComparison,
    pub bootstrap_subset_paths: usize,
    pub mean_cost: f64,
    pub mean_turnover: f64,
    pub max_absolute_holding: f64,
    pub paths: usize,
    pub path_stream_hash: String,
    pub seconds: f64,
    pub sample_candidate_pnl: Vec<f64>,
}
#[derive(Debug, Serialize)]
pub struct HedgeTrainingReport {
    pub schema: String,
    pub config: HedgeTrainingConfig,
    pub train_plan: HedgeMemoryPlan,
    pub evaluation_plan: HedgeMemoryPlan,
    pub completed_updates: u64,
    pub stop_reason: TrainingStopReason,
    pub selected_update: u64,
    pub selected_validation_risk: f64,
    pub training_seconds: f64,
    pub observations: Vec<TrainingObservation>,
    pub evaluation: HedgeEvaluationReport,
    pub checkpoint_sha256: String,
    pub paper_replication_complete: bool,
    pub deviations: Vec<String>,
}

pub fn generate_markets(
    config: &HedgeTrainingConfig,
    seed: u64,
    first: u64,
    count: usize,
) -> Result<Vec<market_paths::MarketPaths>, ComputeError> {
    let mut markets = Vec::with_capacity(config.systems);
    for system in 0..config.systems {
        let mut market = config.market.clone();
        market.seed = seed.wrapping_add((system as u64).wrapping_mul(0x9e3779b97f4a7c15));
        market.first_path_id = first;
        market.paths = count;
        markets.push(market_paths::generate_paths(&market)?);
    }
    Ok(markets)
}

/// Independent fine-grid evolution observed only on the original hedge dates.
/// Substeps=1 is exactly the original generator, including its content hashes.
/// Callers admit the resolution working set before invoking this helper.
pub fn generate_markets_with_substeps(
    config: &HedgeTrainingConfig,
    seed: u64,
    first: u64,
    count: usize,
    substeps: usize,
) -> Result<Vec<market_paths::MarketPaths>, ComputeError> {
    if config.systems == 0 || config.systems > 5 {
        return Err(invalid("resolved market generation requires1..5systems"));
    }
    if substeps == 1 {
        return generate_markets(config, seed, first, count);
    }
    let mut definitions = Vec::with_capacity(config.systems);
    for system in 0..config.systems {
        let mut market = config.market.clone();
        market.seed = seed.wrapping_add((system as u64).wrapping_mul(0x9e3779b97f4a7c15));
        market.first_path_id = first;
        market.paths = count;
        let definition =
            crate::market_path_resolution::MarketPathResolutionConfig { market, substeps };
        definition.validate()?;
        definitions.push(definition);
    }
    definitions
        .iter()
        .map(|definition| {
            Ok(crate::market_path_resolution::generate_resolved_paths(definition)?.market)
        })
        .collect()
}

pub fn generate_batch(
    config: &HedgeTrainingConfig,
    seed: u64,
    first: u64,
    count: usize,
) -> Result<HedgePathBatch, ComputeError> {
    HedgePathBatch::from_markets(
        &generate_markets(config, seed, first, count)?,
        config.liability,
    )
}

pub fn train(
    config: HedgeTrainingConfig,
    resume: Option<TrainingCheckpoint>,
    checkpoint_path: Option<&Path>,
) -> Result<HedgeTrainingReport, ComputeError> {
    let (train_plan, evaluation_plan) = config.validate()?;
    if let Some(checkpoint) = &resume {
        validate_resume(&config, checkpoint)?;
    }
    let _guard = execution_guard()?;
    let session = Session::new(
        config.target,
        config.precision,
        config.policy.max_batch_rows,
    )?;
    let _cleanup = crate::runtime::DeviceCleanupGuard(&session);
    let started = Instant::now();
    let (mut policy, mut eta, mut completed, mut selected, mut selected_update, mut best) =
        if let Some(checkpoint) = resume {
            validate_resume(&config, &checkpoint)?;
            (
                HedgePolicy::from_snapshot(&session, checkpoint.current_policy)?,
                checkpoint
                    .current_eta
                    .map(|v| EtaState::from_state(&session, v))
                    .transpose()?,
                checkpoint.completed_updates,
                checkpoint.selected_policy,
                checkpoint.selected_update,
                checkpoint.selected_validation_risk,
            )
        } else {
            let mut policy = HedgePolicy::new(&session, config.policy.clone(), config.policy_seed)?;
            let eta = if matches!(config.objective, RiskObjective::Cvar { .. }) {
                let batch = generate_batch(&config, config.market.seed, 0, config.batch_paths)?;
                let no_hedge: Vec<_> = batch.payoffs.iter().map(|p| -p).collect();
                let risk = risk_objective::weighted_risk(&no_hedge, None, config.objective)
                    .map_err(execution)?;
                Some(EtaState::from_state(
                    &session,
                    EtaSnapshot {
                        value: risk.cvar_boundary_loss.unwrap_or(0.0),
                        first_moment: 0.0,
                        second_moment: 0.0,
                        step: 0,
                    },
                )?)
            } else {
                None
            };
            calibrate_policy(&session, &config, &mut policy)?;
            let validation = evaluate_values(
                &session,
                &config,
                &mut policy,
                config.validation_seed,
                config.validation_paths,
                false,
            )?;
            let best = risk_objective::weighted_risk(&validation.pnl, None, config.objective)
                .map_err(execution)?
                .value;
            // Zero trading is an admissible policy and a useful validation
            // candidate, particularly for separately optimized empty liabilities.
            // Keep the random current policy so training can still learn.
            let zero_risk =
                risk_objective::weighted_risk(&validation.no_hedge, None, config.objective)
                    .map_err(execution)?
                    .value;
            let mut snapshot = policy.snapshot()?;
            let best = if zero_risk < best {
                for parameter in &mut snapshot.parameters {
                    parameter.values.fill(0.0);
                }
                zero_risk
            } else {
                best
            };
            (policy, eta, 0, snapshot, 0, best)
        };
    let mut observations = Vec::new();
    let mut stop_reason = TrainingStopReason::MaximumUpdates;
    while completed < config.max_updates {
        let step_started = Instant::now();
        let first = completed
            .checked_mul(config.batch_paths as u64)
            .ok_or_else(|| invalid("training stream overflow"))?;
        let batch = generate_batch(&config, config.market.seed, first, config.batch_paths)?;
        let prepared = PreparedHedgePaths::upload(&session, &batch)?;
        // Squared error needs the common fair premium; monetary risks are cash
        // invariant, and are trained with zero premium for capital estimation.
        let premium = if matches!(config.objective, RiskObjective::SquaredLoss) {
            config.premium_for_comparison
        } else {
            0.0
        };
        let rollout = hedge_application::policy_rollout(
            &session,
            &prepared,
            &mut policy,
            config.cost,
            premium,
            true,
            false,
        )?;
        let loss = hedge_application::tensor_risk(
            rollout.pnl,
            config.objective,
            eta.as_ref().map(|v| v.leaf.clone()),
        )?;
        let loss_value = read_one(&loss)?;
        let mut gradients = loss.backward();
        let next_eta = if let Some(state) = &eta {
            let grad = state
                .leaf
                .grad_remove(&mut gradients)
                .ok_or_else(|| invalid("eta gradient missing"))?;
            Some(state.candidate(read_one(&grad)?, &config)?)
        } else {
            None
        };
        let update = policy.backward_update(&mut gradients, config.learning_rate)?;
        if let Some(next) = next_eta {
            eta = Some(EtaState::from_state(&session, next)?);
        }
        session.synchronize()?;
        check_allocator(&session, &config.budget)?;
        completed += 1;
        let update_seconds = step_started.elapsed().as_secs_f64();
        if completed % config.validation_interval == 0 || completed == config.max_updates {
            calibrate_policy(&session, &config, &mut policy)?;
            let validation = evaluate_values(
                &session,
                &config,
                &mut policy,
                config.validation_seed,
                config.validation_paths,
                false,
            )?;
            let validation_risk =
                risk_objective::weighted_risk(&validation.pnl, None, config.objective)
                    .map_err(execution)?
                    .value;
            let improved = validation_risk < best;
            if improved {
                best = validation_risk;
                selected = policy.snapshot()?;
                selected_update = completed;
            }
            observations.push(TrainingObservation {
                update: completed,
                training_loss: loss_value,
                gradient_norm: update.gradient_l2_norm,
                update_seconds,
                validation_risk,
                selected: improved,
            });
            if let Some(path) = checkpoint_path {
                save_checkpoint(
                    path,
                    &TrainingCheckpoint {
                        schema: "volt.compute.hedge_training.v1".into(),
                        config: config.clone(),
                        completed_updates: completed,
                        current_policy: policy.snapshot()?,
                        current_eta: eta.as_ref().map(|v| v.state.clone()),
                        selected_policy: selected.clone(),
                        selected_update,
                        selected_validation_risk: best,
                    },
                )?;
            }
            if completed < config.max_updates
                && config.early_stopping.as_ref().is_some_and(|stopping| {
                    completed >= stopping.minimum_updates
                        && completed.saturating_sub(selected_update) >= stopping.patience_updates
                })
            {
                stop_reason = TrainingStopReason::ValidationPatience;
                break;
            }
        }
    }
    let training_seconds = started.elapsed().as_secs_f64();
    let mut selected_policy = HedgePolicy::from_snapshot(&session, selected.clone())?;
    let evaluation = evaluate(&session, &config, &mut selected_policy)?;
    let checkpoint_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&selected).map_err(execution)?)
    );
    session.synchronize()?;
    let mut deviations=vec!["Historical simplified-BK endpoint weights unspecified; declared quadrature rule recorded".into(),
        "Initialization/BatchNorm/Adam defaults explicitly frozen; historical framework details not fully specified".into(),
        "CVaR eta uses host f64 Adam; policy parameters/moments stay on selected device precision".into(),
        "Matched Heston model-hedge, integer-ledger audit, multi-seed paper campaign remain required".into()];
    if config.test_paths < 1_000_000 {
        deviations.push("Held-out sample count below original million-path comparison".into());
    }
    if config.policy.action_scales.is_some() || config.policy.feature_affine.is_some() {
        deviations.push("Explicit feature/action scaling differs from raw paper profile".into());
    }
    Ok(HedgeTrainingReport {
        schema: "volt.compute.deep_hedging_run.v1".into(),
        config,
        train_plan,
        evaluation_plan,
        completed_updates: completed,
        stop_reason,
        selected_update,
        selected_validation_risk: best,
        training_seconds,
        observations,
        evaluation,
        checkpoint_sha256,
        paper_replication_complete: false,
        deviations,
    })
}

struct EvaluationValues {
    pnl: Vec<f64>,
    no_hedge: Vec<f64>,
    cost: f64,
    turnover: f64,
    max_holding: f64,
    hash: String,
}
fn evaluate_values(
    session: &Session,
    config: &HedgeTrainingConfig,
    policy: &mut HedgePolicy,
    seed: u64,
    count: usize,
    retain: bool,
) -> Result<EvaluationValues, ComputeError> {
    let mut output = EvaluationValues {
        pnl: Vec::with_capacity(count),
        no_hedge: Vec::with_capacity(count),
        cost: 0.0,
        turnover: 0.0,
        max_holding: 0.0,
        hash: String::new(),
    };
    let mut hash = Sha256::new();
    for start in (0..count).step_by(config.evaluation_batch_paths) {
        let paths = config.evaluation_batch_paths.min(count - start);
        let batch = generate_batch(config, seed, start as u64, paths)?;
        hash.update(batch.source_hash.as_bytes());
        let prepared = PreparedHedgePaths::upload(session, &batch)?;
        let premium = config.premium_for_comparison;
        let rollout = hedge_application::policy_rollout(
            session,
            &prepared,
            policy,
            config.cost,
            premium,
            false,
            retain,
        )?;
        let values = read_vector(&rollout.pnl)?;
        let costs = read_vector(&rollout.costs)?;
        let turnover = read_vector(&rollout.turnover)?;
        for h in rollout.holdings {
            for value in h.try_to_vec_as::<f64>().map_err(execution)? {
                output.max_holding = output.max_holding.max(value.abs());
            }
        }
        if values.len() != paths {
            return Err(invalid("evaluation result shape mismatch"));
        }
        output.pnl.extend(values);
        output
            .no_hedge
            .extend(batch.payoffs.iter().map(|p| premium - p));
        output.cost += costs.iter().sum::<f64>();
        output.turnover += turnover.iter().sum::<f64>();
        check_allocator(session, &config.budget)?;
    }
    output.hash = format!("{:x}", hash.finalize());
    Ok(output)
}
pub fn evaluate(
    session: &Session,
    config: &HedgeTrainingConfig,
    policy: &mut HedgePolicy,
) -> Result<HedgeEvaluationReport, ComputeError> {
    let started = Instant::now();
    let values = evaluate_values(
        session,
        config,
        policy,
        config.test_seed,
        config.test_paths,
        true,
    )?;
    let candidate =
        risk_objective::weighted_risk(&values.pnl, None, config.objective).map_err(execution)?;
    let no_hedge = risk_objective::weighted_risk(&values.no_hedge, None, config.objective)
        .map_err(execution)?;
    let n = config.bootstrap_paths.min(config.test_paths);
    let paired_subset = risk_objective::paired_path_comparison(
        &values.pnl[..n],
        &values.no_hedge[..n],
        config.objective,
        BootstrapConfig {
            replicates: config.bootstrap_replicates,
            confidence: 0.95,
            seed: config.test_seed ^ 0xfeedbeef,
        },
    )
    .map_err(execution)?;
    Ok(HedgeEvaluationReport {
        candidate,
        no_hedge,
        paired_subset,
        bootstrap_subset_paths: n,
        mean_cost: values.cost / config.test_paths as f64,
        mean_turnover: values.turnover / config.test_paths as f64,
        max_absolute_holding: values.max_holding,
        paths: config.test_paths,
        path_stream_hash: values.hash,
        seconds: started.elapsed().as_secs_f64(),
        sample_candidate_pnl: values.pnl.into_iter().take(16).collect(),
    })
}
fn calibrate_policy(
    session: &Session,
    config: &HedgeTrainingConfig,
    policy: &mut HedgePolicy,
) -> Result<(), ComputeError> {
    if config.calibration_paths == 0 || !config.policy.batch_norm.enabled {
        return Ok(());
    }
    let batch = generate_batch(config, config.calibration_seed, 0, config.calibration_paths)?;
    let prepared = PreparedHedgePaths::upload(session, &batch)?;
    hedge_application::calibrate_policy_paths(session, &prepared, policy)?;
    session.synchronize()?;
    check_allocator(session, &config.budget)
}
fn default_calibration_seed() -> u64 {
    5011
}

fn check_allocator(session: &Session, budget: &MemoryBudget) -> Result<(), ComputeError> {
    if session
        .device()
        .memory_pool_usage()
        .is_some_and(|u| u.bytes_reserved > budget.compute_bytes)
    {
        return Err(ComputeError::BudgetExceeded(
            "hedge allocator reservation exceeds compute budget".into(),
        ));
    }
    Ok(())
}
fn read_vector(t: &Tensor<1>) -> Result<Vec<f64>, ComputeError> {
    let values = t
        .clone()
        .without_autodiff()
        .try_to_vec_as::<f64>()
        .map_err(execution)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(ComputeError::Execution("nonfinite hedge output".into()));
    }
    Ok(values)
}
fn read_one(t: &Tensor<1>) -> Result<f64, ComputeError> {
    let values = read_vector(t)?;
    if values.len() != 1 {
        return Err(invalid("loss must be scalar"));
    }
    Ok(values[0])
}
pub fn validate_resume(
    config: &HedgeTrainingConfig,
    c: &TrainingCheckpoint,
) -> Result<(), ComputeError> {
    let mut expected = c.config.clone();
    expected.target = config.target;
    expected.budget = config.budget.clone();
    expected.max_updates = config.max_updates;
    crate::hedge_policy::validate_policy_snapshot(&c.current_policy)?;
    crate::hedge_policy::validate_policy_snapshot(&c.selected_policy)?;
    let objective_eta = matches!(config.objective, RiskObjective::Cvar { .. });
    if c.schema != "volt.compute.hedge_training.v1"
        || c.completed_updates > config.max_updates
        || c.current_policy.updates != c.completed_updates
        || c.selected_update > c.completed_updates
        || c.selected_policy.updates != c.selected_update
        || !c.selected_validation_risk.is_finite()
        || c.current_policy.batch_norm.iter().any(|n| {
            n.batches_tracked
                != if config.policy.batch_norm.enabled {
                    c.completed_updates
                } else {
                    0
                }
        })
        || c.selected_policy.batch_norm.iter().any(|n| {
            n.batches_tracked
                != if config.policy.batch_norm.enabled {
                    c.selected_update
                } else {
                    0
                }
        })
        || c.current_policy.precision != config.precision
        || c.selected_policy.precision != config.precision
        || serde_json::to_value(&c.current_policy.config).map_err(execution)?
            != serde_json::to_value(&config.policy).map_err(execution)?
        || serde_json::to_value(&c.selected_policy.config).map_err(execution)?
            != serde_json::to_value(&config.policy).map_err(execution)?
        || objective_eta != c.current_eta.is_some()
        || (config.precision == Precision::F32
            && c.current_eta
                .as_ref()
                .is_some_and(|e| !(e.value as f32).is_finite()))
        || c.current_eta.as_ref().is_some_and(|e| {
            e.step != c.completed_updates
                || [e.value, e.first_moment, e.second_moment]
                    .iter()
                    .any(|v| !v.is_finite())
                || e.second_moment < 0.0
        })
        || serde_json::to_value(expected).map_err(execution)?
            != serde_json::to_value(config).map_err(execution)?
    {
        return Err(invalid(
            "checkpoint/config/stream mismatch; only target,budgets,maxupdates may change",
        ));
    }
    Ok(())
}
pub fn save_checkpoint(path: &Path, checkpoint: &TrainingCheckpoint) -> Result<(), ComputeError> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(execution)?;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(execution)?
        .as_nanos();
    let temp = path.with_extension(format!("partial-{}-{stamp}", std::process::id()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)
            .map_err(execution)?;
        serde_json::to_writer(&mut file, checkpoint).map_err(execution)?;
        file.flush().map_err(execution)?;
        file.sync_all().map_err(execution)?;
        std::fs::rename(&temp, path).map_err(execution)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}
pub fn load_checkpoint(path: &Path) -> Result<TrainingCheckpoint, ComputeError> {
    serde_json::from_reader(std::fs::File::open(path).map_err(execution)?).map_err(execution)
}
fn invalid(s: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn execution(e: impl std::fmt::Display) -> ComputeError {
    ComputeError::Execution(e.to_string())
}
