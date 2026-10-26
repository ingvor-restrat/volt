//! Persistent episode learner. Market scenarios are constants; gradients pass
//! through continuous holdings, financed cash, costs and the terminal objective.
use crate::{
    ComputeError, ExecutionTarget, Precision, Session,
    budget::MemoryBudget,
    hedge_application::{HedgePathBatch, plan_hedging, tensor_risk},
    hedge_policy::{HedgePolicy, HedgePolicyConfig, PolicySnapshot},
    risk_objective::RiskObjective,
};
use burn::tensor::{DType, Tensor, TensorCreationOptions, TensorData, activation::relu};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const SCHEMA: &str = "volt.native_learner.v2";
pub const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const MAX_BATCHES: usize = 4;

fn invalid(s: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn execution(e: impl std::fmt::Display) -> ComputeError {
    ComputeError::Execution(e.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Objective {
    SquaredLoss,
    SemiSquaredLoss,
    /// Exact empirical fractional-tail objective; a subgradient away from ties.
    Cvar {
        alpha: f64,
    },
    Entropic {
        risk_aversion: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LearnerConfig {
    pub target: ExecutionTarget,
    pub precision: Precision,
    pub policy_engine: String,
    pub policy: HedgePolicyConfig,
    pub seed: u64,
    pub learning_rate: f64,
    pub objective: Objective,
    pub premium: f64,
    /// Per-instrument fraction of traded notional, including opening trades.
    pub trading_costs: Vec<f64>,
    pub liquidation_costs: Vec<f64>,
    /// Per-interval gross cash growth; negative cash accrues at the same rate.
    pub cash_growth: Vec<f64>,
    /// Per-interval, per-instrument gross total-return multiplier.
    /// A dividend-paying stock uses exp(q*dt); an option uses 1.
    pub income_growth: Vec<f64>,
    pub budget: MemoryBudget,
    /// Serialized dataset partition metadata, independent of trajectory count.
    pub provenance_budget_bytes: usize,
}
impl Default for LearnerConfig {
    fn default() -> Self {
        let mut policy = HedgePolicyConfig::paper_heston(512);
        policy.time_steps = 2;
        policy.hidden_width = 8;
        policy.batch_norm.enabled = false;
        Self {
            target: ExecutionTarget::Cpu,
            precision: Precision::F64,
            policy_engine: "volt_time_mlp".into(),
            policy,
            seed: 23,
            learning_rate: 0.001,
            objective: Objective::SquaredLoss,
            premium: 0.0,
            trading_costs: vec![0.0; 2],
            liquidation_costs: vec![0.0; 2],
            cash_growth: vec![1.0; 2],
            income_growth: vec![1.0; 4],
            budget: MemoryBudget::default(),
            provenance_budget_bytes: 1024 * 1024,
        }
    }
}
impl LearnerConfig {
    fn validate(&self) -> Result<(), ComputeError> {
        self.policy.validate()?;
        if self.provenance_budget_bytes == 0
            || self.provenance_budget_bytes > MAX_MESSAGE_BYTES / 4
            || self.provenance_budget_bytes as u64 > self.budget.host_working_bytes / 8
        {
            return Err(invalid(
                "provenance metadata allowance must fit host and protocol budgets",
            ));
        }
        let h = self.policy.hedge_count;
        let t = self.policy.time_steps;
        if self.policy_engine != "volt_time_mlp" || self.policy.batch_norm.enabled {
            return Err(invalid(
                "native v1 supports volt_time_mlp with batch_norm disabled",
            ));
        }
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.premium.is_finite()
            || self.trading_costs.len() != h
            || self.liquidation_costs.len() != h
            || self.cash_growth.len() != t
            || self.income_growth.len() != t * h
            || self
                .trading_costs
                .iter()
                .chain(&self.liquidation_costs)
                .any(|x| !x.is_finite() || *x < 0.0)
            || self
                .cash_growth
                .iter()
                .chain(&self.income_growth)
                .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(invalid(
                "invalid learning rate, premium, cost or growth arrays",
            ));
        }
        match self.objective {
            Objective::Cvar { alpha } if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) => {
                return Err(invalid("invalid CVaR confidence"));
            }
            Objective::Entropic { risk_aversion }
                if !risk_aversion.is_finite() || risk_aversion <= 0.0 =>
            {
                return Err(invalid("invalid entropic risk aversion"));
            }
            _ => {}
        }
        if self.precision == Precision::F32
            && self
                .trading_costs
                .iter()
                .chain(&self.liquidation_costs)
                .chain(&self.cash_growth)
                .chain(&self.income_growth)
                .chain([&self.premium, &self.learning_rate])
                .any(|x| !(*x as f32).is_finite())
        {
            return Err(invalid("config values overflow f32"));
        }
        plan_hedging(
            &self.policy,
            self.policy.max_batch_rows,
            self.precision,
            self.target,
            &self.budget,
            true,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetPartition {
    pub split: Split,
    #[serde(with = "decimal_u64")]
    pub first_path: u64,
    #[serde(with = "decimal_u64")]
    pub paths: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchProvenance {
    /// Immutable archive/generator identity. Producer guarantees unique scenarios.
    pub dataset_id: String,
    #[serde(with = "decimal_u64")]
    pub first_path: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpisodeBatch {
    pub paths: usize,
    /// Contiguous range within a registered immutable dataset partition.
    pub provenance: BatchProvenance,
    /// Flattened [path, time including terminal, instrument] quoted-unit prices.
    pub marks: Vec<f64>,
    /// Flattened [path, decision, feature]. Each row uses current/past data only.
    pub features: Vec<f64>,
    /// Terminal liabilities, positive for an amount owed. Multipliers applied upstream.
    pub payoffs: Vec<f64>,
}
struct ResidentBatch {
    split: Split,
    batch: EpisodeBatch,
    hash: String,
    // Upload once and retain native device tensors across training commands.
    marks: Vec<Tensor<2>>,
    features: Vec<Tensor<2>>,
    payoffs: Tensor<1>,
    compute_bytes: u64,
    host_bytes: u64,
    update_compute_bytes: u64,
    update_host_bytes: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    schema: String,
    config: LearnerConfig,
    policy: PolicySnapshot,
    /// Split assignments survive restart to prevent held-out reuse for training.
    datasets: BTreeMap<String, Vec<DatasetPartition>>,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Configure {
        config: Box<LearnerConfig>,
    },
    RegisterDataset {
        dataset_id: String,
        partitions: Vec<DatasetPartition>,
    },
    Upload {
        batch_id: String,
        split: Split,
        batch: EpisodeBatch,
    },
    DropBatch {
        batch_id: String,
    },
    Update {
        batch_id: String,
        expected_version: u64,
    },
    Evaluate {
        batch_id: String,
        #[serde(default)]
        retain_holdings: bool,
    },
    Infer {
        step: usize,
        features: Vec<f64>,
        previous_holdings: Vec<f64>,
        paths: usize,
    },
    Checkpoint,
    Restore {
        checkpoint: Box<Checkpoint>,
    },
    Status,
    Shutdown,
}

pub struct NativeLearner {
    config: LearnerConfig,
    session: Session,
    policy: HedgePolicy,
    batches: BTreeMap<String, ResidentBatch>,
    datasets: BTreeMap<String, Vec<DatasetPartition>>,
}

fn tensor2(session: &Session, data: Vec<f64>, shape: [usize; 2]) -> Tensor<2> {
    let dtype = match session.precision() {
        Precision::F32 => DType::F32,
        Precision::F64 => DType::F64,
    };
    Tensor::from_data(
        TensorData::new(data, shape),
        TensorCreationOptions::new(session.device().clone()).with_dtype(dtype),
    )
}

impl NativeLearner {
    pub fn new(config: LearnerConfig) -> Result<Self, ComputeError> {
        config.validate()?;
        let session = Session::new(
            config.target,
            config.precision,
            config.policy.max_batch_rows,
        )?;
        let policy = HedgePolicy::new(&session, config.policy.clone(), config.seed)?;
        Ok(Self {
            config,
            session,
            policy,
            batches: BTreeMap::new(),
            datasets: BTreeMap::new(),
        })
    }
    pub fn restore(checkpoint: Checkpoint) -> Result<Self, ComputeError> {
        if checkpoint.schema != SCHEMA
            || serde_json::to_value(&checkpoint.config.policy).map_err(execution)?
                != serde_json::to_value(&checkpoint.policy.config).map_err(execution)?
        {
            return Err(invalid("incompatible native checkpoint"));
        }
        let mut learner = Self::new(checkpoint.config)?;
        learner.policy = HedgePolicy::from_snapshot(&learner.session, checkpoint.policy)?;
        for (id, partitions) in checkpoint.datasets {
            learner.register_dataset(id, partitions)?;
        }
        Ok(learner)
    }
    pub fn handle(&mut self, command: Command) -> Result<Value, ComputeError> {
        match command {
            Command::Status => Ok(
                json!({"schema":SCHEMA,"backend":"burn","policy_engine":self.config.policy_engine,"version":self.policy.updates(),"target":self.config.target,"precision":self.config.precision,"fusion_compiled":cfg!(feature="fusion"),"fusion_active":cfg!(feature="fusion") && self.config.target != ExecutionTarget::Cpu,"resident_batches":self.batches.len(),"dataset_count":self.datasets.len(),"provenance_bytes":serde_json::to_vec(&self.datasets).map_err(execution)?.len(),"config":self.config}),
            ),
            Command::Checkpoint => serde_json::to_value(Checkpoint {
                schema: SCHEMA.into(),
                config: self.config.clone(),
                policy: self.policy.snapshot()?,
                datasets: self.datasets.clone(),
            })
            .map_err(execution),
            Command::RegisterDataset {
                dataset_id,
                partitions,
            } => self.register_dataset(dataset_id, partitions),
            Command::Upload {
                batch_id,
                split,
                batch,
            } => self.upload(batch_id, split, batch),
            Command::DropBatch { batch_id } => {
                self.batches
                    .remove(&batch_id)
                    .ok_or_else(|| invalid("unknown batch_id"))?;
                Ok(json!({"dropped":batch_id}))
            }
            Command::Update {
                batch_id,
                expected_version,
            } => {
                if expected_version != self.policy.updates() {
                    return Err(invalid("stale expected_version; query status before retry"));
                }
                let batch = self
                    .batches
                    .get(&batch_id)
                    .ok_or_else(|| invalid("unknown batch_id"))?;
                if batch.split != Split::Train {
                    return Err(invalid("updates require a train batch"));
                }
                // Commit only a successful complete update; failures preserve live weights/Adam.
                let mut candidate = self.policy.fork_trainable();
                let (pnl, _, _) = rollout(
                    &self.session,
                    &self.config,
                    batch,
                    &mut candidate,
                    true,
                    false,
                )?;
                let loss = objective(&self.session, pnl, &self.config.objective)?;
                let value = self.session.read_vector(&loss)?[0];
                if !value.is_finite() {
                    return Err(invalid("nonfinite objective"));
                }
                let start = std::time::Instant::now();
                let stats =
                    candidate.backward_update(&mut loss.backward(), self.config.learning_rate)?;
                self.session.synchronize()?;
                self.policy = candidate;
                Ok(
                    json!({"version":self.policy.updates(),"loss_before_update":value,"gradient_l2_norm":stats.gradient_l2_norm,"updated_parameters":stats.updated_parameters,"backward_update_ms":start.elapsed().as_secs_f64()*1000.0,"batch_id":batch_id,"batch_sha256":batch.hash,"paths":batch.batch.paths}),
                )
            }
            Command::Evaluate {
                batch_id,
                retain_holdings,
            } => {
                let batch = self
                    .batches
                    .get(&batch_id)
                    .ok_or_else(|| invalid("unknown batch_id"))?;
                let (pnl, costs, holdings) = rollout(
                    &self.session,
                    &self.config,
                    batch,
                    &mut self.policy,
                    false,
                    retain_holdings,
                )?;
                let loss = objective(&self.session, pnl.clone(), &self.config.objective)?;
                let loss = self.session.read_vector(&loss)?[0];
                let pnl = self.session.read_vector(&pnl)?;
                let costs = self.session.read_vector(&costs)?;
                let holdings = holdings
                    .into_iter()
                    .map(|x| x.try_to_vec_as::<f64>().map_err(execution))
                    .collect::<Result<Vec<_>, _>>()?;
                if !loss.is_finite()
                    || pnl
                        .iter()
                        .chain(&costs)
                        .chain(holdings.iter().flatten())
                        .any(|x| !x.is_finite())
                {
                    return Err(invalid("nonfinite evaluation"));
                }
                Ok(
                    json!({"version":self.policy.updates(),"split":batch.split,"batch_sha256":batch.hash,"loss":loss,"pnl":pnl,"costs":costs,"holdings":holdings}),
                )
            }
            Command::Infer {
                step,
                features,
                previous_holdings,
                paths,
            } => {
                let c = &self.config.policy;
                if paths == 0
                    || paths > c.max_batch_rows
                    || features.len() != paths * c.feature_count
                    || previous_holdings.len() != paths * c.hedge_count
                    || features.iter().chain(&previous_holdings).any(|x| {
                        !x.is_finite()
                            || (self.config.precision == Precision::F32 && !(*x as f32).is_finite())
                    })
                {
                    return Err(invalid("invalid inference shapes/values"));
                }
                let result = self.policy.forward_at(
                    step,
                    tensor2(&self.session, features, [paths, c.feature_count]),
                    tensor2(&self.session, previous_holdings, [paths, c.hedge_count]),
                    false,
                )?;
                let holdings = result.try_to_vec_as::<f64>().map_err(execution)?;
                if holdings.iter().any(|x| !x.is_finite()) {
                    return Err(invalid("nonfinite holdings"));
                }
                Ok(json!({"version":self.policy.updates(),"holdings":holdings}))
            }
            _ => Err(invalid(
                "configure/restore/shutdown are worker lifecycle commands",
            )),
        }
    }
    fn register_dataset(
        &mut self,
        dataset_id: String,
        mut partitions: Vec<DatasetPartition>,
    ) -> Result<Value, ComputeError> {
        if dataset_id.is_empty() || dataset_id.len() > 128 || partitions.is_empty() {
            return Err(invalid(
                "dataset identity and nonempty partitions are required",
            ));
        }
        partitions.sort_by_key(|p| p.first_path);
        let mut previous_end = 0;
        for p in &partitions {
            let end = p
                .first_path
                .checked_add(p.paths)
                .ok_or_else(|| invalid("dataset path range overflow"))?;
            if p.paths == 0 || p.first_path < previous_end {
                return Err(invalid("dataset partitions must be nonempty and disjoint"));
            }
            previous_end = end;
        }
        if let Some(existing) = self.datasets.get(&dataset_id) {
            if *existing != partitions {
                return Err(invalid("registered dataset partitions are immutable"));
            }
            return Ok(
                json!({"dataset_id":dataset_id,"partitions":existing,"already_registered":true}),
            );
        }
        let mut candidate = self.datasets.clone();
        candidate.insert(dataset_id.clone(), partitions.clone());
        let bytes = serde_json::to_vec(&candidate).map_err(execution)?.len();
        if bytes > self.config.provenance_budget_bytes {
            return Err(ComputeError::BudgetExceeded(
                "dataset partition metadata exceeds provenance allowance".into(),
            ));
        }
        self.datasets = candidate;
        Ok(json!({"dataset_id":dataset_id,"partitions":partitions,"provenance_bytes":bytes}))
    }
    fn upload(
        &mut self,
        batch_id: String,
        split: Split,
        batch: EpisodeBatch,
    ) -> Result<Value, ComputeError> {
        if batch_id.is_empty()
            || batch_id.len() > 128
            || self.batches.contains_key(&batch_id)
            || self.batches.len() >= MAX_BATCHES
        {
            return Err(invalid(
                "batch_id must be new and bounded; at most four resident batches",
            ));
        }
        let c = &self.config.policy;
        let end = batch
            .provenance
            .first_path
            .checked_add(batch.paths as u64)
            .ok_or_else(|| invalid("batch path range overflow"))?;
        let partitions = self
            .datasets
            .get(&batch.provenance.dataset_id)
            .ok_or_else(|| invalid("register dataset partitions before upload"))?;
        if batch.paths == 0
            || !partitions.iter().any(|p| {
                p.split == split
                    && batch.provenance.first_path >= p.first_path
                    && end <= p.first_path + p.paths
            })
        {
            return Err(invalid(
                "batch range must stay within its registered split partition",
            ));
        }
        let bytes = serde_json::to_vec(&batch).map_err(execution)?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let model = HedgePathBatch {
            paths: batch.paths,
            steps: c.time_steps,
            hedge_count: c.hedge_count,
            feature_count: c.feature_count,
            marks: batch.marks.clone(),
            features: batch.features.clone(),
            payoffs: batch.payoffs.clone(),
            source_hash: hash.clone(),
        };
        model.validate()?;
        // Reuse Volt's full recurrent-policy planner. Reserve a second full job
        // for transactional candidate weights and this financed recurrence.
        let plan = plan_hedging(
            c,
            batch.paths,
            self.config.precision,
            self.config.target,
            &self.config.budget,
            true,
        )?;
        let update_compute = plan
            .estimated_compute_bytes
            .checked_mul(2)
            .ok_or_else(|| invalid("memory overflow"))?;
        let update_host = plan
            .estimated_host_working_bytes
            .checked_mul(2)
            .ok_or_else(|| invalid("memory overflow"))?;
        let old_compute: u64 = self.batches.values().map(|b| b.compute_bytes).sum();
        let old_host: u64 = self.batches.values().map(|b| b.host_bytes).sum();
        let scalar = if self.config.precision == Precision::F64 {
            8
        } else {
            4
        };
        let elements = (batch.marks.len() + batch.features.len() + batch.payoffs.len()) as u64;
        let compute = elements.saturating_mul(scalar);
        let host = elements
            .saturating_mul(8)
            .saturating_add(bytes.len() as u64)
            .saturating_add((batch.paths as u64).saturating_mul(512));
        // Resident input tensors accumulate. Only one rollout/update executes
        // at a time, so share the largest working reserve across all batches.
        let largest_compute = self
            .batches
            .values()
            .map(|b| b.update_compute_bytes)
            .max()
            .unwrap_or(0)
            .max(update_compute);
        let largest_host = self
            .batches
            .values()
            .map(|b| b.update_host_bytes)
            .max()
            .unwrap_or(0)
            .max(update_host);
        if compute
            .saturating_add(old_compute)
            .saturating_add(largest_compute)
            > plan.compute_admission_limit_bytes
            || host
                .saturating_add(old_host)
                .saturating_add(largest_host)
                .saturating_add((self.config.provenance_budget_bytes as u64).saturating_mul(4))
                > plan.host_admission_limit_bytes
        {
            return Err(ComputeError::BudgetExceeded(
                "resident episode batches plus update reserve exceed budget".into(),
            ));
        }
        if self.config.precision == Precision::F32
            && batch
                .marks
                .iter()
                .chain(&batch.features)
                .chain(&batch.payoffs)
                .any(|v| !(*v as f32).is_finite())
        {
            return Err(invalid("batch values overflow f32"));
        }
        let mut marks = Vec::new();
        let mut features = Vec::new();
        for t in 0..=c.time_steps {
            marks.push(tensor2(
                &self.session,
                (0..batch.paths)
                    .flat_map(|p| {
                        let s = (p * (c.time_steps + 1) + t) * c.hedge_count;
                        batch.marks[s..s + c.hedge_count].iter().copied()
                    })
                    .collect(),
                [batch.paths, c.hedge_count],
            ));
            if t < c.time_steps {
                features.push(tensor2(
                    &self.session,
                    (0..batch.paths)
                        .flat_map(|p| {
                            let s = (p * c.time_steps + t) * c.feature_count;
                            batch.features[s..s + c.feature_count].iter().copied()
                        })
                        .collect(),
                    [batch.paths, c.feature_count],
                ));
            }
        }
        let payoffs = self.session.vector(batch.payoffs.clone());
        self.session.synchronize()?;
        let paths = batch.paths;
        self.batches.insert(
            batch_id.clone(),
            ResidentBatch {
                split,
                batch,
                hash: hash.clone(),
                marks,
                features,
                payoffs,
                compute_bytes: compute,
                host_bytes: host,
                update_compute_bytes: update_compute,
                update_host_bytes: update_host,
            },
        );
        Ok(
            json!({"batch_id":batch_id,"batch_sha256":hash,"paths":paths,"split":split,"resident_compute_bytes":compute,"resident_host_bytes":host,"estimated_working_compute_bytes":compute+old_compute+largest_compute,"estimated_working_host_bytes":host+old_host+largest_host+(self.config.provenance_budget_bytes as u64)*4}),
        )
    }
}

fn rollout(
    session: &Session,
    config: &LearnerConfig,
    batch: &ResidentBatch,
    policy: &mut HedgePolicy,
    train: bool,
    retain: bool,
) -> Result<(Tensor<1>, Tensor<1>, Vec<Tensor<2>>), ComputeError> {
    let n = batch.batch.paths;
    let h = config.policy.hedge_count;
    let mut previous = tensor2(session, vec![0.0; n * h], [n, h]);
    let mut value = session.vector(vec![config.premium; n]);
    let mut costs = session.vector(vec![0.0; n]);
    let fees = tensor2(session, config.trading_costs.clone(), [1, h]);
    let mut holdings = Vec::new();
    for t in 0..config.policy.time_steps {
        let target = policy.forward_at(t, batch.features[t].clone(), previous.clone(), train)?;
        let charge = ((target.clone() - previous).abs() * batch.marks[t].clone() * fees.clone())
            .sum_dim(1)
            .reshape([n]);
        let cash = value
            - (target.clone() * batch.marks[t].clone())
                .sum_dim(1)
                .reshape([n])
            - charge.clone();
        let income = tensor2(
            session,
            config.income_growth[t * h..(t + 1) * h].to_vec(),
            [1, h],
        );
        value = cash * config.cash_growth[t]
            + (target.clone() * batch.marks[t + 1].clone() * income)
                .sum_dim(1)
                .reshape([n]);
        costs = costs + charge.without_autodiff();
        previous = target.clone();
        if retain {
            holdings.push(target.without_autodiff());
        }
    }
    let terminal = (previous.abs()
        * batch.marks[config.policy.time_steps].clone()
        * tensor2(session, config.liquidation_costs.clone(), [1, h]))
    .sum_dim(1)
    .reshape([n]);
    Ok((
        value - terminal.clone() - batch.payoffs.clone(),
        costs + terminal.without_autodiff(),
        holdings,
    ))
}

fn objective(
    session: &Session,
    pnl: Tensor<1>,
    kind: &Objective,
) -> Result<Tensor<1>, ComputeError> {
    let n = pnl.dims()[0];
    match *kind {
        Objective::SquaredLoss => tensor_risk(pnl, RiskObjective::SquaredLoss, None),
        Objective::SemiSquaredLoss => Ok(relu(-pnl).square().mean()),
        Objective::Entropic { risk_aversion } => {
            tensor_risk(pnl, RiskObjective::Entropic { risk_aversion }, None)
        }
        Objective::Cvar { alpha } => {
            let values = session.read_vector(&pnl)?;
            let mut indices: Vec<_> = (0..n).collect();
            indices.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
            let mass = (1.0 - alpha) * n as f64;
            let mut left = mass;
            let mut weights = vec![0.0; n];
            for i in indices {
                let take = left.min(1.0);
                weights[i] = take / mass;
                left -= take;
                if left <= 0.0 {
                    break;
                }
            }
            Ok((-pnl * session.vector(weights)).sum())
        }
    }
}

// Accept exact integer JSON inputs and decimal strings; emit strings so browser
// clients preserve range identities above JavaScript's exact-integer limit.
mod decimal_u64 {
    use serde::{
        Deserializer, Serializer,
        de::{self, Visitor},
    };
    use std::fmt;
    pub fn serialize<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        struct Decimal;
        impl Visitor<'_> for Decimal {
            type Value = u64;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "an unsigned integer or decimal string")
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<u64, E> {
                Ok(value)
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<u64, E> {
                value.parse().map_err(E::custom)
            }
        }
        deserializer.deserialize_any(Decimal)
    }
}
