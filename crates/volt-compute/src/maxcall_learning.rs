//! Multi-asset continuation regression with explicit BatchNorm reconstruction.
//! Existing one-asset policies/checkpoints are independent of this schema.
use crate::{
    ComputeError, ExecutionTarget, Precision, Session,
    hedge_policy::{AdamEpsilonConvention, strict_matmul},
    multi_asset_market::{MaxCallMarket, generate_paths},
};
use burn::tensor::{DType, Int, Tensor, TensorCreationOptions, TensorData, activation::tanh};
use serde::{Deserialize, Serialize};
use std::time::Instant;
pub const SCHEMA: &str = "volt.compute.maxcall_learning.v1";
fn invalid(x: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(x.into())
}
fn execution(x: impl ToString) -> ComputeError {
    ComputeError::Execution(x.to_string())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductArithmetic {
    StrictNativeDtype,
    BackendMatmul,
}
fn is_false(x: &bool) -> bool {
    !*x
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MaxCallBatchNorm {
    pub input: bool,
    pub hidden: bool,
    pub epsilon: f64,
    /// Weight of NEW population moments; retention=1-new_weight.
    pub new_weight: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub canonical_zero_translations: bool,
}
impl Default for MaxCallBatchNorm {
    fn default() -> Self {
        Self {
            input: true,
            hidden: true,
            epsilon: 1e-3,
            new_weight: 0.01,
            canonical_zero_translations: false,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningRateStage {
    pub updates: usize,
    pub learning_rate: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxCallTrainingConfig {
    pub market: MaxCallMarket,
    pub target: ExecutionTarget,
    pub precision: Precision,
    pub hidden_width: usize,
    pub batch_rows: usize,
    pub training_paths: usize,
    pub validation_paths: usize,
    pub validation_every: usize,
    pub final_date_schedule: Vec<LearningRateStage>,
    pub earlier_date_schedule: Vec<LearningRateStage>,
    pub normalization: MaxCallBatchNorm,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub epsilon_convention: AdamEpsilonConvention,
    pub warm_start: bool,
    pub training_seed: u64,
    pub validation_seed: u64,
    pub initialization_seed: u64,
    pub arithmetic: ProductArithmetic,
    pub max_expanded_elements: usize,
    pub compute_budget_bytes: u64,
    pub host_budget_bytes: u64,
}
impl Default for MaxCallTrainingConfig {
    fn default() -> Self {
        Self {
            market: MaxCallMarket::paper_five(),
            target: ExecutionTarget::Cpu,
            precision: Precision::F64,
            hidden_width: 55,
            batch_rows: 8192,
            training_paths: 131072,
            validation_paths: 16384,
            validation_every: 500,
            final_date_schedule: vec![
                LearningRateStage {
                    updates: 1000,
                    learning_rate: 0.1,
                },
                LearningRateStage {
                    updates: 2000,
                    learning_rate: 0.01,
                },
                LearningRateStage {
                    updates: 2000,
                    learning_rate: 0.001,
                },
                LearningRateStage {
                    updates: 1000,
                    learning_rate: 0.0001,
                },
            ],
            earlier_date_schedule: vec![
                LearningRateStage {
                    updates: 500,
                    learning_rate: 0.1,
                },
                LearningRateStage {
                    updates: 1000,
                    learning_rate: 0.01,
                },
                LearningRateStage {
                    updates: 1500,
                    learning_rate: 0.001,
                },
                LearningRateStage {
                    updates: 500,
                    learning_rate: 0.0001,
                },
            ],
            normalization: Default::default(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            epsilon_convention: AdamEpsilonConvention::BiasCorrected,
            warm_start: true,
            training_seed: 20100000001,
            validation_seed: 20200000001,
            initialization_seed: 41003,
            arithmetic: ProductArithmetic::StrictNativeDtype,
            max_expanded_elements: 128 * 1024 * 1024,
            compute_budget_bytes: 4 * 1024 * 1024 * 1024,
            host_budget_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCallMemoryPlan {
    pub modeled_compute_bytes: u64,
    pub modeled_host_bytes: u64,
    pub parameters: usize,
    pub largest_expanded_product: usize,
}
impl MaxCallTrainingConfig {
    pub fn validate(&self) -> Result<MaxCallMemoryPlan, ComputeError> {
        self.market.validate()?;
        let d = self.market.spots.len();
        let h = self.hidden_width;
        let f = d + 1;
        if h == 0
            || h > 256
            || self.batch_rows < 2
            || self.batch_rows > 65536
            || self.training_paths < self.batch_rows
            || self.training_paths > 2_000_000
            || self.validation_paths < 2
            || self.validation_paths > 2_000_000
            || self.validation_every == 0
            || self.training_seed == self.validation_seed
            || self.final_date_schedule.is_empty()
            || self.earlier_date_schedule.is_empty()
            || !self.normalization.epsilon.is_finite()
            || self.normalization.epsilon <= 0.
            || !self.normalization.new_weight.is_finite()
            || self.normalization.new_weight <= 0.
            || self.normalization.new_weight > 1.
            || (self.normalization.canonical_zero_translations && !self.normalization.hidden)
            || !self.beta1.is_finite()
            || !(0. ..1.).contains(&self.beta1)
            || !self.beta2.is_finite()
            || !(0. ..1.).contains(&self.beta2)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.
        {
            return Err(invalid("invalid max-call dimensions/seeds/BN/optimizer"));
        }
        for stages in [&self.final_date_schedule, &self.earlier_date_schedule] {
            let count = stages.iter().try_fold(0usize, |n, s| {
                if s.updates == 0
                    || !s.learning_rate.is_finite()
                    || s.learning_rate <= 0.
                    || s.learning_rate > 1.
                {
                    return None;
                }
                n.checked_add(s.updates)
            });
            if count.is_none_or(|n| n > 1_000_000) {
                return Err(invalid("invalid explicit learning-rate stage counts"));
            }
        }
        let params = layout(f, h, 1, &self.normalization)
            .iter()
            .map(|(_, s)| s[0] * s[1])
            .sum::<usize>();
        let largest = self
            .batch_rows
            .checked_mul(f.max(h))
            .and_then(|n| n.checked_mul(h))
            .ok_or_else(|| invalid("expanded shape overflow"))?;
        if self.arithmetic == ProductArithmetic::StrictNativeDtype
            && largest > self.max_expanded_elements
        {
            return Err(ComputeError::BudgetExceeded(
                "strict product exceeds expanded-element cap".into(),
            ));
        }
        let bytes = if self.precision == Precision::F64 {
            8u128
        } else {
            4
        };
        let dataset =
            (self.training_paths + self.validation_paths) as u128 * (f + 1) as u128 * bytes;
        let products = self.batch_rows as u128 * (f * h + h * h + h) as u128;
        let tape = if self.target == ExecutionTarget::Cpu
            || self.arithmetic == ProductArithmetic::BackendMatmul
        {
            self.batch_rows as u128 * (f + 8 * h + 16) as u128 * bytes * 24
        } else {
            products * bytes * 8
        };
        let compute = dataset + tape + params as u128 * bytes * 12 + 32 * 1024 * 1024;
        let host = (self.training_paths + self.validation_paths) as u128
            * (self.market.times.len() * d + 3 * f + 16) as u128
            * 8
            + params as u128 * 64
            + 32 * 1024 * 1024;
        if compute > self.compute_budget_bytes as u128 || host > self.host_budget_bytes as u128 {
            return Err(ComputeError::BudgetExceeded(format!(
                "max-call modeled compute {compute},host {host} exceed allowances"
            )));
        }
        Ok(MaxCallMemoryPlan {
            modeled_compute_bytes: compute as u64,
            modeled_host_bytes: host as u64,
            parameters: params,
            largest_expanded_product: largest,
        })
    }
    fn schedule(&self, date: usize) -> &[LearningRateStage] {
        if (1..self.market.times.len() - 1)
            .rev()
            .find(|n| self.market.exercise[*n])
            == Some(date)
        {
            &self.final_date_schedule
        } else {
            &self.earlier_date_schedule
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxCallParameter {
    pub name: String,
    pub shape: [usize; 2],
    pub values: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxCallNormalizationState {
    pub name: String,
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub batches: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxCallNetworkSnapshot {
    pub input_width: usize,
    pub hidden_width: usize,
    pub output_width: usize,
    pub normalization: MaxCallBatchNorm,
    pub parameters: Vec<MaxCallParameter>,
    pub running: Vec<MaxCallNormalizationState>,
}
fn layout(f: usize, h: usize, o: usize, bn: &MaxCallBatchNorm) -> Vec<(String, [usize; 2])> {
    let mut v = vec![
        ("hidden0.weight".into(), [f, h]),
        ("hidden0.bias".into(), [1, h]),
        ("hidden1.weight".into(), [h, h]),
        ("hidden1.bias".into(), [1, h]),
        ("output.weight".into(), [h, o]),
        ("output.bias".into(), [1, o]),
    ];
    for (name, n) in norm_layout(f, h, bn) {
        v.push((format!("{name}.gamma"), [1, n]));
        v.push((format!("{name}.beta"), [1, n]));
    }
    v
}
fn norm_layout(f: usize, h: usize, b: &MaxCallBatchNorm) -> Vec<(String, usize)> {
    let mut v = Vec::new();
    if b.input {
        v.push(("input_bn".into(), f));
    }
    if b.hidden {
        v.push(("hidden0_bn".into(), h));
        v.push(("hidden1_bn".into(), h));
    }
    v
}
impl MaxCallNetworkSnapshot {
    pub fn validate(&self) -> Result<(), ComputeError> {
        if self.input_width < 2
            || self.input_width > 101
            || self.hidden_width == 0
            || self.hidden_width > 256
            || self.output_width == 0
            || self.output_width > 100
            || !self.normalization.epsilon.is_finite()
            || self.normalization.epsilon <= 0.
            || !self.normalization.new_weight.is_finite()
            || self.normalization.new_weight <= 0.
            || self.normalization.new_weight > 1.
            || (self.normalization.canonical_zero_translations && !self.normalization.hidden)
        {
            return Err(invalid("invalid network shape/BN"));
        }
        let l = layout(
            self.input_width,
            self.hidden_width,
            self.output_width,
            &self.normalization,
        );
        if l.len() != self.parameters.len()
            || self.parameters.iter().zip(l).any(|(p, (name, shape))| {
                p.name != name
                    || p.shape != shape
                    || p.values.len() != shape[0] * shape[1]
                    || p.values.iter().any(|x| !x.is_finite())
            })
        {
            return Err(invalid("invalid named max-call parameter leaves"));
        }
        let l = norm_layout(self.input_width, self.hidden_width, &self.normalization);
        if l.len() != self.running.len()
            || self.running.iter().zip(l).any(|(s, (name, n))| {
                s.name != name
                    || s.mean.len() != n
                    || s.variance.len() != n
                    || s.mean.iter().any(|x| !x.is_finite())
                    || s.variance.iter().any(|x| !x.is_finite() || *x < 0.)
            })
        {
            return Err(invalid("invalid BatchNorm running state"));
        }
        if self.normalization.canonical_zero_translations
            && self.parameters.iter().any(|p| {
                fixed_translation(&self.normalization, &p.name) && p.values.iter().any(|v| *v != 0.)
            })
        {
            return Err(invalid("canonical translation slots must be fixed zero"));
        }
        Ok(())
    }
    pub fn initialized(
        input: usize,
        hidden: usize,
        output: usize,
        bn: MaxCallBatchNorm,
        seed: u64,
    ) -> Result<Self, ComputeError> {
        let mut index = 0;
        let parameters = layout(input, hidden, output, &bn)
            .into_iter()
            .enumerate()
            .map(|(j, (name, shape))| {
                let n = shape[0] * shape[1];
                let values = if j < 6 && j % 2 == 0 {
                    let bound = (6. / (shape[0] + shape[1]) as f64).sqrt();
                    (0..n)
                        .map(|_| {
                            index += 1;
                            (uniform(seed, index) - 0.5) * 2. * bound
                        })
                        .collect()
                } else if name.ends_with("gamma") {
                    vec![1.; n]
                } else {
                    vec![0.; n]
                };
                MaxCallParameter {
                    name,
                    shape,
                    values,
                }
            })
            .collect();
        let running = norm_layout(input, hidden, &bn)
            .into_iter()
            .map(|(name, n)| MaxCallNormalizationState {
                name,
                mean: vec![0.; n],
                variance: vec![1.; n],
                batches: 0,
            })
            .collect();
        let s = Self {
            input_width: input,
            hidden_width: hidden,
            output_width: output,
            normalization: bn,
            parameters,
            running,
        };
        s.validate()?;
        Ok(s)
    }
    /// Deterministic scalar CPU-f64 audit path over the stored weights/stats.
    pub fn predict_features(&self, features: &[f64]) -> Result<Vec<f64>, ComputeError> {
        self.validate()?;
        if features.len() != self.input_width || features.iter().any(|x| !x.is_finite()) {
            return Err(invalid("invalid feature vector"));
        }
        let mut x = features.to_vec();
        let mut norm = 0;
        let mut affine = 6;
        let normalize = |x: &mut [f64], n: usize, p: usize| {
            let st = &self.running[n];
            for (j, v) in x.iter_mut().enumerate() {
                *v = (*v - st.mean[j]) / (st.variance[j] + self.normalization.epsilon).sqrt()
                    * self.parameters[p].values[j]
                    + self.parameters[p + 1].values[j];
            }
        };
        if self.normalization.input {
            normalize(&mut x, norm, affine);
            norm += 1;
            affine += 2;
        }
        for layer in 0..3 {
            let out = self.parameters[layer * 2].shape[1];
            let mut y = self.parameters[layer * 2 + 1].values.clone();
            for (i, a) in x.iter().enumerate() {
                for (j, b) in y.iter_mut().enumerate() {
                    *b += a * self.parameters[layer * 2].values[i * out + j];
                }
            }
            if layer < 2 {
                if self.normalization.hidden {
                    normalize(&mut y, norm, affine);
                    norm += 1;
                    affine += 2;
                }
                for y in &mut y {
                    *y = y.tanh();
                }
            }
            x = y;
        }
        if x.iter().any(|x| !x.is_finite()) {
            return Err(invalid("nonfinite scalar network output"));
        }
        Ok(x)
    }
}
struct RunningTensor {
    name: String,
    mean: Tensor<2>,
    variance: Tensor<2>,
    batches: u64,
}
impl Clone for RunningTensor {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            mean: self.mean.clone(),
            variance: self.variance.clone(),
            batches: self.batches,
        }
    }
}
struct Network {
    parameters: Vec<Tensor<2>>,
    running: Vec<RunningTensor>,
    shape: MaxCallNetworkSnapshot,
}
fn tensor(s: &Session, v: Vec<f64>, shape: [usize; 2]) -> Result<Tensor<2>, ComputeError> {
    if v.len() != shape[0] * shape[1]
        || v.iter().any(|x| {
            !x.is_finite() || (s.precision() == Precision::F32 && !(*x as f32).is_finite())
        })
    {
        return Err(invalid("tensor input is nonfinite or outside dtype"));
    }
    Ok(Tensor::from_data(
        TensorData::new(v, shape),
        TensorCreationOptions::new(s.device().clone()).with_dtype(
            if s.precision() == Precision::F64 {
                DType::F64
            } else {
                DType::F32
            },
        ),
    ))
}
fn read<const D: usize>(v: Tensor<D>) -> Result<Vec<f64>, ComputeError> {
    v.without_autodiff()
        .try_to_vec_as::<f64>()
        .map_err(execution)
}
impl Network {
    fn upload(
        s: &Session,
        snapshot: &MaxCallNetworkSnapshot,
        train: bool,
    ) -> Result<Self, ComputeError> {
        snapshot.validate()?;
        let parameters = snapshot
            .parameters
            .iter()
            .map(|p| {
                let t = tensor(s, p.values.clone(), p.shape)?;
                Ok(
                    if train && !fixed_translation(&snapshot.normalization, &p.name) {
                        t.autodiff().require_grad()
                    } else {
                        t
                    },
                )
            })
            .collect::<Result<Vec<_>, ComputeError>>()?;
        let running = snapshot
            .running
            .iter()
            .map(|r| {
                Ok(RunningTensor {
                    name: r.name.clone(),
                    mean: tensor(s, r.mean.clone(), [1, r.mean.len()])?,
                    variance: tensor(s, r.variance.clone(), [1, r.variance.len()])?,
                    batches: r.batches,
                })
            })
            .collect::<Result<Vec<_>, ComputeError>>()?;
        Ok(Self {
            parameters,
            running,
            shape: snapshot.clone(),
        })
    }
    fn forward(
        &self,
        mut x: Tensor<2>,
        training: bool,
        running: &mut [RunningTensor],
        arithmetic: ProductArithmetic,
        cap: usize,
    ) -> Result<Tensor<2>, ComputeError> {
        let b = &self.shape.normalization;
        let mut norm = 0;
        let mut affine = 6;
        let normalize = |x: Tensor<2>,
                         state: &mut RunningTensor,
                         offset: usize|
         -> Result<Tensor<2>, ComputeError> {
            let (mean, var) = if training {
                let mean = x.clone().mean_dim(0);
                let var = (x.clone() - mean.clone()).square().mean_dim(0);
                state.batches = state
                    .batches
                    .checked_add(1)
                    .ok_or_else(|| invalid("BN counter overflow"))?;
                state.mean = state.mean.clone() * (1. - b.new_weight)
                    + mean.clone().without_autodiff() * b.new_weight;
                state.variance = state.variance.clone() * (1. - b.new_weight)
                    + var.clone().without_autodiff() * b.new_weight;
                (mean, var)
            } else {
                (state.mean.clone(), state.variance.clone())
            };
            Ok(
                (x - mean) / (var + b.epsilon).sqrt() * self.parameters[offset].clone()
                    + self.parameters[offset + 1].clone(),
            )
        };
        if b.input {
            x = normalize(x, &mut running[norm], affine)?;
            norm += 1;
            affine += 2;
        }
        for layer in 0..3 {
            x = match arithmetic {
                ProductArithmetic::StrictNativeDtype => {
                    strict_matmul(x, self.parameters[layer * 2].clone(), cap)?
                }
                ProductArithmetic::BackendMatmul => x.matmul(self.parameters[layer * 2].clone()),
            } + self.parameters[layer * 2 + 1].clone();
            if layer < 2 {
                if b.hidden {
                    x = normalize(x, &mut running[norm], affine)?;
                    norm += 1;
                    affine += 2;
                }
                x = tanh(x);
            }
        }
        Ok(x)
    }
    fn snapshot(&self) -> Result<MaxCallNetworkSnapshot, ComputeError> {
        let mut s = self.shape.clone();
        for (p, t) in s.parameters.iter_mut().zip(&self.parameters) {
            p.values = read(t.clone())?;
        }
        for (r, t) in s.running.iter_mut().zip(&self.running) {
            r.mean = read(t.mean.clone())?;
            r.variance = read(t.variance.clone())?;
            r.batches = t.batches;
        }
        s.validate()?;
        Ok(s)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepTimingMode {
    EndToEnd,
    SynchronizedStages,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaxCallStepTiming {
    pub upload_seconds: f64,
    pub forward_seconds: f64,
    pub backward_seconds: f64,
    pub optimizer_seconds: f64,
    pub readback_seconds: f64,
    pub synchronization_seconds: f64,
    pub total_seconds: f64,
    pub synchronized_stages: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCallStepResult {
    pub completed_updates: u64,
    pub loss_before_update: f64,
    pub gradient_l2_norm: f64,
    pub timing: MaxCallStepTiming,
}
struct ResidentDataset {
    x: Tensor<2>,
    y: Tensor<2>,
    rows: usize,
}
/// Owns one date's parameters, fresh Adam state and at most one resident split each.
pub struct MaxCallDateLearner {
    config: MaxCallTrainingConfig,
    date: usize,
    session: Session,
    network: Network,
    first: Vec<Tensor<2>>,
    second: Vec<Tensor<2>>,
    updates: u64,
    last_gradients: Option<Vec<Tensor<2>>>,
    training: Option<ResidentDataset>,
    validation: Option<ResidentDataset>,
}
fn payoff(m: &MaxCallMarket, date: usize, state: &[f64]) -> Result<f64, ComputeError> {
    if date >= m.times.len()
        || state.len() != m.spots.len()
        || state.iter().any(|s| !s.is_finite() || *s <= 0.)
    {
        return Err(invalid("invalid max-call date/state"));
    }
    let y = (state.iter().copied().fold(0., f64::max) - m.strike).max(0.)
        * (-m.rate * m.times[date]).exp();
    if !y.is_finite() {
        return Err(invalid("nonfinite discounted payoff"));
    }
    Ok(y)
}
fn feature_rows(m: &MaxCallMarket, date: usize, states: &[f64]) -> Result<Vec<f64>, ComputeError> {
    let d = m.spots.len();
    if states.is_empty() || !states.len().is_multiple_of(d) {
        return Err(invalid("state array shape differs from asset count"));
    }
    let mut v = Vec::with_capacity(states.len() + states.len() / d);
    for state in states.chunks(d) {
        v.extend_from_slice(state);
        v.push(payoff(m, date, state)?);
    }
    Ok(v)
}
impl MaxCallDateLearner {
    pub fn new(
        c: &MaxCallTrainingConfig,
        date: usize,
        warm: Option<&MaxCallNetworkSnapshot>,
    ) -> Result<Self, ComputeError> {
        Self::new_output(c, date, 1, warm)
    }
    pub fn new_output(
        c: &MaxCallTrainingConfig,
        date: usize,
        output_width: usize,
        warm: Option<&MaxCallNetworkSnapshot>,
    ) -> Result<Self, ComputeError> {
        let base_plan = c.validate()?;
        if output_width > 1 {
            let scalar = if c.precision == Precision::F64 {
                8u128
            } else {
                4
            };
            let extra_dataset = (c.training_paths + c.validation_paths) as u128
                * (output_width - 1) as u128
                * scalar;
            let extra_parameters =
                (c.hidden_width + 1) as u128 * (output_width - 1) as u128 * scalar * 12;
            if base_plan.modeled_compute_bytes as u128 + extra_dataset + extra_parameters
                > c.compute_budget_bytes as u128
                || base_plan.modeled_host_bytes as u128 + extra_dataset * 2
                    > c.host_budget_bytes as u128
            {
                return Err(ComputeError::BudgetExceeded(
                    "vector-output workspace exceeds allowances".into(),
                ));
            }
        }
        if date >= c.market.times.len() || output_width == 0 || output_width > c.market.spots.len()
        {
            return Err(invalid("invalid date/output width"));
        }
        let snapshot = match warm {
            Some(s) => {
                if c.normalization.canonical_zero_translations
                    && !s.normalization.canonical_zero_translations
                {
                    s.canonicalized_translations()?
                } else {
                    s.clone()
                }
            }
            None => MaxCallNetworkSnapshot::initialized(
                c.market.spots.len() + 1,
                c.hidden_width,
                output_width,
                c.normalization.clone(),
                c.initialization_seed.wrapping_add(date as u64),
            )?,
        };
        snapshot.validate()?;
        if snapshot.input_width != c.market.spots.len() + 1
            || snapshot.hidden_width != c.hidden_width
            || snapshot.output_width != output_width
            || snapshot.normalization != c.normalization
        {
            return Err(invalid("warm-start network/config mismatch"));
        }
        let session = Session::new(c.target, c.precision, c.batch_rows)?;
        let network = Network::upload(&session, &snapshot, true)?;
        let first = network
            .parameters
            .iter()
            .map(|p| p.clone().without_autodiff() * 0.)
            .collect::<Vec<_>>();
        let second = first.clone();
        Ok(Self {
            config: c.clone(),
            date,
            session,
            network,
            first,
            second,
            updates: 0,
            last_gradients: None,
            training: None,
            validation: None,
        })
    }
    fn dataset(
        &self,
        states: &[f64],
        targets: &[f64],
        limit: usize,
    ) -> Result<ResidentDataset, ComputeError> {
        let rows = states.len() / self.config.market.spots.len();
        if rows < 2 || rows > limit || targets.len() != rows * self.network.shape.output_width {
            return Err(invalid("resident split rows/target shape invalid"));
        }
        let features = feature_rows(&self.config.market, self.date, states)?;
        Ok(ResidentDataset {
            x: tensor(
                &self.session,
                features,
                [rows, self.network.shape.input_width],
            )?,
            y: tensor(
                &self.session,
                targets.to_vec(),
                [rows, self.network.shape.output_width],
            )?,
            rows,
        })
    }
    pub fn upload_training(
        &mut self,
        states: &[f64],
        targets: &[f64],
    ) -> Result<f64, ComputeError> {
        if self.training.is_some() {
            return Err(invalid("training split already resident"));
        }
        let t = Instant::now();
        let data = self.dataset(states, targets, self.config.training_paths)?;
        self.session.synchronize()?;
        self.training = Some(data);
        Ok(t.elapsed().as_secs_f64())
    }
    pub fn upload_validation(
        &mut self,
        states: &[f64],
        targets: &[f64],
    ) -> Result<f64, ComputeError> {
        if self.validation.is_some() {
            return Err(invalid("validation split already resident"));
        }
        let t = Instant::now();
        let data = self.dataset(states, targets, self.config.validation_paths)?;
        self.session.synchronize()?;
        self.validation = Some(data);
        Ok(t.elapsed().as_secs_f64())
    }
    pub fn snapshot(&self) -> Result<MaxCallNetworkSnapshot, ComputeError> {
        self.network.snapshot()
    }
    pub fn completed_updates(&self) -> u64 {
        self.updates
    }
    pub fn update_rows(
        &mut self,
        indices: &[usize],
        learning_rate: f64,
        mode: StepTimingMode,
    ) -> Result<MaxCallStepResult, ComputeError> {
        let total = Instant::now();
        if indices.len() < 2
            || indices.len() > self.config.batch_rows
            || !learning_rate.is_finite()
            || learning_rate <= 0.
            || learning_rate > 1.
            || self.updates == u64::MAX
        {
            return Err(invalid("invalid update rows/rate/counter"));
        }
        let data = self
            .training
            .as_ref()
            .ok_or_else(|| invalid("upload training split first"))?;
        if indices.iter().any(|i| *i >= data.rows) {
            return Err(invalid("row index outside resident split"));
        }
        let mut timing = MaxCallStepTiming {
            synchronized_stages: mode == StepTimingMode::SynchronizedStages,
            ..Default::default()
        };
        let sync = |s: &Session, timing: &mut MaxCallStepTiming| -> Result<(), ComputeError> {
            if timing.synchronized_stages {
                let t = Instant::now();
                s.synchronize()?;
                timing.synchronization_seconds += t.elapsed().as_secs_f64();
            }
            Ok(())
        };
        let t = Instant::now();
        let index = Tensor::<1, Int>::from_data(
            TensorData::new(
                indices.iter().map(|i| *i as i32).collect::<Vec<_>>(),
                [indices.len()],
            ),
            TensorCreationOptions::new(self.session.device().clone()).with_dtype(DType::I32),
        );
        let x = data.x.clone().select(0, index.clone());
        let y = data.y.clone().select(0, index);
        sync(&self.session, &mut timing)?;
        timing.upload_seconds = t.elapsed().as_secs_f64();
        let mut running = self.network.running.clone();
        let t = Instant::now();
        let output = self.network.forward(
            x,
            true,
            &mut running,
            self.config.arithmetic,
            self.config.max_expanded_elements,
        )?;
        let loss = (output - y).square().mean();
        sync(&self.session, &mut timing)?;
        timing.forward_seconds = t.elapsed().as_secs_f64();
        let loss_value = loss.clone().without_autodiff();
        let t = Instant::now();
        let mut gradient = loss.backward();
        let gradients = self
            .network
            .parameters
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if fixed_translation(
                    &self.network.shape.normalization,
                    &self.network.shape.parameters[i].name,
                ) {
                    Ok(p.clone().without_autodiff() * 0.)
                } else {
                    p.grad_remove(&mut gradient)
                        .map(|g| g.without_autodiff())
                        .ok_or_else(|| invalid("missing continuation gradient"))
                }
            })
            .collect::<Result<Vec<_>, ComputeError>>()?;
        sync(&self.session, &mut timing)?;
        timing.backward_seconds = t.elapsed().as_secs_f64();
        let norm = gradients
            .iter()
            .map(|g| g.clone().square().sum())
            .reduce(|a, b| a + b)
            .ok_or_else(|| invalid("no gradients"))?
            .sqrt();
        let t = Instant::now();
        let step = (self.updates + 1) as f64;
        let a = &self.config;
        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut parameters = Vec::new();
        for (i, g) in gradients.iter().enumerate() {
            let g = g.clone();
            let m = self.first[i].clone() * a.beta1 + g.clone() * (1. - a.beta1);
            let v = self.second[i].clone() * a.beta2 + g.square() * (1. - a.beta2);
            let delta = match a.epsilon_convention {
                AdamEpsilonConvention::BiasCorrected => {
                    (m.clone() / (1. - a.beta1.powf(step)))
                        / ((v.clone() / (1. - a.beta2.powf(step))).sqrt() + a.epsilon)
                        * learning_rate
                }
                AdamEpsilonConvention::TensorflowUncorrected => {
                    m.clone() / (v.clone().sqrt() + a.epsilon)
                        * (learning_rate * (1. - a.beta2.powf(step)).sqrt()
                            / (1. - a.beta1.powf(step)))
                }
            };
            let value = self.network.parameters[i].clone().without_autodiff() - delta;
            parameters.push(
                if fixed_translation(
                    &self.network.shape.normalization,
                    &self.network.shape.parameters[i].name,
                ) {
                    value
                } else {
                    value.autodiff().require_grad()
                },
            );
            first.push(m);
            second.push(v);
        }
        sync(&self.session, &mut timing)?;
        timing.optimizer_seconds = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let finite = parameters
            .iter()
            .chain(&first)
            .chain(&second)
            .map(|p| p.clone().without_autodiff().abs().sum())
            .chain(
                running
                    .iter()
                    .flat_map(|s| [s.mean.clone().abs().sum(), s.variance.clone().abs().sum()]),
            )
            .reduce(|a, b| a + b)
            .ok_or_else(|| invalid("empty proposed state"))?;
        let checks = read(Tensor::cat(vec![loss_value, norm, finite], 0))?;
        if checks.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "nonfinite loss/gradient/optimizer/BN state; update not committed",
            ));
        }
        let scalar = checks[0];
        let gradient_norm = checks[1];
        timing.readback_seconds += t.elapsed().as_secs_f64();
        self.network.parameters = parameters;
        self.network.running = running;
        self.first = first;
        self.second = second;
        self.last_gradients = Some(gradients);
        self.updates += 1;
        timing.total_seconds = total.elapsed().as_secs_f64();
        Ok(MaxCallStepResult {
            completed_updates: self.updates,
            loss_before_update: scalar,
            gradient_l2_norm: gradient_norm,
            timing,
        })
    }
    fn predictions(&self, data: &ResidentDataset) -> Result<Vec<f64>, ComputeError> {
        let mut out = Vec::with_capacity(data.rows * self.network.shape.output_width);
        for first in (0..data.rows).step_by(self.config.batch_rows) {
            let end = (first + self.config.batch_rows).min(data.rows);
            let x = data
                .x
                .clone()
                .slice([first..end, 0..self.network.shape.input_width]);
            let mut running = self.network.running.clone();
            let net = Network {
                parameters: self
                    .network
                    .parameters
                    .iter()
                    .map(|p| p.clone().without_autodiff())
                    .collect(),
                running: Vec::new(),
                shape: self.network.shape.clone(),
            };
            out.extend(read(net.forward(
                x,
                false,
                &mut running,
                self.config.arithmetic,
                self.config.max_expanded_elements,
            )?)?);
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(invalid("nonfinite continuation predictions"));
        }
        Ok(out)
    }
    pub fn predict_training(&self) -> Result<Vec<f64>, ComputeError> {
        self.predictions(
            self.training
                .as_ref()
                .ok_or_else(|| invalid("no training split"))?,
        )
    }
    pub fn predict_validation(&self) -> Result<Vec<f64>, ComputeError> {
        self.predictions(
            self.validation
                .as_ref()
                .ok_or_else(|| invalid("no validation split"))?,
        )
    }
    pub fn validation_mse(&self) -> Result<f64, ComputeError> {
        let data = self
            .validation
            .as_ref()
            .ok_or_else(|| invalid("no validation split"))?;
        let mut sum = 0.;
        for first in (0..data.rows).step_by(self.config.batch_rows) {
            let end = (first + self.config.batch_rows).min(data.rows);
            let x = data
                .x
                .clone()
                .slice([first..end, 0..self.network.shape.input_width]);
            let y = data
                .y
                .clone()
                .slice([first..end, 0..self.network.shape.output_width]);
            let mut running = self.network.running.clone();
            let net = Network {
                parameters: self
                    .network
                    .parameters
                    .iter()
                    .map(|p| p.clone().without_autodiff())
                    .collect(),
                running: Vec::new(),
                shape: self.network.shape.clone(),
            };
            let loss = (net.forward(
                x,
                false,
                &mut running,
                self.config.arithmetic,
                self.config.max_expanded_elements,
            )? - y)
                .square()
                .sum();
            sum += read(loss)?[0];
        }
        let mse = sum / (data.rows * self.network.shape.output_width) as f64;
        if !mse.is_finite() {
            return Err(invalid("nonfinite validation loss"));
        }
        Ok(mse)
    }
}
fn uniform(seed: u64, index: u64) -> f64 {
    let mut z = seed.wrapping_add(index.wrapping_mul(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    ((z >> 12) as f64 + 0.5) / 4503599627370496.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxCallCheckpoint {
    pub schema: String,
    pub config: MaxCallTrainingConfig,
    pub dates: Vec<Option<MaxCallNetworkSnapshot>>,
    pub initial_training_continuation: f64,
    pub stop_initial: bool,
    pub basis: String,
    pub training_market_sha256: String,
    pub validation_market_sha256: String,
}
impl MaxCallCheckpoint {
    pub fn market(&self) -> &MaxCallMarket {
        &self.config.market
    }
    pub fn validate(&self) -> Result<(), ComputeError> {
        self.config.validate()?;
        let m = &self.config.market;
        let n = m.times.len();
        if self.schema != SCHEMA
            || self.dates.len() != n
            || !self.initial_training_continuation.is_finite()
            || self.initial_training_continuation < 0.
            || self.stop_initial && !m.exercise[0]
        {
            return Err(invalid(
                "invalid max-call checkpoint schema/value/initial decision",
            ));
        }
        for (date, network) in self.dates.iter().enumerate() {
            if date == 0 || date + 1 == n || !m.exercise[date] {
                if network.is_some() {
                    return Err(invalid(
                        "unexpected date model outside trained exercise dates",
                    ));
                }
            } else {
                let network = network
                    .as_ref()
                    .ok_or_else(|| invalid("missing date continuation"))?;
                network.validate()?;
                if network.input_width != m.spots.len() + 1
                    || network.hidden_width != self.config.hidden_width
                    || network.output_width != 1
                    || network.normalization != self.config.normalization
                {
                    return Err(invalid("date architecture/config differs"));
                }
            }
        }
        Ok(())
    }
    pub fn should_stop(&self, date: usize, state: &[f64]) -> Result<bool, ComputeError> {
        self.validate()?;
        let g = payoff(&self.config.market, date, state)?;
        if !self.config.market.exercise[date] {
            return Ok(false);
        }
        if date == 0 {
            return Ok(self.stop_initial);
        }
        if date + 1 == self.dates.len() {
            return Ok(true);
        }
        let features = feature_rows(&self.config.market, date, state)?;
        Ok(g >= self.dates[date]
            .as_ref()
            .ok_or_else(|| invalid("missing continuation"))?
            .predict_features(&features)?[0])
    }
}
/// Resident weights/stats for every selected date; requested device is explicit.
/// Every inference call uses stored statistics and is independent of batch peers.
pub struct MaxCallPredictor {
    checkpoint: MaxCallCheckpoint,
    session: Session,
    networks: Vec<Option<Network>>,
    batch_rows: usize,
}
impl MaxCallPredictor {
    pub fn new(
        checkpoint: &MaxCallCheckpoint,
        target: ExecutionTarget,
        precision: Precision,
        batch_rows: usize,
        max_working_bytes: u64,
    ) -> Result<Self, ComputeError> {
        checkpoint.validate()?;
        if batch_rows == 0 || batch_rows > 65536 {
            return Err(invalid("invalid inference batch cap"));
        }
        let d = checkpoint.config.market.spots.len();
        let h = checkpoint.config.hidden_width;
        let scalar = if precision == Precision::F64 {
            8u128
        } else {
            4
        };
        let params = checkpoint
            .dates
            .iter()
            .flatten()
            .map(|n| n.parameters.iter().map(|p| p.values.len()).sum::<usize>())
            .sum::<usize>();
        let products = batch_rows as u128 * ((d + 1) * h + h * h + h) as u128;
        let tape = if target == ExecutionTarget::Cpu
            || checkpoint.config.arithmetic == ProductArithmetic::BackendMatmul
        {
            batch_rows as u128 * (d + 1 + 8 * h + 8) as u128 * scalar * 8
        } else {
            products * scalar * 3
        };
        let estimate = params as u128 * scalar * 3 + tape + 32 * 1024 * 1024;
        if estimate > max_working_bytes as u128 {
            return Err(ComputeError::BudgetExceeded(format!(
                "inference modeled workspace {estimate} exceeds {max_working_bytes}"
            )));
        }
        if checkpoint.config.arithmetic == ProductArithmetic::StrictNativeDtype
            && batch_rows
                .checked_mul((d + 1).max(h))
                .and_then(|n| n.checked_mul(h))
                .is_none_or(|n| n > checkpoint.config.max_expanded_elements)
        {
            return Err(ComputeError::BudgetExceeded(
                "inference strict product cap exceeded".into(),
            ));
        }
        let session = Session::new(target, precision, batch_rows)?;
        let networks = checkpoint
            .dates
            .iter()
            .map(|s| {
                s.as_ref()
                    .map(|s| Network::upload(&session, s, false))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        session.synchronize()?;
        Ok(Self {
            checkpoint: checkpoint.clone(),
            session,
            networks,
            batch_rows,
        })
    }
    pub fn predict_batch(&mut self, date: usize, states: &[f64]) -> Result<Vec<f64>, ComputeError> {
        let m = &self.checkpoint.config.market;
        let d = m.spots.len();
        if states.is_empty() || !states.len().is_multiple_of(d) || date >= m.times.len() {
            return Err(invalid("invalid batch state/date"));
        }
        if date == 0 {
            return Ok(vec![
                self.checkpoint.initial_training_continuation;
                states.len() / d
            ]);
        }
        if date + 1 == m.times.len() {
            return states.chunks(d).map(|s| payoff(m, date, s)).collect();
        }
        let network = self.networks[date]
            .as_ref()
            .ok_or_else(|| invalid("no continuation at excluded date"))?;
        let mut predictions = Vec::with_capacity(states.len() / d);
        for chunk in states.chunks(self.batch_rows * d) {
            let rows = chunk.len() / d;
            let x = tensor(&self.session, feature_rows(m, date, chunk)?, [rows, d + 1])?;
            let mut stats = network.running.clone();
            let output = network.forward(
                x,
                false,
                &mut stats,
                self.checkpoint.config.arithmetic,
                self.checkpoint.config.max_expanded_elements,
            )?;
            predictions.extend(read(output)?);
        }
        if predictions.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite batched continuation"));
        }
        Ok(predictions)
    }
    pub fn should_stop_batch(
        &mut self,
        date: usize,
        states: &[f64],
    ) -> Result<Vec<bool>, ComputeError> {
        let m = &self.checkpoint.config.market;
        let d = m.spots.len();
        if states.is_empty() || !states.len().is_multiple_of(d) || date >= m.times.len() {
            return Err(invalid("invalid stopping batch"));
        }
        for state in states.chunks(d) {
            payoff(m, date, state)?;
        }
        if !m.exercise[date] {
            return Ok(vec![false; states.len() / d]);
        }
        if date == 0 {
            return Ok(vec![self.checkpoint.stop_initial; states.len() / d]);
        }
        if date + 1 == m.times.len() {
            return Ok(vec![true; states.len() / d]);
        }
        let market = self.checkpoint.config.market.clone();
        let c = self.predict_batch(date, states)?;
        states
            .chunks(d)
            .zip(c)
            .map(|(s, c)| Ok(payoff(&market, date, s)? >= c))
            .collect()
    }
}
impl crate::multi_asset_market::MaxCallStoppingPolicy for MaxCallPredictor {
    fn should_stop(&mut self, date: usize, state: &[f64]) -> Result<bool, ComputeError> {
        Ok(self.should_stop_batch(date, state)?[0])
    }
    fn should_stop_batch(
        &mut self,
        date: usize,
        states: &[f64],
        assets: usize,
    ) -> Result<Vec<bool>, ComputeError> {
        if assets != self.checkpoint.config.market.spots.len() {
            return Err(invalid("policy/market asset count differs"));
        }
        MaxCallPredictor::should_stop_batch(self, date, states)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCallValidationRecord {
    pub completed_updates: u64,
    pub regression_mse: f64,
    pub elapsed_seconds: f64,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaxCallDateTiming {
    pub packing_seconds: f64,
    pub training_upload_seconds: f64,
    pub validation_upload_seconds: f64,
    pub updates_seconds: f64,
    pub validation_seconds: f64,
    pub checkpoint_download_seconds: f64,
    pub total_seconds: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCallDateReport {
    pub date: usize,
    pub completed_updates: usize,
    pub warm_start_from: Option<usize>,
    pub final_validation_mse: f64,
    pub diagnostic_best_mse: f64,
    pub diagnostic_best_update: u64,
    pub last_loss: f64,
    pub maximum_gradient_l2_norm: f64,
    pub validation_trace: Vec<MaxCallValidationRecord>,
    pub timing: MaxCallDateTiming,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxCallPartialCheckpoint {
    pub schema: String,
    pub config: MaxCallTrainingConfig,
    pub next_date: usize,
    pub dates: Vec<Option<MaxCallNetworkSnapshot>>,
    pub diagnostic_dates: Vec<Option<MaxCallNetworkSnapshot>>,
    pub reports: Vec<MaxCallDateReport>,
    pub training_market_sha256: String,
    pub validation_market_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCallTrainingProgress {
    pub date: usize,
    pub completed_updates: u64,
    pub date_update_budget: usize,
    pub validation_mse: f64,
    pub date_complete: bool,
    pub elapsed_seconds: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCallTrainingResult {
    pub final_checkpoint: MaxCallCheckpoint,
    pub diagnostic_validation_checkpoint: MaxCallCheckpoint,
    pub dates: Vec<MaxCallDateReport>,
    pub market_generation_seconds: f64,
    pub elapsed_seconds: f64,
    pub sampling_contract: String,
    pub memory_plan: MaxCallMemoryPlan,
}
fn states_at(values: &[f64], rows: usize, dates: usize, d: usize, date: usize) -> Vec<f64> {
    (0..rows)
        .flat_map(|p| {
            values[(p * dates + date) * d..(p * dates + date + 1) * d]
                .iter()
                .copied()
        })
        .collect()
}
fn stopped_targets(
    config: &MaxCallTrainingConfig,
    dates: &[Option<MaxCallNetworkSnapshot>],
    values: &[f64],
    rows: usize,
    start_date: usize,
) -> Result<Vec<f64>, ComputeError> {
    let m = &config.market;
    let n = m.times.len();
    let d = m.spots.len();
    let mut y = (0..rows)
        .map(|p| payoff(m, n - 1, &values[(p * n + n - 1) * d..(p * n + n) * d]))
        .collect::<Result<Vec<_>, _>>()?;
    for date in (start_date.max(1)..n - 1).rev() {
        if !m.exercise[date] {
            continue;
        }
        let snapshot = dates[date]
            .as_ref()
            .ok_or_else(|| invalid("partial checkpoint missing future date"))?;
        let session = Session::new(config.target, config.precision, config.batch_rows)?;
        let network = Network::upload(&session, snapshot, false)?;
        for first in (0..rows).step_by(config.batch_rows) {
            let end = (first + config.batch_rows).min(rows);
            let states = (first..end)
                .flat_map(|p| {
                    values[(p * n + date) * d..(p * n + date + 1) * d]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            let x = tensor(
                &session,
                feature_rows(m, date, &states)?,
                [end - first, d + 1],
            )?;
            let mut running = network.running.clone();
            let predictions = read(network.forward(
                x,
                false,
                &mut running,
                config.arithmetic,
                config.max_expanded_elements,
            )?)?;
            for (i, (state, c)) in states.chunks(d).zip(predictions).enumerate() {
                let g = payoff(m, date, state)?;
                if g >= c {
                    y[first + i] = g;
                }
            }
        }
    }
    Ok(y)
}
/// Original paper's scheduled FINAL update is primary. Diagnostic per-date best
/// snapshots are saved separately against labels from the primary future policy.
pub fn train(config: &MaxCallTrainingConfig) -> Result<MaxCallTrainingResult, ComputeError> {
    train_with_progress(config, None, |_, _| Ok(()))
}
/// Resume only at completed-date archives: paths/labels are regenerated exactly,
/// weights and BN warm-start, and Adam resets for the next date.
pub fn train_with_progress<F>(
    config: &MaxCallTrainingConfig,
    resume: Option<&MaxCallPartialCheckpoint>,
    mut progress: F,
) -> Result<MaxCallTrainingResult, ComputeError>
where
    F: FnMut(
        &MaxCallTrainingProgress,
        Option<&MaxCallPartialCheckpoint>,
    ) -> Result<(), ComputeError>,
{
    let started = Instant::now();
    let plan = config.validate()?;
    let m = &config.market;
    let n = m.times.len();
    let d = m.spots.len();
    if let Some(p) = resume {
        p.validate()?;
        if serde_json::to_value(&p.config).map_err(execution)?
            != serde_json::to_value(config).map_err(execution)?
        {
            return Err(invalid("partial checkpoint/config differs"));
        }
    }
    let t = Instant::now();
    let training = generate_paths(
        m,
        config.training_paths,
        config.training_seed,
        0,
        10,
        config.host_budget_bytes,
    )?;
    let validation = generate_paths(
        m,
        config.validation_paths,
        config.validation_seed,
        0,
        11,
        config.host_budget_bytes,
    )?;
    let generation = t.elapsed().as_secs_f64();
    if let Some(p) = resume
        && (p.training_market_sha256 != training.content_sha256
            || p.validation_market_sha256 != validation.content_sha256)
    {
        return Err(invalid("regenerated partial market hashes differ"));
    }
    let mut dates = resume
        .map(|p| p.dates.clone())
        .unwrap_or_else(|| vec![None; n]);
    let mut diagnostic = resume
        .map(|p| p.diagnostic_dates.clone())
        .unwrap_or_else(|| vec![None; n]);
    let mut reports = resume.map(|p| p.reports.clone()).unwrap_or_default();
    let next = resume.map(|p| p.next_date).unwrap_or(n - 2);
    let mut y = stopped_targets(config, &dates, &training.values, training.paths, next + 1)?;
    let mut vy = stopped_targets(
        config,
        &dates,
        &validation.values,
        validation.paths,
        next + 1,
    )?;
    for date in (1..=next).rev() {
        if !m.exercise[date] {
            continue;
        }
        let date_start = Instant::now();
        let warm = if config.warm_start {
            ((date + 1)..n - 1).find(|j| dates[*j].is_some())
        } else {
            None
        };
        let mut learner =
            MaxCallDateLearner::new(config, date, warm.and_then(|j| dates[j].as_ref()))?;
        let mut timing = MaxCallDateTiming::default();
        let t = Instant::now();
        let x = states_at(&training.values, training.paths, n, d, date);
        let vx = states_at(&validation.values, validation.paths, n, d, date);
        timing.packing_seconds = t.elapsed().as_secs_f64();
        timing.training_upload_seconds = learner.upload_training(&x, &y)?;
        timing.validation_upload_seconds = learner.upload_validation(&vx, &vy)?;
        let mut trace = Vec::new();
        let mut best = f64::INFINITY;
        let mut best_step = 0;
        let mut best_network = learner.snapshot()?;
        let mut completed = 0;
        let mut last_loss = 0.;
        let mut gradient_max = 0f64;
        let total = config
            .schedule(date)
            .iter()
            .map(|s| s.updates)
            .sum::<usize>();
        let initial = learner.validation_mse()?;
        trace.push(MaxCallValidationRecord {
            completed_updates: 0,
            regression_mse: initial,
            elapsed_seconds: date_start.elapsed().as_secs_f64(),
        });
        if initial < best {
            best = initial;
        }
        for stage in config.schedule(date) {
            for _ in 0..stage.updates {
                let indices = (0..config.batch_rows)
                    .map(|row| {
                        let key = config.training_seed
                            ^ ((date as u64) << 40)
                            ^ ((completed as u64) << 16);
                        let u = uniform(key, row as u64);
                        (u * config.training_paths as f64) as usize
                    })
                    .collect::<Vec<_>>();
                let t = Instant::now();
                let result =
                    learner.update_rows(&indices, stage.learning_rate, StepTimingMode::EndToEnd)?;
                timing.updates_seconds += t.elapsed().as_secs_f64();
                completed += 1;
                last_loss = result.loss_before_update;
                gradient_max = gradient_max.max(result.gradient_l2_norm);
                if completed % config.validation_every == 0 || completed == total {
                    let t = Instant::now();
                    let loss = learner.validation_mse()?;
                    timing.validation_seconds += t.elapsed().as_secs_f64();
                    trace.push(MaxCallValidationRecord {
                        completed_updates: completed as u64,
                        regression_mse: loss,
                        elapsed_seconds: date_start.elapsed().as_secs_f64(),
                    });
                    if loss < best {
                        let t = Instant::now();
                        best_network = learner.snapshot()?;
                        timing.checkpoint_download_seconds += t.elapsed().as_secs_f64();
                        best = loss;
                        best_step = completed as u64;
                    }
                    progress(
                        &MaxCallTrainingProgress {
                            date,
                            completed_updates: completed as u64,
                            date_update_budget: total,
                            validation_mse: loss,
                            date_complete: false,
                            elapsed_seconds: started.elapsed().as_secs_f64(),
                        },
                        None,
                    )?;
                }
            }
        }
        let t = Instant::now();
        let final_network = learner.snapshot()?;
        timing.checkpoint_download_seconds += t.elapsed().as_secs_f64();
        let final_mse = trace
            .last()
            .ok_or_else(|| invalid("empty validation trace"))?
            .regression_mse;
        let pred = learner.predict_training()?;
        let vpred = learner.predict_validation()?;
        for (p, (state, c)) in x.chunks(d).zip(pred).enumerate() {
            let g = payoff(m, date, state)?;
            if g >= c {
                y[p] = g;
            }
        }
        for (p, (state, c)) in vx.chunks(d).zip(vpred).enumerate() {
            let g = payoff(m, date, state)?;
            if g >= c {
                vy[p] = g;
            }
        }
        dates[date] = Some(final_network);
        diagnostic[date] = Some(best_network);
        timing.total_seconds = date_start.elapsed().as_secs_f64();
        reports.push(MaxCallDateReport {
            date,
            completed_updates: completed,
            warm_start_from: warm,
            final_validation_mse: final_mse,
            diagnostic_best_mse: best,
            diagnostic_best_update: best_step,
            last_loss,
            maximum_gradient_l2_norm: gradient_max,
            validation_trace: trace,
            timing,
        });
        let partial = MaxCallPartialCheckpoint {
            schema: "volt.compute.maxcall_partial.v1".into(),
            config: config.clone(),
            next_date: date - 1,
            dates: dates.clone(),
            diagnostic_dates: diagnostic.clone(),
            reports: reports.clone(),
            training_market_sha256: training.content_sha256.clone(),
            validation_market_sha256: validation.content_sha256.clone(),
        };
        progress(
            &MaxCallTrainingProgress {
                date,
                completed_updates: completed as u64,
                date_update_budget: total,
                validation_mse: final_mse,
                date_complete: true,
                elapsed_seconds: started.elapsed().as_secs_f64(),
            },
            Some(&partial),
        )?;
    }
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let primary = MaxCallCheckpoint {
        schema: SCHEMA.into(),
        config: config.clone(),
        dates,
        initial_training_continuation: mean,
        stop_initial: m.exercise[0] && payoff(m, 0, &m.spots)? >= mean,
        basis: "final_scheduled_update_each_date".into(),
        training_market_sha256: training.content_sha256.clone(),
        validation_market_sha256: validation.content_sha256.clone(),
    };
    primary.validate()?;
    let dy = stopped_targets(config, &diagnostic, &training.values, training.paths, 1)?;
    let mean = dy.iter().sum::<f64>() / dy.len() as f64;
    let alternative = MaxCallCheckpoint {
        schema: SCHEMA.into(),
        config: config.clone(),
        dates: diagnostic,
        initial_training_continuation: mean,
        stop_initial: m.exercise[0] && payoff(m, 0, &m.spots)? >= mean,
        basis: "diagnostic_per_date_validation_MSE_against_final_future_targets".into(),
        training_market_sha256: training.content_sha256,
        validation_market_sha256: validation.content_sha256,
    };
    alternative.validate()?;
    Ok(MaxCallTrainingResult{final_checkpoint:primary,diagnostic_validation_checkpoint:alternative,dates:reports,market_generation_seconds:generation,elapsed_seconds:started.elapsed().as_secs_f64(),sampling_contract:"Fixed disjoint market pools; per-update deterministic uniform row sampling WITH replacement from resident training split; raw[d stocks,discounted payoff], unscaled targets; final scheduled date snapshots primary; weights+BN warmstart, Adam resets/date. Diagnostic best never feeds primary labels.".into(),memory_plan:plan})
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCallGradientAnalysis {
    pub training_prediction: Vec<f64>,
    pub loss: f64,
    pub gradients: Vec<MaxCallParameter>,
    pub proposed_running: Vec<MaxCallNormalizationState>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCallOptimizerSnapshot {
    pub completed_updates: u64,
    pub first: Vec<MaxCallParameter>,
    pub second: Vec<MaxCallParameter>,
}
impl MaxCallDateLearner {
    /// Technical tensor interface for independent fixtures/benchmarks. The caller
    /// supplies raw feature semantics; application training derives features itself.
    pub fn upload_training_features(
        &mut self,
        features: &[f64],
        targets: &[f64],
    ) -> Result<f64, ComputeError> {
        if self.training.is_some() {
            return Err(invalid("training split already resident"));
        }
        let f = self.network.shape.input_width;
        let rows = features.len() / f;
        if rows < 2
            || rows > self.config.training_paths
            || features.len() != rows * f
            || targets.len() != rows * self.network.shape.output_width
        {
            return Err(invalid("feature/target split shape invalid"));
        }
        let t = Instant::now();
        let data = ResidentDataset {
            x: tensor(&self.session, features.to_vec(), [rows, f])?,
            y: tensor(
                &self.session,
                targets.to_vec(),
                [rows, self.network.shape.output_width],
            )?,
            rows,
        };
        self.session.synchronize()?;
        self.training = Some(data);
        Ok(t.elapsed().as_secs_f64())
    }
    pub fn analyze_rows(&self, indices: &[usize]) -> Result<MaxCallGradientAnalysis, ComputeError> {
        let data = self
            .training
            .as_ref()
            .ok_or_else(|| invalid("no resident training split"))?;
        if indices.len() < 2
            || indices.len() > self.config.batch_rows
            || indices.iter().any(|i| *i >= data.rows)
        {
            return Err(invalid("invalid analysis indices"));
        }
        let index = Tensor::<1, Int>::from_data(
            TensorData::new(
                indices.iter().map(|i| *i as i32).collect::<Vec<_>>(),
                [indices.len()],
            ),
            TensorCreationOptions::new(self.session.device().clone()).with_dtype(DType::I32),
        );
        let x = data.x.clone().select(0, index.clone());
        let y = data.y.clone().select(0, index);
        let mut running = self.network.running.clone();
        let prediction = self.network.forward(
            x,
            true,
            &mut running,
            self.config.arithmetic,
            self.config.max_expanded_elements,
        )?;
        let loss = (prediction.clone() - y).square().mean();
        let value = read(loss.clone())?[0];
        let training_prediction = read(prediction)?;
        let mut grad = loss.backward();
        let gradients = self
            .network
            .parameters
            .iter()
            .zip(&self.network.shape.parameters)
            .map(|(p, s)| {
                Ok(MaxCallParameter {
                    name: s.name.clone(),
                    shape: s.shape,
                    values: if fixed_translation(&self.network.shape.normalization, &s.name) {
                        vec![0.; s.values.len()]
                    } else {
                        read(
                            p.grad_remove(&mut grad)
                                .ok_or_else(|| invalid("missing analysis gradient"))?,
                        )?
                    },
                })
            })
            .collect::<Result<Vec<_>, ComputeError>>()?;
        let proposed_running = running
            .iter()
            .map(|s| {
                Ok(MaxCallNormalizationState {
                    name: s.name.clone(),
                    mean: read(s.mean.clone())?,
                    variance: read(s.variance.clone())?,
                    batches: s.batches,
                })
            })
            .collect::<Result<Vec<_>, ComputeError>>()?;
        Ok(MaxCallGradientAnalysis {
            training_prediction,
            loss: value,
            gradients,
            proposed_running,
        })
    }
    pub fn optimizer_snapshot(&self) -> Result<MaxCallOptimizerSnapshot, ComputeError> {
        let capture = |values: &[Tensor<2>]| {
            values
                .iter()
                .zip(&self.network.shape.parameters)
                .map(|(t, p)| {
                    Ok(MaxCallParameter {
                        name: p.name.clone(),
                        shape: p.shape,
                        values: read(t.clone())?,
                    })
                })
                .collect::<Result<Vec<_>, ComputeError>>()
        };
        Ok(MaxCallOptimizerSnapshot {
            completed_updates: self.updates,
            first: capture(&self.first)?,
            second: capture(&self.second)?,
        })
    }
}

impl MaxCallDateLearner {
    pub fn last_gradient_snapshot(&self) -> Result<Vec<MaxCallParameter>, ComputeError> {
        self.last_gradients
            .as_ref()
            .ok_or_else(|| invalid("no committed gradient"))?
            .iter()
            .zip(&self.network.shape.parameters)
            .map(|(g, p)| {
                Ok(MaxCallParameter {
                    name: p.name.clone(),
                    shape: p.shape,
                    values: read(g.clone())?,
                })
            })
            .collect()
    }
}

impl MaxCallPartialCheckpoint {
    pub fn validate(&self) -> Result<(), ComputeError> {
        self.config.validate()?;
        let c = &self.config;
        let n = c.market.times.len();
        if self.schema != "volt.compute.maxcall_partial.v1"
            || self.next_date > n - 2
            || self.dates.len() != n
            || self.diagnostic_dates.len() != n
            || self.dates[0].is_some()
            || self.dates[n - 1].is_some()
            || self.diagnostic_dates[0].is_some()
            || self.diagnostic_dates[n - 1].is_some()
        {
            return Err(invalid("invalid partial suffix/schema/endpoints"));
        }
        for hash in [&self.training_market_sha256, &self.validation_market_sha256] {
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(invalid("invalid partial market digest"));
            }
        }
        let expected = ((self.next_date + 1)..n - 1)
            .rev()
            .filter(|i| c.market.exercise[*i])
            .collect::<Vec<_>>();
        if self.reports.iter().map(|r| r.date).collect::<Vec<_>>() != expected {
            return Err(invalid("partial reports do not match completed suffix"));
        }
        for date in 1..n - 1 {
            let present = date > self.next_date && c.market.exercise[date];
            if self.dates[date].is_some() != present
                || self.diagnostic_dates[date].is_some() != present
            {
                return Err(invalid("noncontiguous partial date suffix"));
            }
            if !present {
                continue;
            }
            let report = self
                .reports
                .iter()
                .find(|r| r.date == date)
                .ok_or_else(|| invalid("missing suffix report"))?;
            let steps = c.schedule(date).iter().map(|s| s.updates).sum::<usize>();
            let future = ((date + 1)..n - 1).find(|j| c.market.exercise[*j]);
            let base = if c.warm_start {
                ((date + 1)..n - 1)
                    .filter(|j| c.market.exercise[*j])
                    .map(|j| c.schedule(j).iter().map(|s| s.updates as u64).sum::<u64>())
                    .sum::<u64>()
            } else {
                0
            };
            if report.completed_updates != steps
                || report.diagnostic_best_update > steps as u64
                || report.warm_start_from != if c.warm_start { future } else { None }
                || !report.final_validation_mse.is_finite()
                || report.final_validation_mse < 0.
                || !report.diagnostic_best_mse.is_finite()
                || report.diagnostic_best_mse < 0.
                || !report.last_loss.is_finite()
                || !report.maximum_gradient_l2_norm.is_finite()
                || report.validation_trace.is_empty()
                || report
                    .validation_trace
                    .first()
                    .is_none_or(|v| v.completed_updates != 0)
                || report.validation_trace.last().is_none_or(|v| {
                    v.completed_updates != steps as u64
                        || v.regression_mse != report.final_validation_mse
                })
                || report
                    .validation_trace
                    .windows(2)
                    .any(|w| w[0].completed_updates >= w[1].completed_updates)
                || report
                    .validation_trace
                    .iter()
                    .any(|v| !v.regression_mse.is_finite() || v.regression_mse < 0.)
            {
                return Err(invalid("invalid completed-date report/optimizer count"));
            }
            for (net, count) in [
                (self.dates[date].as_ref().unwrap(), base + steps as u64),
                (
                    self.diagnostic_dates[date].as_ref().unwrap(),
                    base + report.diagnostic_best_update,
                ),
            ] {
                net.validate()?;
                if net.input_width != c.market.spots.len() + 1
                    || net.hidden_width != c.hidden_width
                    || net.output_width != 1
                    || net.normalization != c.normalization
                    || net.running.iter().any(|r| r.batches != count)
                {
                    return Err(invalid("partial network architecture/BN count differs"));
                }
            }
        }
        Ok(())
    }
}

fn fixed_translation(bn: &MaxCallBatchNorm, name: &str) -> bool {
    bn.canonical_zero_translations
        && bn.hidden
        && (name == "hidden0.bias"
            || name == "hidden1.bias"
            || (bn.input && name == "input_bn.beta"))
}
impl MaxCallNetworkSnapshot {
    /// Explicit representation change: zero redundant translations and move their
    /// current offsets out of the subsequent BN running means. Variances unchanged.
    pub fn canonicalized_translations(&self) -> Result<Self, ComputeError> {
        self.validate()?;
        if self.normalization.canonical_zero_translations {
            return Ok(self.clone());
        }
        if !self.normalization.hidden {
            return Err(invalid("translation canonicalization requires hidden BN"));
        }
        let mut out = self.clone();
        let mut offset = self.parameters[1].values.clone();
        if self.normalization.input {
            let beta = self
                .parameters
                .iter()
                .find(|p| p.name == "input_bn.beta")
                .ok_or_else(|| invalid("missing input BN beta"))?;
            for (i, b) in beta.values.iter().enumerate() {
                for (j, value) in offset.iter_mut().enumerate() {
                    *value += b * self.parameters[0].values[i * self.hidden_width + j];
                }
            }
        }
        let h0 = out
            .running
            .iter_mut()
            .find(|s| s.name == "hidden0_bn")
            .ok_or_else(|| invalid("missing hidden0 BN state"))?;
        for (m, shift) in h0.mean.iter_mut().zip(offset) {
            *m -= shift;
        }
        let h1 = out
            .running
            .iter_mut()
            .find(|s| s.name == "hidden1_bn")
            .ok_or_else(|| invalid("missing hidden1 BN state"))?;
        for (m, shift) in h1.mean.iter_mut().zip(&self.parameters[3].values) {
            *m -= shift;
        }
        out.normalization.canonical_zero_translations = true;
        for p in &mut out.parameters {
            if fixed_translation(&out.normalization, &p.name) {
                p.values.fill(0.);
            }
        }
        out.validate()?;
        Ok(out)
    }
}
impl MaxCallCheckpoint {
    pub fn canonicalized_translations(&self) -> Result<Self, ComputeError> {
        self.validate()?;
        let mut out = self.clone();
        out.config.normalization.canonical_zero_translations = true;
        out.dates = self
            .dates
            .iter()
            .map(|n| {
                n.as_ref()
                    .map(MaxCallNetworkSnapshot::canonicalized_translations)
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.basis = format!("canonical_zero_translations_of:{}", self.basis);
        out.validate()?;
        Ok(out)
    }
}
