//! Time-specific, semi-recurrent Deep Hedging MLPs. Features and previous
//! holdings enter two affine -> BatchNorm -> ReLU blocks and a signed linear
//! output. The raw paper profile has no feature standardization/action scaling.
//!
//! BatchNorm follows Ioffe/Szegedy <https://arxiv.org/abs/1502.03167>: training
//! uses population mini-batch variance, inference uses running statistics;
//! running variance receives the unbiased batch estimate. Momentum is the
//! weight of the NEW observation (explicit in the checkpoint). Adam follows
//! Kingma/Ba Algorithm 1 <https://arxiv.org/abs/1412.6980>, epsilon outside sqrt.
//!
//! CPU products use native-dtype GEMM. Accelerator products explicitly broadcast
//! multiply then reduce to avoid matmul precision substitution. The conservative
//! admission model reserves B*input*output elements per affine operation. The cap below bounds one modeled product, not the
//! full recurrent tape or actual allocator; the caller budgets the full job.
//! Forward preserves tracked previous holdings. One complete recurrent loss
//! must be backpropagated once, then backward_update creates fresh leaf handles.

use crate::{ComputeError, Precision, Session};
use burn::tensor::{
    Bool, DType, Device, Gradients, Tensor, TensorCreationOptions, TensorData, activation::relu,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const CHECKPOINT_SCHEMA: &str = "volt.compute.hedge_policy.v1";
pub const MAX_EXPANDED_PRODUCT_ELEMENTS: usize = 16_777_216;
const MAX_PARAMETERS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchNormConfig {
    pub enabled: bool,
    pub epsilon: f64,
    pub momentum: f64,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdamConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}
/// Explicit optional preprocessing (a deviation from the raw paper profile):
/// normalized feature = (raw feature - offset) * scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureAffine {
    pub offsets: Vec<f64>,
    pub scales: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HedgePolicyConfig {
    pub time_steps: usize,
    pub feature_count: usize,
    pub hedge_count: usize,
    pub include_previous_holdings: bool,
    pub hidden_width: usize,
    pub max_batch_rows: usize,
    pub batch_norm: BatchNormConfig,
    pub adam: AdamConfig,
    pub feature_affine: Option<FeatureAffine>,
    pub action_scales: Option<Vec<f64>>,
    /// Explicit quoted-unit scale for recurrent holdings. Input is previous/scale.
    /// None preserves the raw paper parameterization and legacy checkpoints.
    #[serde(default)]
    pub previous_holding_scales: Option<Vec<f64>>,
}
impl HedgePolicyConfig {
    pub fn paper_heston(max_batch_rows: usize) -> Self {
        Self {
            time_steps: 30,
            feature_count: 2,
            hedge_count: 2,
            include_previous_holdings: true,
            hidden_width: 17,
            max_batch_rows,
            batch_norm: BatchNormConfig {
                enabled: true,
                epsilon: 1e-5,
                momentum: 0.1,
            },
            adam: AdamConfig {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            feature_affine: None,
            action_scales: None,
            previous_holding_scales: None,
        }
    }
    pub fn paper_black_scholes(max_batch_rows: usize) -> Self {
        Self {
            feature_count: 1,
            hedge_count: 1,
            hidden_width: 16,
            ..Self::paper_heston(max_batch_rows)
        }
    }
    pub fn input_width(&self) -> usize {
        self.feature_count
            + if self.include_previous_holdings {
                self.hedge_count
            } else {
                0
            }
    }
    pub fn validate(&self) -> Result<(), ComputeError> {
        if !(1..=365).contains(&self.time_steps)
            || !(1..=64).contains(&self.feature_count)
            || !(1..=32).contains(&self.hedge_count)
            || !(1..=256).contains(&self.hidden_width)
            || self.max_batch_rows == 0
        {
            return Err(invalid(
                "policy dimensions exceed supported positive bounds",
            ));
        }
        if !self.batch_norm.epsilon.is_finite()
            || self.batch_norm.epsilon <= 0.0
            || !self.batch_norm.momentum.is_finite()
            || !(0.0..=1.0).contains(&self.batch_norm.momentum)
        {
            return Err(invalid("invalid BatchNorm epsilon/momentum"));
        }
        if [self.adam.beta1, self.adam.beta2, self.adam.epsilon]
            .iter()
            .any(|x| !x.is_finite())
            || !(0.0..1.0).contains(&self.adam.beta1)
            || !(0.0..1.0).contains(&self.adam.beta2)
            || self.adam.epsilon <= 0.0
        {
            return Err(invalid("invalid Adam beta/epsilon"));
        }
        if let Some(a) = &self.feature_affine
            && (a.offsets.len() != self.feature_count
                || a.scales.len() != self.feature_count
                || a.offsets.iter().any(|x| !x.is_finite())
                || a.scales.iter().any(|x| !x.is_finite() || *x <= 0.0))
        {
            return Err(invalid("invalid feature affine dimensions or values"));
        }
        if let Some(a) = &self.action_scales
            && (a.len() != self.hedge_count || a.iter().any(|x| !x.is_finite() || *x <= 0.0))
        {
            return Err(invalid("invalid action scales"));
        }
        if let Some(scales) = &self.previous_holding_scales
            && (scales.len() != self.hedge_count
                || scales
                    .iter()
                    .any(|s| !s.is_finite() || *s <= 0.0 || !(1.0 / s).is_finite()))
        {
            return Err(invalid("invalid previous holding unit scales"));
        }
        let count: usize = parameter_shapes(self).iter().map(|s| s[0] * s[1]).sum();
        if count > MAX_PARAMETERS {
            return Err(invalid("policy exceeds parameter cap"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterSnapshot {
    pub shape: [usize; 2],
    pub values: Vec<f64>,
    pub first_moment: Vec<f64>,
    pub second_moment: Vec<f64>,
    pub adam_step: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchNormSnapshot {
    pub running_mean: Vec<f64>,
    pub running_variance: Vec<f64>,
    pub batches_tracked: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicySnapshot {
    pub schema: String,
    pub config: HedgePolicyConfig,
    pub precision: Precision,
    pub updates: u64,
    pub parameters: Vec<ParameterSnapshot>,
    pub batch_norm: Vec<BatchNormSnapshot>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStats {
    pub updated_parameters: usize,
    pub gradient_l2_norm: f64,
    pub updates: u64,
}
struct Parameter {
    value: Tensor<2>,
    first: Tensor<2>,
    second: Tensor<2>,
    step: u64,
}
struct BatchNormState {
    mean: Tensor<2>,
    variance: Tensor<2>,
    batches: u64,
}

enum ForwardMode {
    Training,
    Inference,
    Calibration,
}

pub struct HedgePolicy {
    config: HedgePolicyConfig,
    precision: Precision,
    device: Device,
    parameters: Vec<Parameter>,
    norms: Vec<BatchNormState>,
    updates: u64,
}
impl HedgePolicy {
    pub fn new(
        session: &Session,
        config: HedgePolicyConfig,
        seed: u64,
    ) -> Result<Self, ComputeError> {
        validate_session_config(session, &config)?;
        let mut rng = seed;
        let shapes = parameter_shapes(&config);
        let mut parameters = Vec::with_capacity(shapes.len());
        for (index, shape) in shapes.into_iter().enumerate() {
            let kind = index % 10;
            let values = (0..shape[0] * shape[1])
                .map(|_| match kind {
                    0 | 4 | 8 => {
                        let limit = (6.0 / (shape[0] + shape[1]) as f64).sqrt();
                        (uniform(&mut rng) * 2.0 - 1.0) * limit
                    }
                    2 | 6 => 1.0,
                    _ => 0.0,
                })
                .collect::<Vec<_>>();
            let value = make_tensor(session.device(), session.precision(), values, shape)
                .autodiff()
                .require_grad();
            let zeros = make_tensor(
                session.device(),
                session.precision(),
                vec![0.0; shape[0] * shape[1]],
                shape,
            );
            parameters.push(Parameter {
                value,
                first: zeros.clone(),
                second: zeros,
                step: 0,
            });
        }
        let mut norms = Vec::with_capacity(config.time_steps * 2);
        for _ in 0..config.time_steps * 2 {
            norms.push(BatchNormState {
                mean: make_tensor(
                    session.device(),
                    session.precision(),
                    vec![0.0; config.hidden_width],
                    [1, config.hidden_width],
                ),
                variance: make_tensor(
                    session.device(),
                    session.precision(),
                    vec![1.0; config.hidden_width],
                    [1, config.hidden_width],
                ),
                batches: 0,
            });
        }
        Ok(Self {
            config,
            precision: session.precision(),
            device: session.device().clone(),
            parameters,
            norms,
            updates: 0,
        })
    }
    pub fn config(&self) -> &HedgePolicyConfig {
        &self.config
    }
    pub fn precision(&self) -> Precision {
        self.precision
    }
    /// Ten leaves per time net, ordered W0,b0,gamma0,beta0,W1,b1,gamma1,beta1,W2,b2.
    pub fn parameters(&self) -> Vec<Tensor<2>> {
        self.parameters.iter().map(|p| p.value.clone()).collect()
    }
    pub fn parameter_count(&self) -> usize {
        self.parameters
            .iter()
            .map(|p| p.value.dims().iter().product::<usize>())
            .sum()
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    /// Training preserves the recurrent AD graph and updates running statistics.
    /// Evaluation detaches all operands and creates no AD graph, including when
    /// callers supply tracked inputs. Eval never updates BatchNorm statistics.
    pub fn forward_at(
        &mut self,
        step: usize,
        features: Tensor<2>,
        previous: Tensor<2>,
        train: bool,
    ) -> Result<Tensor<2>, ComputeError> {
        self.forward_mode(
            step,
            features,
            previous,
            if train {
                ForwardMode::Training
            } else {
                ForwardMode::Inference
            },
        )
    }
    /// Refresh frozen-weight BatchNorm statistics on a TRAIN-only calibration
    /// batch. Both hidden layers store that batch's exact mean and unbiased
    /// variance, and normalize using those same stored values. Returned holdings
    /// therefore replay in inference mode and can be propagated to the next net.
    /// No AD graph, optimizer update, EMA mixing or training counter increment.
    /// With BatchNorm disabled this is ordinary inference with no state change.
    pub fn calibrate_at(
        &mut self,
        step: usize,
        features: Tensor<2>,
        previous: Tensor<2>,
    ) -> Result<Tensor<2>, ComputeError> {
        self.forward_mode(step, features, previous, ForwardMode::Calibration)
    }
    fn forward_mode(
        &mut self,
        step: usize,
        features: Tensor<2>,
        previous: Tensor<2>,
        forward_mode: ForwardMode,
    ) -> Result<Tensor<2>, ComputeError> {
        let train = matches!(forward_mode, ForwardMode::Training);
        let calibrate = matches!(forward_mode, ForwardMode::Calibration);
        if step >= self.config.time_steps {
            return Err(invalid("policy decision step is out of range"));
        }
        let [batch, width] = features.dims();
        if batch == 0
            || batch > self.config.max_batch_rows
            || width != self.config.feature_count
            || previous.dims() != [batch, self.config.hedge_count]
        {
            return Err(invalid(
                "policy feature/holdings shape or batch cap mismatch",
            ));
        }
        if (train || calibrate) && self.config.batch_norm.enabled && batch < 2 {
            return Err(invalid(
                "BatchNorm training/calibration requires at least two rows",
            ));
        }
        check_tensor(&features, &self.device, self.precision)?;
        check_tensor(&previous, &self.device, self.precision)?;
        let mut finite = features
            .clone()
            .without_autodiff()
            .is_finite()
            .all()
            .bool_and(previous.clone().without_autodiff().is_finite().all());
        for (input, output) in [
            (self.config.input_width(), self.config.hidden_width),
            (self.config.hidden_width, self.config.hidden_width),
            (self.config.hidden_width, self.config.hedge_count),
        ] {
            check_product_shape(batch, input, output, MAX_EXPANDED_PRODUCT_ELEMENTS)?;
        }
        if train
            && self.config.batch_norm.enabled
            && self.norms[step * 2..step * 2 + 2]
                .iter()
                .any(|s| s.batches == u64::MAX)
        {
            return Err(invalid("BatchNorm batch counter overflow"));
        }
        let mode = |tensor: Tensor<2>| {
            if train {
                tensor.autodiff()
            } else {
                tensor.without_autodiff()
            }
        };
        let mut features = mode(features);
        if let Some(a) = &self.config.feature_affine {
            features = (features
                - self.constant(a.offsets.clone(), [1, self.config.feature_count]))
                * self.constant(a.scales.clone(), [1, self.config.feature_count]);
        }
        let mut x = if self.config.include_previous_holdings {
            let mut previous = mode(previous);
            if let Some(scales) = &self.config.previous_holding_scales {
                previous = previous
                    * self.constant(
                        scales.iter().map(|s| 1.0 / s).collect(),
                        [1, self.config.hedge_count],
                    );
            }
            Tensor::cat(vec![features, previous], 1)
        } else {
            features
        };
        let offset = step * 10;
        let mut norm_updates = Vec::with_capacity(2);
        for layer in 0..2 {
            let at = offset + layer * 4;
            x = strict_matmul(
                x,
                mode(self.parameters[at].value.clone()),
                MAX_EXPANDED_PRODUCT_ELEMENTS,
            )? + mode(self.parameters[at + 1].value.clone());
            if self.config.batch_norm.enabled {
                let gamma = mode(self.parameters[at + 2].value.clone());
                let beta = mode(self.parameters[at + 3].value.clone());
                let state = &self.norms[step * 2 + layer];
                let (mean, variance) = if train || calibrate {
                    let mean = x.clone().sum_dim(0) * (1.0 / batch as f64);
                    let centered = x.clone() - mean.clone();
                    let variance = centered.square().sum_dim(0) * (1.0 / batch as f64);
                    let (candidate, output_mean, output_variance) = if calibrate {
                        let unbiased = variance * (batch as f64 / (batch - 1) as f64);
                        (
                            BatchNormState {
                                mean: mean.clone(),
                                variance: unbiased.clone(),
                                batches: state.batches,
                            },
                            mean,
                            unbiased,
                        )
                    } else {
                        let momentum = self.config.batch_norm.momentum;
                        let candidate = BatchNormState {
                            mean: state.mean.clone() * (1.0 - momentum)
                                + mean.clone().without_autodiff() * momentum,
                            variance: state.variance.clone() * (1.0 - momentum)
                                + variance.clone().without_autodiff()
                                    * (momentum * batch as f64 / (batch - 1) as f64),
                            batches: state.batches + 1,
                        };
                        (candidate, mean, variance)
                    };
                    finite = finite
                        .bool_and(candidate.mean.clone().is_finite().all())
                        .bool_and(candidate.variance.clone().is_finite().all());
                    norm_updates.push((step * 2 + layer, candidate));
                    (output_mean, output_variance)
                } else {
                    (state.mean.clone(), state.variance.clone())
                };
                x = (x - mean) / (variance + self.config.batch_norm.epsilon).sqrt() * gamma + beta;
            }
            x = relu(x);
        }
        let mut output = strict_matmul(
            x,
            mode(self.parameters[offset + 8].value.clone()),
            MAX_EXPANDED_PRODUCT_ELEMENTS,
        )? + mode(self.parameters[offset + 9].value.clone());
        if let Some(scales) = &self.config.action_scales {
            output = output * self.constant(scales.clone(), [1, self.config.hedge_count]);
        }
        finite = finite.bool_and(output.clone().without_autodiff().is_finite().all());
        if !bool_value(finite)? {
            return Err(ComputeError::Execution(
                "nonfinite policy input, output or BatchNorm candidate; running state unchanged"
                    .into(),
            ));
        }
        for (index, state) in norm_updates {
            self.norms[index] = state;
        }
        Ok(output)
    }
    /// All arithmetic for Adam and its moments stays on the original device.
    /// Only reduced finite flags and norm are downloaded. Unused parameters are
    /// skipped and keep their per-parameter Adam step, as with sparse objectives.
    /// Candidate updates are checked before committing any parameter/moment.
    pub fn backward_update(
        &mut self,
        gradients: &mut Gradients,
        learning_rate: f64,
    ) -> Result<UpdateStats, ComputeError> {
        if !learning_rate.is_finite()
            || learning_rate <= 0.0
            || (self.precision == Precision::F32
                && (!(learning_rate as f32).is_finite() || (learning_rate as f32) <= 0.0))
        {
            return Err(invalid(
                "Adam learning rate must be positive and representable",
            ));
        }
        let next = self
            .updates
            .checked_add(1)
            .ok_or_else(|| invalid("Adam update counter overflow"))?;
        let mut candidates = Vec::new();
        let mut finite_flags: Vec<Tensor<1, Bool>> = Vec::new();
        let mut squared_norms: Vec<Tensor<1>> = Vec::new();
        let a = self.config.adam;
        for (index, p) in self.parameters.iter().enumerate() {
            let Some(g) = p.value.grad_remove(gradients) else {
                continue;
            };
            let step = p
                .step
                .checked_add(1)
                .ok_or_else(|| invalid("Adam parameter counter overflow"))?;
            let first = p.first.clone() * a.beta1 + g.clone() * (1.0 - a.beta1);
            let second = p.second.clone() * a.beta2 + g.clone().square() * (1.0 - a.beta2);
            let corrected_first = first.clone() * (1.0 / (1.0 - a.beta1.powf(step as f64)));
            let corrected_second = second.clone() * (1.0 / (1.0 - a.beta2.powf(step as f64)));
            let value = p.value.clone().without_autodiff()
                - corrected_first / (corrected_second.sqrt() + a.epsilon) * learning_rate;
            finite_flags.push(
                g.clone()
                    .is_finite()
                    .all()
                    .bool_and(value.clone().is_finite().all())
                    .bool_and(first.clone().is_finite().all())
                    .bool_and(second.clone().is_finite().all()),
            );
            squared_norms.push(g.square().sum());
            candidates.push((
                index,
                Parameter {
                    value,
                    first,
                    second,
                    step,
                },
            ));
        }
        if candidates.is_empty() {
            return Err(invalid("Adam update has no parameter gradients"));
        }
        let valid = bool_value(Tensor::cat(finite_flags, 0).all())?;
        let norm = Tensor::cat(squared_norms, 0)
            .sum()
            .try_to_vec_as::<f64>()
            .map_err(execution)?[0]
            .sqrt();
        if !valid || !norm.is_finite() {
            return Err(ComputeError::Execution(
                "nonfinite Adam gradient, moments, update or gradient norm; parameters unchanged"
                    .into(),
            ));
        }
        let updated_parameters = candidates.len();
        for (i, mut p) in candidates {
            p.value = p.value.autodiff().require_grad();
            self.parameters[i] = p;
        }
        self.updates = next;
        Ok(UpdateStats {
            updated_parameters,
            gradient_l2_norm: norm,
            updates: next,
        })
    }
    pub fn snapshot(&self) -> Result<PolicySnapshot, ComputeError> {
        let parameters = self
            .parameters
            .iter()
            .map(|p| {
                Ok(ParameterSnapshot {
                    shape: p.value.dims(),
                    values: read(&p.value)?,
                    first_moment: read(&p.first)?,
                    second_moment: read(&p.second)?,
                    adam_step: p.step,
                })
            })
            .collect::<Result<Vec<_>, ComputeError>>()?;
        let batch_norm = self
            .norms
            .iter()
            .map(|n| {
                Ok(BatchNormSnapshot {
                    running_mean: read(&n.mean)?,
                    running_variance: read(&n.variance)?,
                    batches_tracked: n.batches,
                })
            })
            .collect::<Result<Vec<_>, ComputeError>>()?;
        Ok(PolicySnapshot {
            schema: CHECKPOINT_SCHEMA.into(),
            config: self.config.clone(),
            precision: self.precision,
            updates: self.updates,
            parameters,
            batch_norm,
        })
    }
    pub fn from_snapshot(session: &Session, s: PolicySnapshot) -> Result<Self, ComputeError> {
        validate_session_config(session, &s.config)?;
        if s.schema != CHECKPOINT_SCHEMA || s.precision != session.precision() {
            return Err(invalid(
                "policy checkpoint schema or precision mismatch; dtype conversion must be explicit",
            ));
        }
        validate_policy_snapshot(&s)?;
        let parameters = s
            .parameters
            .into_iter()
            .map(|p| Parameter {
                value: make_tensor(session.device(), s.precision, p.values, p.shape)
                    .autodiff()
                    .require_grad(),
                first: make_tensor(session.device(), s.precision, p.first_moment, p.shape),
                second: make_tensor(session.device(), s.precision, p.second_moment, p.shape),
                step: p.adam_step,
            })
            .collect();
        let norms = s
            .batch_norm
            .into_iter()
            .map(|n| BatchNormState {
                mean: make_tensor(
                    session.device(),
                    s.precision,
                    n.running_mean,
                    [1, s.config.hidden_width],
                ),
                variance: make_tensor(
                    session.device(),
                    s.precision,
                    n.running_variance,
                    [1, s.config.hidden_width],
                ),
                batches: n.batches_tracked,
            })
            .collect();
        Ok(Self {
            config: s.config,
            precision: s.precision,
            device: session.device().clone(),
            parameters,
            norms,
            updates: s.updates,
        })
    }
    /// Publish a complete checkpoint by same-directory rename, preserving the
    /// previous checkpoint if serialization or writing the candidate fails.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ComputeError> {
        use std::io::Write;
        use std::sync::atomic::{AtomicU64, Ordering};
        static NONCE: AtomicU64 = AtomicU64::new(0);
        let bytes = serde_json::to_vec(&self.snapshot()?).map_err(execution)?;
        let path = path.as_ref();
        let mut name = path
            .file_name()
            .ok_or_else(|| invalid("checkpoint path must name a file"))?
            .to_os_string();
        name.push(format!(
            ".tmp-{}-{}",
            std::process::id(),
            NONCE.fetch_add(1, Ordering::Relaxed)
        ));
        let temporary = path.with_file_name(name);
        let mut created = false;
        let result = (|| -> std::io::Result<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            created = true;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temporary, path)
        })();
        if result.is_err() && created {
            let _ = std::fs::remove_file(&temporary);
        }
        result.map_err(execution)
    }
    pub fn load(session: &Session, path: impl AsRef<Path>) -> Result<Self, ComputeError> {
        let bytes = std::fs::read(path).map_err(execution)?;
        let snapshot: PolicySnapshot = serde_json::from_slice(&bytes).map_err(execution)?;
        Self::from_snapshot(session, snapshot)
    }
    fn constant(&self, values: Vec<f64>, shape: [usize; 2]) -> Tensor<2> {
        make_tensor(&self.device, self.precision, values, shape)
    }
}

/// Validate checkpoint schema, shapes, finite values, dtype representation and
/// optimizer/normalization state without creating a session or any tensors.
pub fn validate_policy_snapshot(s: &PolicySnapshot) -> Result<(), ComputeError> {
    validate_policy_config(&s.config, s.precision)?;
    if s.schema != CHECKPOINT_SCHEMA {
        return Err(invalid("policy checkpoint schema mismatch"));
    }
    let shapes = parameter_shapes(&s.config);
    if s.parameters.len() != shapes.len() || s.batch_norm.len() != s.config.time_steps * 2 {
        return Err(invalid("policy checkpoint parameter/state count mismatch"));
    }
    for (p, shape) in s.parameters.iter().zip(&shapes) {
        let count = shape[0] * shape[1];
        if p.shape != *shape
            || p.values.len() != count
            || p.first_moment.len() != count
            || p.second_moment.len() != count
            || p.adam_step > s.updates
        {
            return Err(invalid("policy checkpoint parameter shape/step mismatch"));
        }
        check_values(&p.values, s.precision)?;
        check_values(&p.first_moment, s.precision)?;
        check_values(&p.second_moment, s.precision)?;
        if p.second_moment.iter().any(|x| *x < 0.0)
            || (p.adam_step == 0
                && p.first_moment
                    .iter()
                    .chain(&p.second_moment)
                    .any(|x| *x != 0.0))
        {
            return Err(invalid("invalid Adam checkpoint moments"));
        }
    }
    for n in &s.batch_norm {
        if n.running_mean.len() != s.config.hidden_width
            || n.running_variance.len() != s.config.hidden_width
            || n.running_variance.iter().any(|x| *x < 0.0)
        {
            return Err(invalid("invalid BatchNorm checkpoint state"));
        }
        check_values(&n.running_mean, s.precision)?;
        check_values(&n.running_variance, s.precision)?;
    }
    Ok(())
}

/// Differentiable native-dtype matrix product with explicit modeled expansion.
/// Plain operands become constants when mixed with tracked operands. No tracked
/// operand is detached, including recurrent previous holdings.
pub fn strict_matmul(
    left: Tensor<2>,
    right: Tensor<2>,
    max_expanded_elements: usize,
) -> Result<Tensor<2>, ComputeError> {
    let [batch, input] = left.dims();
    let [other, output] = right.dims();
    if input != other
        || left.dtype() != right.dtype()
        || !matches!(left.dtype(), DType::F32 | DType::F64)
        || left.clone().without_autodiff().device() != right.clone().without_autodiff().device()
    {
        return Err(invalid(
            "strict policy matmul shape/device/precision mismatch",
        ));
    }
    check_product_shape(batch, input, output, max_expanded_elements)?;
    // Flex's CPU GEMM preserves the operand dtype and its matmul adjoints.
    // Accelerator Auto matmul can substitute TF32, so those backends retain
    // the explicit product/reduction path. Keep conservative expansion admission.
    if left.clone().without_autodiff().device() == burn::tensor::Device::flex() {
        return Ok(left.matmul(right));
    }
    Ok((left.unsqueeze_dim::<3>(2) * right.unsqueeze_dim::<3>(0))
        .sum_dim(1)
        .reshape([batch, output]))
}
fn check_product_shape(b: usize, i: usize, o: usize, cap: usize) -> Result<(), ComputeError> {
    if b == 0
        || i == 0
        || o == 0
        || b.checked_mul(i)
            .and_then(|n| n.checked_mul(o))
            .is_none_or(|n| n > cap)
    {
        return Err(ComputeError::BudgetExceeded(
            "strict policy matmul exceeds modeled expanded-product cap".into(),
        ));
    }
    Ok(())
}
fn parameter_shapes(c: &HedgePolicyConfig) -> Vec<[usize; 2]> {
    let mut out = Vec::with_capacity(c.time_steps * 10);
    for _ in 0..c.time_steps {
        let w = c.hidden_width;
        out.extend([
            [c.input_width(), w],
            [1, w],
            [1, w],
            [1, w],
            [w, w],
            [1, w],
            [1, w],
            [1, w],
            [w, c.hedge_count],
            [1, c.hedge_count],
        ]);
    }
    out
}
fn validate_session_config(session: &Session, c: &HedgePolicyConfig) -> Result<(), ComputeError> {
    if c.max_batch_rows > session.max_batch_rows() {
        return Err(invalid("policy batch cap exceeds session cap"));
    }
    validate_policy_config(c, session.precision())
}
/// Pure configuration/precision preflight; no runtime or device initialization.
pub fn validate_policy_config(
    c: &HedgePolicyConfig,
    precision: Precision,
) -> Result<(), ComputeError> {
    c.validate()?;
    let values = [
        c.batch_norm.epsilon,
        c.batch_norm.momentum,
        c.adam.beta1,
        c.adam.beta2,
        c.adam.epsilon,
    ];
    check_values(&values, precision)?;
    if precision == Precision::F32
        && ((c.batch_norm.epsilon as f32) <= 0.0
            || (c.adam.epsilon as f32) <= 0.0
            || (c.adam.beta1 as f32) >= 1.0
            || (c.adam.beta2 as f32) >= 1.0)
    {
        return Err(invalid(
            "policy optimizer/normalization constants collapse in f32",
        ));
    }
    if let Some(a) = &c.feature_affine {
        check_values(&a.offsets, precision)?;
        check_values(&a.scales, precision)?;
        if a.scales
            .iter()
            .any(|x| precision == Precision::F32 && (*x as f32) <= 0.0)
        {
            return Err(invalid("feature scale underflows in f32"));
        }
    }
    if let Some(a) = &c.previous_holding_scales {
        let inverse: Vec<_> = a.iter().map(|s| 1.0 / s).collect();
        check_values(a, precision)?;
        check_values(&inverse, precision)?;
        if precision == Precision::F32 && a.iter().any(|s| (*s as f32) <= 0.0) {
            return Err(invalid("previous holding scale underflows in f32"));
        }
    }
    if let Some(a) = &c.action_scales {
        check_values(a, precision)?;
        if a.iter()
            .any(|x| precision == Precision::F32 && (*x as f32) <= 0.0)
        {
            return Err(invalid("action scale underflows in f32"));
        }
    }
    Ok(())
}
fn dtype(p: Precision) -> DType {
    match p {
        Precision::F32 => DType::F32,
        Precision::F64 => DType::F64,
    }
}
fn make_tensor(device: &Device, p: Precision, v: Vec<f64>, shape: [usize; 2]) -> Tensor<2> {
    Tensor::from_data(
        TensorData::new(v, shape),
        TensorCreationOptions::new(device.clone()).with_dtype(dtype(p)),
    )
}
fn check_tensor(t: &Tensor<2>, d: &Device, p: Precision) -> Result<(), ComputeError> {
    if t.dtype() != dtype(p) || t.clone().without_autodiff().device() != *d {
        return Err(invalid("policy input device/precision mismatch"));
    }
    Ok(())
}
fn check_values(v: &[f64], p: Precision) -> Result<(), ComputeError> {
    if v.iter()
        .any(|x| !x.is_finite() || (p == Precision::F32 && !(*x as f32).is_finite()))
    {
        return Err(invalid(
            "nonfinite or unrepresentable policy checkpoint/config value",
        ));
    }
    Ok(())
}
fn read(t: &Tensor<2>) -> Result<Vec<f64>, ComputeError> {
    let values = t
        .clone()
        .without_autodiff()
        .try_to_vec_as::<f64>()
        .map_err(execution)?;
    if values.iter().any(|x| !x.is_finite()) {
        return Err(ComputeError::Execution("nonfinite policy snapshot".into()));
    }
    Ok(values)
}
fn bool_value(t: Tensor<1, Bool>) -> Result<bool, ComputeError> {
    Ok(t.try_to_vec_as::<bool>().map_err(execution)?[0])
}
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (z >> 11) as f64 * (1.0 / 9007199254740992.0)
}
fn invalid(s: &str) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn execution(e: impl std::fmt::Display) -> ComputeError {
    ComputeError::Execution(e.to_string())
}
