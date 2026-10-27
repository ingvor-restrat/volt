//! Learned Bermudan stopping on exact-transition risk-neutral GBM.
//!
//! 2019 direct stopping and 2020 continuation regression are separate algorithms.
//! The nested one-step frozen-value-proxy dual below is a valid alternate
//! martingale construction, not the papers' policy-tail estimator. See the guide.
use crate::{ComputeError, ExecutionTarget, Precision, Session};
use burn::tensor::{
    DType, Tensor, TensorCreationOptions, TensorData,
    activation::{relu, sigmoid, tanh},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA: &str = "volt.compute.deep_stopping.v1";
fn invalid(x: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(x.into())
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GbmPutSpec {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub times: Vec<f64>,
    pub exercise: Vec<bool>,
}
impl GbmPutSpec {
    pub fn monthly_put() -> Self {
        Self {
            spot: 100.,
            strike: 100.,
            rate: 0.05,
            dividend_yield: 0.,
            volatility: 0.2,
            times: (0..=12).map(|n| n as f64 / 12.).collect(),
            exercise: (0..=12).map(|n| n > 0).collect(),
        }
    }
    pub fn validate(&self) -> Result<(), ComputeError> {
        let n = self.times.len();
        if !(2..=65).contains(&n)
            || self.exercise.len() != n
            || !self.exercise[n - 1]
            || self.times[0] != 0.
            || !self.spot.is_finite()
            || !(1e-12..=1e12).contains(&self.spot)
            || !self.strike.is_finite()
            || !(1e-12..=1e12).contains(&self.strike)
            || !self.rate.is_finite()
            || !self.dividend_yield.is_finite()
            || !self.volatility.is_finite()
            || self.volatility < 0.
            || self.times.iter().any(|t| !t.is_finite() || *t < 0.)
            || self.times.windows(2).any(|w| w[1] <= w[0])
            || self.rate.abs() * self.times[n - 1] > 20.
            || self.dividend_yield.abs() * self.times[n - 1] > 100.
            || self.volatility * self.times[n - 1].sqrt() > 10.
            || !self.payoff_cap().is_finite()
            || self.payoff_cap() > 1e12
        {
            return Err(invalid("invalid or unsupported finite GBM put/grid"));
        }
        Ok(())
    }
    fn payoff(&self, date: usize, spot: f64) -> f64 {
        (self.strike - spot).max(0.) * (-self.rate * self.times[date]).exp()
    }
    fn step(&self, date: usize, spot: f64, z: f64) -> Result<f64, ComputeError> {
        let dt = self.times[date + 1] - self.times[date];
        let s = spot
            * ((self.rate - self.dividend_yield - 0.5 * self.volatility.powi(2)) * dt
                + self.volatility * dt.sqrt() * z)
                .exp();
        if !s.is_finite() || s <= 0. {
            return Err(invalid(
                "GBM transition overflow/underflow; no path deletion",
            ));
        }
        Ok(s)
    }
    fn features(&self, date: usize, spot: f64) -> [f64; 2] {
        [
            (spot / self.strike).ln(),
            self.payoff(date, spot) / self.strike,
        ]
    }
    fn payoff_cap(&self) -> f64 {
        self.times
            .iter()
            .map(|t| self.strike * (-self.rate * t).exp())
            .fold(0., f64::max)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoppingAlgorithm {
    DirectStopping2019,
    ContinuationRegression2020,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoppingTrainingConfig {
    pub spec: GbmPutSpec,
    pub algorithm: StoppingAlgorithm,
    pub target: ExecutionTarget,
    pub precision: Precision,
    pub width: usize,
    pub training_paths: usize,
    pub validation_paths: usize,
    pub batch_rows: usize,
    pub updates_per_date: usize,
    pub validation_every: usize,
    pub learning_rate: f64,
    pub training_seed: u64,
    pub validation_seed: u64,
    pub initialization_seed: u64,
    pub max_working_bytes: u64,
}
impl Default for StoppingTrainingConfig {
    fn default() -> Self {
        Self {
            spec: GbmPutSpec::monthly_put(),
            algorithm: StoppingAlgorithm::ContinuationRegression2020,
            target: ExecutionTarget::Cpu,
            precision: Precision::F64,
            width: 16,
            training_paths: 32768,
            validation_paths: 8192,
            batch_rows: 512,
            updates_per_date: 500,
            validation_every: 100,
            learning_rate: 0.003,
            training_seed: 10010001,
            validation_seed: 10020001,
            initialization_seed: 10030001,
            max_working_bytes: 512 * 1024 * 1024,
        }
    }
}
impl StoppingTrainingConfig {
    pub fn validate(&self) -> Result<(), ComputeError> {
        self.spec.validate()?;
        let n = self.spec.times.len();
        if self.width == 0
            || self.width > 128
            || self.training_paths < 2
            || self.training_paths > 1_000_000
            || self.validation_paths < 2
            || self.validation_paths > 1_000_000
            || self.batch_rows < 2
            || self.batch_rows > self.training_paths
            || self.batch_rows > 8192
            || self.updates_per_date == 0
            || self.updates_per_date > 100_000
            || self.validation_every == 0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || self.learning_rate > 1.
            || self.training_seed == self.validation_seed
        {
            return Err(invalid(
                "invalid stopping training dimensions/seeds/optimizer",
            ));
        }
        let bytes = (self.training_paths + self.validation_paths) as u128 * n as u128 * 32
            + (self.batch_rows as u128 * self.width as u128 * self.width as u128 * 128)
            + 16 * 1024 * 1024;
        if bytes > self.max_working_bytes as u128 {
            return Err(ComputeError::BudgetExceeded(format!(
                "stopping modeled workspace {bytes} exceeds {}",
                self.max_working_bytes
            )));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoppingNetwork {
    pub width: usize,
    pub parameters: Vec<Vec<f64>>,
}
impl StoppingNetwork {
    fn validate(&self) -> Result<(), ComputeError> {
        let w = self.width;
        if w == 0
            || w > 128
            || self.parameters.len() != 6
            || self
                .parameters
                .iter()
                .zip([2 * w, w, w * w, w, w, 1])
                .any(|(p, n)| p.len() != n || p.iter().any(|v| !v.is_finite() || v.abs() > 1e6))
        {
            return Err(invalid("invalid stopping network"));
        }
        Ok(())
    }
    fn new(width: usize, seed: u64, date: usize, direct: bool) -> Self {
        let sizes = [
            (2, width),
            (1, width),
            (width, width),
            (1, width),
            (width, 1),
            (1, 1),
        ];
        let mut index = 0;
        let parameters = sizes
            .iter()
            .enumerate()
            .map(|(j, (a, b))| {
                (0..a * b)
                    .map(|_| {
                        index += 1;
                        if j % 2 == 1 {
                            0.
                        } else {
                            let z = normal(seed, index, date, if direct { 8 } else { 7 }, 0);
                            z * (2. / (*a + *b) as f64).sqrt()
                        }
                    })
                    .collect()
            })
            .collect();
        Self { width, parameters }
    }
    fn predict(&self, x: [f64; 2], direct: bool) -> f64 {
        let w = self.width;
        let activation = |v: f64| if direct { v.max(0.) } else { v.tanh() };
        let mut a = vec![0.; w];
        let mut b = vec![0.; w];
        for (j, y) in a.iter_mut().enumerate() {
            *y = activation(
                self.parameters[1][j]
                    + x[0] * self.parameters[0][j]
                    + x[1] * self.parameters[0][w + j],
            );
        }
        for (j, y) in b.iter_mut().enumerate() {
            *y = activation(
                self.parameters[3][j]
                    + a.iter()
                        .enumerate()
                        .map(|(i, v)| v * self.parameters[2][i * w + j])
                        .sum::<f64>(),
            );
        }
        let y = self.parameters[5][0]
            + b.iter()
                .zip(&self.parameters[4])
                .map(|(a, b)| a * b)
                .sum::<f64>();
        if direct {
            if y >= 0. {
                1. / (1. + (-y).exp())
            } else {
                let e = y.exp();
                e / (1. + e)
            }
        } else {
            y
        }
    }
    fn output_bound(&self) -> f64 {
        self.parameters[5][0].abs() + self.parameters[4].iter().map(|x| x.abs()).sum::<f64>()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoppingDateModel {
    pub continuation: StoppingNetwork,
    pub decision: Option<StoppingNetwork>,
    pub selected_update: usize,
    pub completed_updates: usize,
    pub training_parameter_change_l2: f64,
    pub validation_lower_mean: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoppingCheckpoint {
    pub schema: String,
    pub config: StoppingTrainingConfig,
    pub dates: Vec<Option<StoppingDateModel>>,
    pub stop_initial: bool,
    pub initial_training_continuation: f64,
}
impl StoppingCheckpoint {
    pub fn validate(&self) -> Result<(), ComputeError> {
        self.config.validate()?;
        if self.schema != SCHEMA
            || self.dates.len() != self.config.spec.times.len()
            || !self.initial_training_continuation.is_finite()
        {
            return Err(invalid("invalid stopping checkpoint shape/schema"));
        }
        let last = self.dates.len() - 1;
        for (n, m) in self.dates.iter().enumerate() {
            if n == 0 || n == last || !self.config.spec.exercise[n] {
                if m.is_some() {
                    return Err(invalid("unexpected network at excluded/endpoint date"));
                }
            } else {
                let m = m
                    .as_ref()
                    .ok_or_else(|| invalid("missing stopping date model"))?;
                m.continuation.validate()?;
                if m.continuation.width != self.config.width
                    || !m.continuation.output_bound().is_finite()
                    || !(m.continuation.output_bound() * self.config.spec.strike).is_finite()
                    || m.continuation.output_bound() * self.config.spec.strike > 1e12
                    || m.completed_updates > self.config.updates_per_date
                    || !m.training_parameter_change_l2.is_finite()
                    || m.training_parameter_change_l2 < 0.
                    || m.selected_update > self.config.updates_per_date
                    || !m.validation_lower_mean.is_finite()
                    || m.decision.is_some()
                        != (self.config.algorithm == StoppingAlgorithm::DirectStopping2019)
                {
                    return Err(invalid("inconsistent stopping date configuration"));
                }
                if let Some(d) = &m.decision {
                    d.validate()?;
                    if d.width != self.config.width {
                        return Err(invalid("decision width differs"));
                    }
                }
            }
        }
        if self.stop_initial && !self.config.spec.exercise[0] {
            return Err(invalid("initial stop outside exercise schedule"));
        }
        Ok(())
    }
    pub fn should_stop(&self, date: usize, spot: f64) -> Result<bool, ComputeError> {
        self.validate_state(date, spot)?;
        if !self.config.spec.exercise[date] {
            return Ok(false);
        }
        if date == 0 {
            return Ok(self.stop_initial);
        }
        if date + 1 == self.dates.len() {
            return Ok(true);
        }
        let m = self.dates[date]
            .as_ref()
            .ok_or_else(|| invalid("missing date network"))?;
        let x = self.config.spec.features(date, spot);
        Ok(match &m.decision {
            Some(d) => d.predict(x, true) >= 0.5,
            None => {
                self.config.spec.payoff(date, spot)
                    >= m.continuation.predict(x, false) * self.config.spec.strike
            }
        })
    }
    fn validate_state(&self, date: usize, spot: f64) -> Result<(), ComputeError> {
        if date >= self.dates.len() || !spot.is_finite() || spot <= 0. {
            return Err(invalid("invalid stopping state"));
        }
        Ok(())
    }
    fn proxy(&self, date: usize, spot: f64) -> Result<f64, ComputeError> {
        self.validate_state(date, spot)?;
        let s = &self.config.spec;
        let g = s.payoff(date, spot);
        if date + 1 == s.times.len() {
            return Ok(g);
        }
        let c = if let Some(m) = &self.dates[date] {
            m.continuation.predict(s.features(date, spot), false) * s.strike
        } else {
            0.
        };
        let v = g.max(c);
        if !v.is_finite() {
            return Err(invalid("nonfinite proxy"));
        }
        Ok(v)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoppedPayoff {
    pub step: usize,
    pub amount: f64,
    pub discounted_amount: f64,
}
/// Stops at the first permitted decision using only the observed path prefix.
pub fn stop_path(model: &StoppingCheckpoint, spots: &[f64]) -> Result<StoppedPayoff, ComputeError> {
    model.validate()?;
    if spots.len() != model.dates.len() {
        return Err(invalid("invalid path length/spot"));
    }
    for (n, s) in spots.iter().enumerate() {
        if model.should_stop(n, *s)? {
            return Ok(StoppedPayoff {
                step: n,
                amount: (model.config.spec.strike - s).max(0.),
                discounted_amount: model.config.spec.payoff(n, *s),
            });
        }
    }
    Err(invalid("missing terminal stopping decision"))
}
/// Domain-separated Philox4x32-10 counter streams; exact lognormal GBM transitions.
pub fn simulate_gbm_paths(
    spec: &GbmPutSpec,
    paths: usize,
    seed: u64,
    first_path: u64,
) -> Result<Vec<f64>, ComputeError> {
    simulate(spec, paths, seed, first_path, 3)
}
fn simulate(
    spec: &GbmPutSpec,
    paths: usize,
    seed: u64,
    first: u64,
    namespace: u32,
) -> Result<Vec<f64>, ComputeError> {
    spec.validate()?;
    if paths == 0 || paths > 1_000_000 || first.checked_add(paths as u64).is_none() {
        return Err(invalid("path count/range unsupported"));
    }
    let n = spec.times.len();
    let mut out = vec![0.; paths * n];
    for (p, row) in out.chunks_mut(n).enumerate() {
        row[0] = spec.spot;
        for t in 0..n - 1 {
            row[t + 1] = spec.step(t, row[t], normal(seed, first + p as u64, t, namespace, 0))?;
        }
    }
    Ok(out)
}
struct Trained {
    network: StoppingNetwork,
    step: usize,
    score: f64,
    change: f64,
}
fn tensor(s: &Session, x: Vec<f64>, shape: [usize; 2]) -> Tensor<2> {
    Tensor::from_data(
        TensorData::new(x, shape),
        TensorCreationOptions::new(s.device().clone()).with_dtype(
            if s.precision() == Precision::F64 {
                DType::F64
            } else {
                DType::F32
            },
        ),
    )
}
struct FitData<'a> {
    x: &'a [[f64; 2]],
    target: &'a [f64],
    immediate: &'a [f64],
    gains: Option<&'a [f64]>,
}
fn train_network(
    session: &Session,
    c: &StoppingTrainingConfig,
    date: usize,
    training: FitData<'_>,
    validation: FitData<'_>,
    direct: bool,
) -> Result<Trained, ComputeError> {
    let train_x = training.x;
    let target = training.target;
    let immediate = training.immediate;
    let val_x = validation.x;
    let val_target = validation.target;
    let val_immediate = validation.immediate;
    let mut net = StoppingNetwork::new(c.width, c.initialization_seed, date, direct);
    let w = c.width;
    let shapes = [[2, w], [1, w], [w, w], [1, w], [w, 1], [1, 1]];
    let mut parameters: Vec<Tensor<2>> = net
        .parameters
        .iter()
        .zip(shapes)
        .map(|(p, shape)| tensor(session, p.clone(), shape).autodiff().require_grad())
        .collect();
    let mut first: Vec<Tensor<2>> = parameters
        .iter()
        .map(|p| p.clone().without_autodiff() * 0.)
        .collect();
    let mut second = first.clone();
    let mut best = net.clone();
    let mut best_score = f64::NEG_INFINITY;
    let mut selected = 0;
    for update in 0..=c.updates_per_date {
        if update % c.validation_every == 0 || update == c.updates_per_date {
            for (i, p) in parameters.iter().enumerate() {
                net.parameters[i] = p
                    .clone()
                    .without_autodiff()
                    .try_to_vec_as::<f64>()
                    .map_err(|e| ComputeError::Execution(e.to_string()))?;
            }
            let score = val_x
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    if let Some(gains) = validation.gains {
                        return -(net.predict(*x, false) * gains[i] - val_target[i]).powi(2);
                    }
                    let stop = if direct {
                        net.predict(*x, true) >= 0.5
                    } else {
                        val_immediate[i] >= net.predict(*x, false)
                    };
                    if stop {
                        val_immediate[i]
                    } else {
                        val_target[i]
                    }
                })
                .sum::<f64>()
                / val_x.len() as f64;
            if score > best_score {
                best_score = score;
                best = net.clone();
                selected = update;
            }
        }
        if update == c.updates_per_date {
            break;
        }
        let indices: Vec<usize> = (0..c.batch_rows)
            .map(|row| {
                let r = philox(
                    [
                        row as u32,
                        (row as u64 >> 32) as u32,
                        date as u32,
                        update as u32,
                    ],
                    [c.training_seed as u32, (c.training_seed >> 32) as u32],
                );
                ((r[0] as u64 | ((r[1] as u64) << 32)) % train_x.len() as u64) as usize
            })
            .collect();
        let x = tensor(
            session,
            indices.iter().flat_map(|i| train_x[*i]).collect(),
            [c.batch_rows, 2],
        );
        let y = tensor(
            session,
            indices.iter().map(|i| target[*i]).collect(),
            [c.batch_rows, 1],
        );
        let g = tensor(
            session,
            indices.iter().map(|i| immediate[*i]).collect(),
            [c.batch_rows, 1],
        );
        let activate = |x: Tensor<2>| if direct { relu(x) } else { tanh(x) };
        let h = activate(x.matmul(parameters[0].clone()) + parameters[1].clone());
        let h = activate(h.matmul(parameters[2].clone()) + parameters[3].clone());
        let out = h.matmul(parameters[4].clone()) + parameters[5].clone();
        let loss = if let Some(gains) = training.gains {
            let dx = tensor(
                session,
                indices.iter().map(|i| gains[*i]).collect(),
                [c.batch_rows, 1],
            );
            (out * dx - y).square().mean()
        } else if direct {
            let p = sigmoid(out);
            {
                let complement: Tensor<2> = p.clone() * -1. + 1.;
                (p * g + complement * y).mean() * -1.
            }
        } else {
            (out - y).square().mean()
        };
        let value = loss
            .clone()
            .without_autodiff()
            .try_to_vec_as::<f64>()
            .map_err(|e| ComputeError::Execution(e.to_string()))?[0];
        if !value.is_finite() {
            return Err(invalid("nonfinite stopping training loss"));
        }
        let mut gradients = loss.backward();
        let step = (update + 1) as f64;
        for i in 0..parameters.len() {
            let grad = parameters[i]
                .grad_remove(&mut gradients)
                .ok_or_else(|| invalid("missing stopping gradient"))?
                .without_autodiff();
            first[i] = first[i].clone() * 0.9 + grad.clone() * 0.1;
            second[i] = second[i].clone() * 0.999 + grad.square() * 0.001;
            let delta = (first[i].clone() / (1. - 0.9f64.powf(step)))
                / ((second[i].clone() / (1. - 0.999f64.powf(step))).sqrt() + 1e-8)
                * c.learning_rate;
            parameters[i] = (parameters[i].clone().without_autodiff() - delta)
                .autodiff()
                .require_grad();
        }
    }
    best.validate()?;
    let initial = StoppingNetwork::new(c.width, c.initialization_seed, date, direct);
    let change = net
        .parameters
        .iter()
        .flatten()
        .zip(initial.parameters.iter().flatten())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(Trained {
        network: best,
        step: selected,
        score: best_score,
        change,
    })
}
/// Backward training uses frozen future decisions and disjoint validation paths.
/// This bounded implementation uses fixed path pools and omits paper batchnorm.
pub fn train(c: &StoppingTrainingConfig) -> Result<StoppingCheckpoint, ComputeError> {
    c.validate()?;
    let session = Session::new(c.target, c.precision, c.batch_rows)?;
    let s = &c.spec;
    let n = s.times.len();
    let paths = simulate(s, c.training_paths, c.training_seed, 0, 1)?;
    let val = simulate(s, c.validation_paths, c.validation_seed, 0, 2)?;
    let mut y: Vec<f64> = paths
        .chunks(n)
        .map(|p| s.payoff(n - 1, p[n - 1]) / s.strike)
        .collect();
    let mut vy: Vec<f64> = val
        .chunks(n)
        .map(|p| s.payoff(n - 1, p[n - 1]) / s.strike)
        .collect();
    let mut dates = vec![None; n];
    for date in (1..n - 1).rev() {
        if !s.exercise[date] {
            continue;
        }
        let x: Vec<_> = paths.chunks(n).map(|p| s.features(date, p[date])).collect();
        let vx: Vec<_> = val.chunks(n).map(|p| s.features(date, p[date])).collect();
        let g: Vec<_> = paths
            .chunks(n)
            .map(|p| s.payoff(date, p[date]) / s.strike)
            .collect();
        let vg: Vec<_> = val
            .chunks(n)
            .map(|p| s.payoff(date, p[date]) / s.strike)
            .collect();
        let continuation = train_network(
            &session,
            c,
            date,
            FitData {
                x: &x,
                target: &y,
                immediate: &g,
                gains: None,
            },
            FitData {
                x: &vx,
                target: &vy,
                immediate: &vg,
                gains: None,
            },
            false,
        )?;
        let decision = if c.algorithm == StoppingAlgorithm::DirectStopping2019 {
            Some(train_network(
                &session,
                c,
                date,
                FitData {
                    x: &x,
                    target: &y,
                    immediate: &g,
                    gains: None,
                },
                FitData {
                    x: &vx,
                    target: &vy,
                    immediate: &vg,
                    gains: None,
                },
                true,
            )?)
        } else {
            None
        };
        let selected = decision.as_ref().unwrap_or(&continuation);
        let model = StoppingDateModel {
            continuation: continuation.network.clone(),
            decision: decision.as_ref().map(|d| d.network.clone()),
            selected_update: selected.step,
            completed_updates: c.updates_per_date,
            training_parameter_change_l2: selected.change,
            validation_lower_mean: selected.score * s.strike,
        };
        let stops = |x: [f64; 2], g: f64| {
            if let Some(d) = &model.decision {
                d.predict(x, true) >= 0.5
            } else {
                g >= model.continuation.predict(x, false)
            }
        };
        for i in 0..y.len() {
            if stops(x[i], g[i]) {
                y[i] = g[i];
            }
        }
        for i in 0..vy.len() {
            if stops(vx[i], vg[i]) {
                vy[i] = vg[i];
            }
        }
        dates[date] = Some(model);
    }
    let initial = y.iter().sum::<f64>() / y.len() as f64 * s.strike;
    let model = StoppingCheckpoint {
        schema: SCHEMA.into(),
        config: c.clone(),
        dates,
        stop_initial: s.exercise[0] && s.payoff(0, s.spot) >= initial,
        initial_training_continuation: initial,
    };
    model.validate()?;
    Ok(model)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoppingEvaluationConfig {
    pub lower_paths: usize,
    pub upper_paths: usize,
    pub inner_paths: usize,
    pub lower_seed: u64,
    pub upper_seed: u64,
    pub first_path: u64,
    pub max_proxy_evaluations: u64,
}
impl Default for StoppingEvaluationConfig {
    fn default() -> Self {
        Self {
            lower_paths: 65536,
            upper_paths: 4096,
            inner_paths: 64,
            lower_seed: 10040001,
            upper_seed: 10050001,
            first_path: 0,
            max_proxy_evaluations: 100_000_000,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanEstimate {
    pub mean: f64,
    pub standard_error: f64,
    pub paths: usize,
    pub asymptotic_95: [f64; 2],
}
fn estimate(x: &[f64]) -> Result<MeanEstimate, ComputeError> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let variance = x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se = (variance / n as f64).sqrt();
    if !mean.is_finite() || !variance.is_finite() || !se.is_finite() {
        return Err(invalid("nonfinite Monte Carlo statistics"));
    }
    Ok(MeanEstimate {
        mean,
        standard_error: se,
        paths: n,
        asymptotic_95: [mean - 1.959963984540054 * se, mean + 1.959963984540054 * se],
    })
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoppingBounds {
    pub evaluation: StoppingEvaluationConfig,
    pub lower_market_sha256: String,
    pub upper_market_sha256: String,
    pub lower: MeanEstimate,
    pub upper: MeanEstimate,
    pub midpoint: f64,
    pub asymptotic_joint_95_enclosure: [f64; 2],
    pub hoeffding_joint_95_enclosure: [f64; 2],
    pub mean_martingale_increment: Vec<f64>,
    pub exercise_counts: Vec<usize>,
    pub lower_samples: Vec<f64>,
    pub upper_samples: Vec<f64>,
    pub estimator: String,
}
/// Independent lower/upper cohorts. Upper means estimate a true expectation
/// upper bound; neither an individual sample nor a sample mean is a sure bound.
pub fn evaluate_bounds(
    model: &StoppingCheckpoint,
    c: &StoppingEvaluationConfig,
) -> Result<StoppingBounds, ComputeError> {
    model.validate()?;
    let s = &model.config.spec;
    let n = s.times.len();
    let unique = [
        model.config.training_seed,
        model.config.validation_seed,
        c.lower_seed,
        c.upper_seed,
    ];
    if c.lower_paths < 2
        || c.upper_paths < 2
        || c.inner_paths == 0
        || c.inner_paths > 4096
        || c.lower_paths > 1_000_000
        || c.upper_paths > 1_000_000
        || unique
            .iter()
            .enumerate()
            .any(|(i, x)| unique[..i].contains(x))
        || c.first_path
            .checked_add(c.lower_paths.max(c.upper_paths) as u64)
            .is_none()
    {
        return Err(invalid(
            "invalid evaluation counts/ranges or nonindependent seeds",
        ));
    }
    let workspace =
        (c.lower_paths as u128 * n as u128 + c.lower_paths as u128 + c.upper_paths as u128) * 8
            + 16 * 1024 * 1024;
    if workspace > model.config.max_working_bytes as u128 {
        return Err(ComputeError::BudgetExceeded(
            "stopping evaluation workspace exceeds model allowance".into(),
        ));
    }
    let work = c.upper_paths as u128 * (n - 1) as u128 * (c.inner_paths + 1) as u128;
    if work > c.max_proxy_evaluations as u128 {
        return Err(ComputeError::BudgetExceeded(
            "nested proxy evaluation count exceeds budget".into(),
        ));
    }
    let lower_paths = simulate(s, c.lower_paths, c.lower_seed, c.first_path, 3)?;
    let mut lower = Vec::with_capacity(c.lower_paths);
    let mut counts = vec![0; n];
    for row in lower_paths.chunks(n) {
        let stopped = stop_path(model, row)?;
        counts[stopped.step] += 1;
        lower.push(stopped.discounted_amount);
    }
    let mut lower_hash = Sha256::new();
    for x in &lower_paths {
        lower_hash.update(x.to_le_bytes());
    }
    let mut upper_hash = Sha256::new();
    let mut upper = Vec::with_capacity(c.upper_paths);
    let mut increments = vec![0.; n - 1];
    for path in 0..c.upper_paths {
        let id = c.first_path + path as u64;
        let mut spot = s.spot;
        upper_hash.update(spot.to_le_bytes());
        let mut martingale = 0.;
        let mut maximum = if s.exercise[0] {
            s.payoff(0, spot)
        } else {
            f64::NEG_INFINITY
        };
        for (date, total_increment) in increments.iter_mut().enumerate() {
            let next = s.step(date, spot, normal(c.upper_seed, id, date, 4, 0))?;
            upper_hash.update(next.to_le_bytes());
            let realized = model.proxy(date + 1, next)?;
            let mut conditional = 0.;
            for k in 0..c.inner_paths {
                let inner = s.step(date, spot, normal(c.upper_seed, id, date, 5, k as u32))?;
                conditional += model.proxy(date + 1, inner)?;
            }
            let increment = realized - conditional / c.inner_paths as f64;
            martingale += increment;
            *total_increment += increment;
            if s.exercise[date + 1] {
                maximum = maximum.max(s.payoff(date + 1, next) - martingale);
            }
            spot = next;
        }
        if !maximum.is_finite() {
            return Err(invalid("nonfinite dual sample"));
        }
        upper.push(maximum);
    }
    let l = estimate(&lower)?;
    let u = estimate(&upper)?;
    let payoff_cap = s.payoff_cap();
    let sum_proxy: f64 = (1..n)
        .map(|date| {
            if date + 1 == n {
                payoff_cap
            } else {
                model.dates[date]
                    .as_ref()
                    .map(|m| payoff_cap.max(m.continuation.output_bound() * s.strike))
                    .unwrap_or(payoff_cap)
            }
        })
        .sum();
    let lower_radius = payoff_cap * ((40f64).ln() / (2. * c.lower_paths as f64)).sqrt();
    let upper_radius =
        (payoff_cap + 2. * sum_proxy) * ((40f64).ln() / (2. * c.upper_paths as f64)).sqrt();
    let out=StoppingBounds{evaluation:c.clone(),lower_market_sha256:format!("{:x}",lower_hash.finalize()),upper_market_sha256:format!("{:x}",upper_hash.finalize()),midpoint:0.5*(l.mean+u.mean),asymptotic_joint_95_enclosure:[l.mean-1.959963984540054*l.standard_error,u.mean+1.959963984540054*u.standard_error],hoeffding_joint_95_enclosure:[l.mean-lower_radius,u.mean+upper_radius],lower:l,upper:u,mean_martingale_increment:increments.iter().map(|x|x/c.upper_paths as f64).collect(),exercise_counts:counts,lower_samples:lower,upper_samples:upper,estimator:"Frozen deterministic max(payoff,continuation) proxy; fresh conditional one-step Philox inner draws; enlarged-filtration martingale; alternate to paper policy-tail dual. Confidence assumes independent idealized RNG draws; CLT enclosure asymptotic, Hoeffding conservative finite-sample.".into()};
    Ok(out)
}
fn normal(seed: u64, path: u64, date: usize, namespace: u32, draw: u32) -> f64 {
    let x = philox(
        [
            path as u32,
            (path >> 32) as u32,
            date as u32,
            (namespace << 28) | draw,
        ],
        [seed as u32, (seed >> 32) as u32],
    );
    let open = |a: u32, b: u32| {
        let bits = ((a as u64) << 32) | (b as u64);
        ((bits >> 12) as f64 + 0.5) * (1. / 4503599627370496.)
    };
    let u = open(x[0], x[1]);
    let v = open(x[2], x[3]);
    (-2. * u.ln()).sqrt() * (2. * std::f64::consts::PI * v).cos()
}
fn philox(mut c: [u32; 4], mut k: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let a = (c[0] as u64) * 0xD2511F53;
        let b = (c[2] as u64) * 0xCD9E8D57;
        c = [
            (b >> 32) as u32 ^ c[1] ^ k[0],
            b as u32,
            (a >> 32) as u32 ^ c[3] ^ k[1],
            a as u32,
        ];
        k[0] = k[0].wrapping_add(0x9E3779B9);
        k[1] = k[1].wrapping_add(0xBB67AE85);
    }
    c
}

/// Section 4.2 interval hedge: a different regression from full-path hedging.
/// Holdings refer to the discounted dividend-reinvested stock P=S*exp((q-r)t).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntervalHedgeCheckpoint {
    pub schema: String,
    pub holder: StoppingCheckpoint,
    pub config: StoppingTrainingConfig,
    pub initial_price: f64,
    pub networks: Vec<StoppingNetwork>,
    pub validation_interval_mse: Vec<f64>,
}
impl IntervalHedgeCheckpoint {
    pub fn validate(&self) -> Result<(), ComputeError> {
        self.holder.validate()?;
        self.config.validate()?;
        if self.config.spec.exercise[1..].iter().any(|x| !*x)
            || self.config.algorithm != StoppingAlgorithm::ContinuationRegression2020
            || self.holder.config.algorithm != StoppingAlgorithm::ContinuationRegression2020
            || self.schema != "volt.compute.stopping_interval_hedge.v1"
            || serde_json::to_value(&self.holder.config.spec).ok()
                != serde_json::to_value(&self.config.spec).ok()
            || self.networks.len() + 1 != self.config.spec.times.len()
            || self.validation_interval_mse.len() != self.networks.len()
            || !self.initial_price.is_finite()
            || self.initial_price.abs() > 1e12
            || self
                .validation_interval_mse
                .iter()
                .any(|x| !x.is_finite() || *x < 0.)
        {
            return Err(invalid("invalid interval hedge checkpoint"));
        }
        for network in &self.networks {
            network.validate()?;
        }
        Ok(())
    }
}
fn continuation(model: &StoppingCheckpoint, date: usize, spot: f64) -> f64 {
    if date == 0 {
        model.initial_training_continuation.max(0.)
    } else {
        model.dates[date]
            .as_ref()
            .map(|m| {
                (m.continuation
                    .predict(model.config.spec.features(date, spot), false)
                    * model.config.spec.strike)
                    .max(0.)
            })
            .unwrap_or(0.)
    }
}
/// Trains h_n by MSE of C_n + h_n*(P_{n+1}-P_n) - max(g_{n+1},c_{n+1}).
/// This first version has one hedge decision between consecutive exercise dates.
pub fn train_interval_hedge(
    holder: &StoppingCheckpoint,
    c: &StoppingTrainingConfig,
    initial_price: f64,
) -> Result<IntervalHedgeCheckpoint, ComputeError> {
    holder.validate()?;
    c.validate()?;
    if holder.config.algorithm != StoppingAlgorithm::ContinuationRegression2020
        || c.algorithm != StoppingAlgorithm::ContinuationRegression2020
        || c.spec.exercise[1..].iter().any(|x| !*x)
    {
        return Err(invalid(
            "interval method requires continuation holder and each interior grid date exercisable",
        ));
    }
    let seeds = [
        holder.config.training_seed,
        holder.config.validation_seed,
        c.training_seed,
        c.validation_seed,
    ];
    if seeds
        .iter()
        .enumerate()
        .any(|(i, x)| seeds[..i].contains(x))
        || serde_json::to_value(&holder.config.spec).ok() != serde_json::to_value(&c.spec).ok()
        || !initial_price.is_finite()
        || initial_price.abs() > 1e12
    {
        return Err(invalid(
            "interval hedge market/seeds/capital differ or are invalid",
        ));
    }
    let session = Session::new(c.target, c.precision, c.batch_rows)?;
    let spec = &c.spec;
    let n = spec.times.len();
    let train = simulate(spec, c.training_paths, c.training_seed, 0, 9)?;
    let val = simulate(spec, c.validation_paths, c.validation_seed, 0, 10)?;
    let mut networks = Vec::new();
    let mut metrics = Vec::new();
    for date in 0..n - 1 {
        let prepare = |paths: &[f64]| -> Result<_, ComputeError> {
            let mut x = Vec::new();
            let mut y = Vec::new();
            let mut gains = Vec::new();
            for row in paths.chunks(n) {
                x.push(spec.features(date, row[date]));
                y.push(
                    (holder.proxy(date + 1, row[date + 1])?
                        - continuation(holder, date, row[date]))
                        / spec.strike,
                );
                gains.push(
                    (row[date + 1]
                        * ((spec.dividend_yield - spec.rate) * spec.times[date + 1]).exp()
                        - row[date] * ((spec.dividend_yield - spec.rate) * spec.times[date]).exp())
                        / spec.strike,
                );
            }
            Ok((x, y, gains))
        };
        let (x, y, dx) = prepare(&train)?;
        let (vx, vy, vdx) = prepare(&val)?;
        let fitted = train_network(
            &session,
            c,
            date,
            FitData {
                x: &x,
                target: &y,
                immediate: &y,
                gains: Some(&dx),
            },
            FitData {
                x: &vx,
                target: &vy,
                immediate: &vy,
                gains: Some(&vdx),
            },
            false,
        )?;
        metrics.push(-fitted.score * spec.strike.powi(2));
        networks.push(fitted.network);
    }
    let out = IntervalHedgeCheckpoint {
        schema: "volt.compute.stopping_interval_hedge.v1".into(),
        holder: holder.clone(),
        config: c.clone(),
        initial_price,
        networks,
        validation_interval_mse: metrics,
    };
    out.validate()?;
    Ok(out)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalHedgeReport {
    pub discounted_error: MeanEstimate,
    pub mse: f64,
    pub mean_shortfall: f64,
    pub zero_hedge_mse: f64,
    pub exercise_counts: Vec<usize>,
    pub errors: Vec<f64>,
    pub scope: String,
}
/// Fresh full paths, trading only before the fixed holder's actual stopping time.
pub fn evaluate_interval_hedge(
    model: &IntervalHedgeCheckpoint,
    paths: usize,
    seed: u64,
    first_path: u64,
) -> Result<IntervalHedgeReport, ComputeError> {
    model.validate()?;
    if [
        model.holder.config.training_seed,
        model.holder.config.validation_seed,
        model.config.training_seed,
        model.config.validation_seed,
    ]
    .contains(&seed)
        || paths < 2
    {
        return Err(invalid("interval hedge evaluation seed/count invalid"));
    }
    let spec = &model.config.spec;
    let n = spec.times.len();
    let workspace = paths as u128 * (n as u128 + 1) * 8 + 16 * 1024 * 1024;
    if workspace > model.config.max_working_bytes as u128 {
        return Err(ComputeError::BudgetExceeded(
            "interval hedge evaluation workspace exceeds model allowance".into(),
        ));
    }
    let data = simulate(spec, paths, seed, first_path, 11)?;
    let mut errors = Vec::with_capacity(paths);
    let mut zero = 0.;
    let mut counts = vec![0; n];
    for row in data.chunks(n) {
        let stopped = stop_path(&model.holder, row)?;
        counts[stopped.step] += 1;
        let mut value = model.initial_price;
        for date in 0..stopped.step {
            let h = model.networks[date].predict(spec.features(date, row[date]), false);
            let p0 = row[date] * ((spec.dividend_yield - spec.rate) * spec.times[date]).exp();
            let p1 =
                row[date + 1] * ((spec.dividend_yield - spec.rate) * spec.times[date + 1]).exp();
            value += h * (p1 - p0);
        }
        let error = value - stopped.discounted_amount;
        if !error.is_finite() {
            return Err(invalid("nonfinite stopped hedge error"));
        }
        zero += (model.initial_price - stopped.discounted_amount).powi(2);
        errors.push(error);
    }
    let estimate = estimate(&errors)?;
    let mse = errors.iter().map(|x| x * x).sum::<f64>() / paths as f64;
    let shortfall = errors.iter().map(|x| (-x).max(0.)).sum::<f64>() / paths as f64;
    if !mse.is_finite() || !shortfall.is_finite() || !zero.is_finite() {
        return Err(invalid("nonfinite hedge statistics"));
    }
    Ok(IntervalHedgeReport{discounted_error:estimate,mse,mean_shortfall:shortfall,zero_hedge_mse:zero/paths as f64,exercise_counts:counts,errors,scope:"2020 Sec4.2 interval continuation/value regression, one rebalance per exercise interval, discounted reinvested stock, no transaction fees; stopped at fixed learned holder. Distinct from full-path native DeepHedging writer.".into()})
}
