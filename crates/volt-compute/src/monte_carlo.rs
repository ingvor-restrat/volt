//! European GBM Monte Carlo with pathwise reverse AD and likelihood-ratio gamma.
//!
//! First derivatives propagate through the discounted terminal payoff. Its kink
//! has probability zero under GBM, permitting pathwise first derivatives, but
//! naively differentiating its indicator again would miss gamma. Instead we use
//! payoff times f_SS/f, differentiating the lognormal density at fixed terminal
//! asset value: score = z/(S sigma sqrt(T)), and
//! f_SS/f = (z²-1)/(S² sigma² T) - z/(S² sigma sqrt(T)).
//! See Broadie & Glasserman (1996), *Estimating Security Price Derivatives Using
//! Simulation*, Proposition 4: <https://www.columbia.edu/~mnb2/broadie/Assets/bg_ms_1996.pdf>
//! An antithetic pair, not each correlated path, is one observation for errors.
//! Standard errors describe numerical estimator uncertainty. Shared normal draws
//! are common random numbers for covariance-aware comparison and aggregation,
//! not a calibrated multiasset market correlation or P&L risk model.

use crate::{
    ComputeError, OptionRow, OptionSide, Precision, Session,
    chunked::{CompensatedSum, validate_job},
};
use burn::tensor::{DType, Gradients, Tensor, TensorCreationOptions, TensorData, activation::relu};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Estimate {
    pub mean: f64,
    pub standard_error: Option<f64>,
    pub samples: u64,
}

/// Mergeable central moments of independent pair averages. Compensation guards
/// accumulation of small increments; m2 is the sum of squared deviations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OnlineMoments {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    #[serde(skip)]
    mean_correction: f64,
    #[serde(skip)]
    m2_correction: f64,
}

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let adjusted = value - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

impl OnlineMoments {
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        compensated_add(
            &mut self.mean,
            &mut self.mean_correction,
            delta / self.count as f64,
        );
        compensated_add(
            &mut self.m2,
            &mut self.m2_correction,
            delta * (value - self.mean),
        );
    }

    pub fn merge(&mut self, rhs: &Self) {
        if rhs.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = rhs.clone();
            return;
        }
        let count = self.count + rhs.count;
        let delta = rhs.mean - self.mean;
        let cross = delta * delta * self.count as f64 * (rhs.count as f64 / count as f64);
        compensated_add(&mut self.m2, &mut self.m2_correction, rhs.m2);
        compensated_add(&mut self.m2, &mut self.m2_correction, cross);
        compensated_add(
            &mut self.mean,
            &mut self.mean_correction,
            delta * (rhs.count as f64 / count as f64),
        );
        self.count = count;
    }

    pub fn estimate(&self) -> Estimate {
        Estimate {
            mean: self.mean,
            samples: self.count,
            standard_error: (self.count >= 2)
                .then(|| (self.m2.max(0.0) / (self.count - 1) as f64 / self.count as f64).sqrt()),
        }
    }
}

/// Annual-fraction vega/rho/dividend derivatives; calendar theta = -dP/dT.
/// All row-level observations are per underlying unit, before position weights.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GreekMoments {
    pub price: OnlineMoments,
    pub delta: OnlineMoments,
    pub vega: OnlineMoments,
    pub rho: OnlineMoments,
    pub theta: OnlineMoments,
    pub dividend_sensitivity: OnlineMoments,
    pub gamma: OnlineMoments,
}

impl GreekMoments {
    pub fn merge(&mut self, rhs: &Self) {
        self.price.merge(&rhs.price);
        self.delta.merge(&rhs.delta);
        self.vega.merge(&rhs.vega);
        self.rho.merge(&rhs.rho);
        self.theta.merge(&rhs.theta);
        self.dividend_sensitivity.merge(&rhs.dividend_sensitivity);
        self.gamma.merge(&rhs.gamma);
    }
    fn add(&mut self, values: [f64; 7]) {
        self.price.add(values[0]);
        self.delta.add(values[1]);
        self.vega.add(values[2]);
        self.rho.add(values[3]);
        self.theta.add(values[4]);
        self.dividend_sensitivity.add(values[5]);
        self.gamma.add(values[6]);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloRow {
    pub row_id: String,
    pub moments: GreekMoments,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    pub rows: Vec<MonteCarloRow>,
    /// Signed quantity*multiplier aggregation within each shared pair, preserving
    /// cross-row covariance before estimating standard errors.
    pub portfolio: GreekMoments,
    pub pair_count: usize,
    pub max_sampled_allocator_live_bytes: Option<u64>,
    pub max_sampled_allocator_reserved_bytes: Option<u64>,
}

/// Stateless indexed standard normal, so changing batch boundaries does not
/// change paths. SplitMix64 counter mixing followed by Box–Muller; IEEE math
/// implementations may differ by a few last bits across host platforms.
/// Two counters are consumed per pair; unique stream indices are 0..2^63.
pub fn normal_at(seed: u64, pair_index: u64) -> f64 {
    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    let counter = pair_index
        .wrapping_mul(2)
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(seed);
    let uniform = |bits: u64| ((bits >> 12) as f64 + 0.5) / 4_503_599_627_370_496.0;
    let u1 = uniform(mix(counter.wrapping_add(0x9e3779b97f4a7c15)));
    let u2 = uniform(mix(counter.wrapping_add(0x3c6ef372fe94f82a)));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Low-level shared-normal batch. The expanded row*pair*2 tensor length is
/// bounded by the session cap. The caller owns job admission and pair chunking.
pub fn monte_carlo_rows(
    session: &Session,
    rows: &[OptionRow],
    pair_normals: &[f64],
) -> Result<MonteCarloResult, ComputeError> {
    let length = rows
        .len()
        .checked_mul(pair_normals.len())
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| ComputeError::InvalidInput("Monte Carlo batch length overflow".into()))?;
    if length == 0 || length > session.max_batch_rows() {
        return Err(ComputeError::InvalidInput(format!(
            "Monte Carlo expanded batch has {length} paths; allowed range 1..={}",
            session.max_batch_rows()
        )));
    }
    validate_job(rows, session.precision())?;
    if pair_normals.iter().any(|z| {
        !z.is_finite() || (session.precision() == Precision::F32 && !(*z as f32).is_finite())
    }) {
        return Err(ComputeError::InvalidInput(
            "normal draws must be finite at device precision".into(),
        ));
    }
    let device = session.device().clone().autodiff();
    let dtype = match session.precision() {
        Precision::F32 => DType::F32,
        Precision::F64 => DType::F64,
    };
    let tensor = |values: Vec<f64>| {
        Tensor::<1>::from_data(
            TensorData::new(values, [length]),
            TensorCreationOptions::new(device.clone()).with_dtype(dtype),
        )
    };
    let vector = |get: fn(&OptionRow) -> f64| {
        tensor(
            rows.iter()
                .flat_map(|row| std::iter::repeat_n(get(row), pair_normals.len() * 2))
                .collect(),
        )
    };
    let spot = vector(|r| r.spot).require_grad();
    let strike = vector(|r| r.strike);
    let time = vector(|r| r.maturity_years).require_grad();
    let volatility = vector(|r| r.volatility).require_grad();
    let rate = vector(|r| r.rate).require_grad();
    let dividend = vector(|r| r.dividend_yield).require_grad();
    let sign = vector(|r| match r.side {
        OptionSide::Call => 1.0,
        OptionSide::Put => -1.0,
    });
    let z = tensor(
        (0..rows.len())
            .flat_map(|_| pair_normals.iter().flat_map(|z| [*z, -*z]))
            .collect(),
    );
    let sigma_t = volatility.clone() * time.clone().sqrt();
    let terminal = spot.clone()
        * ((rate.clone() - dividend.clone() - volatility.clone() * volatility.clone() * 0.5)
            * time.clone()
            + sigma_t.clone() * z.clone())
        .exp();
    let payoff = relu(sign * (terminal - strike)) * (-rate.clone() * time.clone()).exp();
    // Differentiate the density at fixed terminal asset, not the kinked payoff.
    let base_spot = spot.clone().without_autodiff();
    let base_sigma_t = sigma_t.without_autodiff();
    let base_z = z.without_autodiff();
    let square_spot = base_spot.clone() * base_spot;
    let lr_weight = (base_z.clone() * base_z.clone() - 1.0)
        / (square_spot.clone() * base_sigma_t.clone() * base_sigma_t.clone())
        - base_z / (square_spot * base_sigma_t);
    let gamma_tensor = payoff.clone().without_autodiff() * lr_weight;
    session.synchronize()?;
    let mut telemetry = Telemetry::default();
    telemetry.sample(session);
    let prices = read(session, &payoff, length, "price")?;
    let gamma = read(session, &gamma_tensor, length, "likelihood-ratio gamma")?;
    let mut gradients = payoff.sum().backward();
    session.synchronize()?;
    telemetry.sample(session);
    let delta = gradient(session, &spot, &mut gradients, length, "delta")?;
    let vega = gradient(session, &volatility, &mut gradients, length, "vega")?;
    let rho = gradient(session, &rate, &mut gradients, length, "rho")?;
    let mut theta = gradient(session, &time, &mut gradients, length, "time")?;
    for value in &mut theta {
        *value = -*value;
    }
    let dividend_sensitivity = gradient(session, &dividend, &mut gradients, length, "dividend")?;
    telemetry.sample(session);
    let values = [
        &prices,
        &delta,
        &vega,
        &rho,
        &theta,
        &dividend_sensitivity,
        &gamma,
    ];
    let mut result_rows: Vec<_> = rows
        .iter()
        .map(|row| MonteCarloRow {
            row_id: row.row_id.clone(),
            moments: GreekMoments::default(),
        })
        .collect();
    let mut portfolio = GreekMoments::default();
    for pair in 0..pair_normals.len() {
        let mut portfolio_sums: [CompensatedSum; 7] =
            std::array::from_fn(|_| CompensatedSum::default());
        for (i, row) in rows.iter().enumerate() {
            let offset = (i * pair_normals.len() + pair) * 2;
            let pair_values =
                std::array::from_fn(|k| 0.5 * values[k][offset] + 0.5 * values[k][offset + 1]);
            result_rows[i].moments.add(pair_values);
            let weight = row.quantity as f64 * row.multiplier as f64;
            for k in 0..7 {
                let weighted = pair_values[k] * weight;
                if !weighted.is_finite() {
                    return Err(ComputeError::Execution(
                        "Monte Carlo weighted observation overflow".into(),
                    ));
                }
                portfolio_sums[k].add(weighted);
            }
        }
        let portfolio_values = portfolio_sums.map(CompensatedSum::value);
        if portfolio_values.iter().any(|x| !x.is_finite()) {
            return Err(ComputeError::Execution(
                "Monte Carlo portfolio overflow".into(),
            ));
        }
        portfolio.add(portfolio_values);
    }
    for moments in result_rows
        .iter()
        .map(|row| &row.moments)
        .chain(std::iter::once(&portfolio))
    {
        for moment in [
            &moments.price,
            &moments.delta,
            &moments.vega,
            &moments.rho,
            &moments.theta,
            &moments.dividend_sensitivity,
            &moments.gamma,
        ] {
            if !moment.mean.is_finite() || !moment.m2.is_finite() || moment.m2 < 0.0 {
                return Err(ComputeError::Execution(
                    "Monte Carlo pair moments overflow".into(),
                ));
            }
        }
    }
    Ok(MonteCarloResult {
        rows: result_rows,
        portfolio,
        pair_count: pair_normals.len(),
        max_sampled_allocator_live_bytes: telemetry.live,
        max_sampled_allocator_reserved_bytes: telemetry.reserved,
    })
}

fn read(
    session: &Session,
    tensor: &Tensor<1>,
    length: usize,
    name: &str,
) -> Result<Vec<f64>, ComputeError> {
    let values = session.read_vector(&tensor.clone().without_autodiff())?;
    if values.len() != length || values.iter().any(|x| !x.is_finite()) {
        return Err(ComputeError::Execution(format!(
            "nonfinite or malformed Monte Carlo {name}"
        )));
    }
    Ok(values)
}
fn gradient(
    session: &Session,
    leaf: &Tensor<1>,
    gradients: &mut Gradients,
    length: usize,
    name: &str,
) -> Result<Vec<f64>, ComputeError> {
    let value = leaf
        .grad_remove(gradients)
        .ok_or_else(|| ComputeError::Execution(format!("missing Monte Carlo {name} gradient")))?;
    read(session, &value, length, name)
}
#[derive(Default)]
struct Telemetry {
    live: Option<u64>,
    reserved: Option<u64>,
}
impl Telemetry {
    fn sample(&mut self, session: &Session) {
        if let Some(usage) = session.device().memory_pool_usage() {
            self.live = Some(self.live.unwrap_or(0).max(usage.bytes_in_use));
            self.reserved = Some(self.reserved.unwrap_or(0).max(usage.bytes_reserved));
        }
    }
}
