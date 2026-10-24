//! Second-order forward-mode automatic differentiation in the spot direction.
//!
//! A jet carries the value and its first two derivatives through every pricing
//! operation on the selected device. This is differentiation of the smooth
//! European Black–Scholes pricing graph, not a Monte Carlo estimator for the
//! gamma of a nonsmooth terminal payoff, and not reverse-over-reverse AAD.

use crate::{ComputeError, OptionRow, OptionSide, Session, chunked::validate_job};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Per-unit prices, dP/dS and d²P/dS², in input-row order. Position quantities
/// and contract multipliers are deliberately not applied here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondOrderResult {
    pub prices: Vec<f64>,
    pub delta: Vec<f64>,
    pub gamma: Vec<f64>,
    /// Sampled allocations, not a guarantee of instantaneous peak usage.
    pub max_sampled_allocator_live_bytes: Option<u64>,
    pub max_sampled_allocator_reserved_bytes: Option<u64>,
}

#[derive(Clone)]
struct Jet<const D: usize> {
    value: Tensor<D>,
    first: Tensor<D>,
    second: Tensor<D>,
}

impl<const D: usize> Jet<D> {
    fn constant(value: Tensor<D>) -> Self {
        Self {
            first: value.zeros_like(),
            second: value.zeros_like(),
            value,
        }
    }

    fn variable(value: Tensor<D>) -> Self {
        Self {
            first: value.ones_like(),
            second: value.zeros_like(),
            value,
        }
    }

    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            first: self.first + rhs.first,
            second: self.second + rhs.second,
        }
    }

    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            first: self.first - rhs.first,
            second: self.second - rhs.second,
        }
    }

    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value.clone() * rhs.value.clone(),
            first: self.first.clone() * rhs.value.clone() + self.value.clone() * rhs.first.clone(),
            second: self.second * rhs.value
                + self.first * rhs.first * 2.0
                + self.value * rhs.second,
        }
    }

    fn scale(self, scalar: f64) -> Self {
        Self {
            value: self.value * scalar,
            first: self.first * scalar,
            second: self.second * scalar,
        }
    }

    /// Chain rule: (f(x))'' = f''(x) x'² + f'(x) x''.
    fn unary(self, value: Tensor<D>, slope: Tensor<D>, curvature: Tensor<D>) -> Self {
        Self {
            value,
            first: slope.clone() * self.first.clone(),
            second: curvature * self.first.clone() * self.first + slope * self.second,
        }
    }

    fn reciprocal(self) -> Self {
        let inverse = self.value.ones_like() / self.value.clone();
        let square = inverse.clone() * inverse.clone();
        self.unary(inverse.clone(), -square.clone(), square * inverse * 2.0)
    }

    fn div(self, rhs: Self) -> Self {
        self.mul(rhs.reciprocal())
    }

    fn log(self) -> Self {
        let value = self.value.clone().log();
        let inverse = self.value.ones_like() / self.value.clone();
        self.unary(value, inverse.clone(), -(inverse.clone() * inverse))
    }

    fn exp(self) -> Self {
        let value = self.value.clone().exp();
        self.unary(value.clone(), value.clone(), value)
    }

    fn sqrt(self) -> Self {
        let value = self.value.clone().sqrt();
        let slope = value.ones_like() / value.clone() * 0.5;
        let curvature = value.ones_like() / (self.value.clone() * value.clone()) * -0.25;
        self.unary(value, slope, curvature)
    }

    fn normal_cdf(self) -> Self {
        let value = (self
            .value
            .clone()
            .mul_scalar(std::f64::consts::FRAC_1_SQRT_2)
            .erf()
            + 1.0)
            * 0.5;
        let density = (self.value.clone() * self.value.clone() * -0.5).exp()
            / (2.0 * std::f64::consts::PI).sqrt();
        let curvature = -self.value.clone() * density.clone();
        self.unary(value, density, curvature)
    }
}

/// Independent scalar Black–Scholes gamma for validation of the tensor AD
/// result. This formula is never used to produce the device derivative.
pub fn analytic_gamma(row: &OptionRow) -> Result<f64, ComputeError> {
    row.validate()?;
    let root_time = row.maturity_years.sqrt();
    let d1 = ((row.spot / row.strike).ln()
        + (row.rate - row.dividend_yield + 0.5 * row.volatility.powi(2)) * row.maturity_years)
        / (row.volatility * root_time);
    let result = (-row.dividend_yield * row.maturity_years - 0.5 * d1 * d1).exp()
        / ((2.0 * std::f64::consts::PI).sqrt() * row.spot * row.volatility * root_time);
    if !result.is_finite() {
        return Err(ComputeError::InvalidInput(format!(
            "{}: analytic gamma overflow",
            row.row_id
        )));
    }
    Ok(result)
}

/// Low-level resident batch; the caller owns admission, job serialization and
/// chunking. Input validation completes before any device tensor is created.
pub fn spot_second_order(
    session: &Session,
    rows: &[OptionRow],
) -> Result<SecondOrderResult, ComputeError> {
    if rows.is_empty() || rows.len() > session.max_batch_rows() {
        return Err(ComputeError::InvalidInput(format!(
            "second-order batch has {} rows; allowed range is 1..={}",
            rows.len(),
            session.max_batch_rows()
        )));
    }
    validate_job(rows, session.precision())?;
    let vector = |get: fn(&OptionRow) -> f64| session.vector(rows.iter().map(get).collect());
    let spot = Jet::variable(vector(|r| r.spot));
    let strike = Jet::constant(vector(|r| r.strike));
    let maturity = Jet::constant(vector(|r| r.maturity_years));
    let volatility = Jet::constant(vector(|r| r.volatility));
    let rate = Jet::constant(vector(|r| r.rate));
    let dividend = Jet::constant(vector(|r| r.dividend_yield));
    let sign = Jet::constant(vector(|r| match r.side {
        OptionSide::Call => 1.0,
        OptionSide::Put => -1.0,
    }));
    let mut telemetry = Telemetry::default();
    session.synchronize()?;
    telemetry.sample(session);

    let sigma_t = volatility.clone().mul(maturity.clone().sqrt());
    let drift = rate
        .clone()
        .sub(dividend.clone())
        .add(volatility.clone().mul(volatility).scale(0.5));
    let d1 = spot
        .clone()
        .div(strike.clone())
        .log()
        .add(drift.mul(maturity.clone()))
        .div(sigma_t.clone());
    let d2 = d1.clone().sub(sigma_t);
    let discounted_spot = spot.mul(dividend.scale(-1.0).mul(maturity.clone()).exp());
    let discounted_strike = strike.mul(rate.scale(-1.0).mul(maturity).exp());
    let price = sign.clone().mul(
        discounted_spot
            .mul(sign.clone().mul(d1).normal_cdf())
            .sub(discounted_strike.mul(sign.mul(d2).normal_cdf())),
    );
    session.synchronize()?;
    telemetry.sample(session);
    let prices = read_finite(session, &price.value, rows.len(), "price")?;
    let delta = read_finite(session, &price.first, rows.len(), "delta")?;
    let gamma = read_finite(session, &price.second, rows.len(), "gamma")?;
    telemetry.sample(session);
    Ok(SecondOrderResult {
        prices,
        delta,
        gamma,
        max_sampled_allocator_live_bytes: telemetry.live,
        max_sampled_allocator_reserved_bytes: telemetry.reserved,
    })
}

fn read_finite(
    session: &Session,
    tensor: &Tensor<1>,
    length: usize,
    name: &str,
) -> Result<Vec<f64>, ComputeError> {
    let values = session.read_vector(tensor)?;
    if values.len() != length || values.iter().any(|value| !value.is_finite()) {
        return Err(ComputeError::Execution(format!(
            "nonfinite or malformed second-order {name}"
        )));
    }
    Ok(values)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionTarget, Precision};

    #[test]
    fn generic_jet_chain_rules_propagate_nonzero_curvature() {
        let target = std::env::var("VOLT_TEST_DEVICE")
            .unwrap_or_else(|_| "cpu".into())
            .parse()
            .unwrap();
        let precision = if target == ExecutionTarget::Metal {
            Precision::F32
        } else {
            Precision::F64
        };
        let session = Session::new(target, precision, 3).unwrap();
        let points = vec![0.7_f64, 1.5, 3.0];
        let x = Jet::variable(session.vector(points.clone()));
        let four = Jet::constant(session.vector(vec![4.0; 3]));
        // exp(log(sqrt(x*x+4))) exercises unary and product rules with
        // nonzero first and second derivatives, even where market inputs in
        // the spot-only pricing graph would be constant.
        let smooth = x.clone().mul(x.clone()).add(four).sqrt().log().exp();
        let two = Jet::constant(session.vector(vec![2.0; 3]));
        let rational = x.clone().div(x.add(two));
        let smooth_first = session.read_vector(&smooth.first).unwrap();
        let smooth_second = session.read_vector(&smooth.second).unwrap();
        let rational_first = session.read_vector(&rational.first).unwrap();
        let rational_second = session.read_vector(&rational.second).unwrap();
        let tolerance = if precision == Precision::F32 {
            2e-6
        } else {
            1e-12
        };
        for (i, x) in points.iter().enumerate() {
            for (actual, expected) in [
                (smooth_first[i], x / (x * x + 4.0).sqrt()),
                (smooth_second[i], 4.0 / (x * x + 4.0).powf(1.5)),
                (rational_first[i], 2.0 / (x + 2.0).powi(2)),
                (rational_second[i], -4.0 / (x + 2.0).powi(3)),
            ] {
                assert!(
                    (actual - expected).abs() < tolerance,
                    "{actual} vs {expected}"
                );
            }
        }
    }
}
