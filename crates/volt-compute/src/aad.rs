//! First-order reverse differentiation of the shared Black–Scholes graph.
//!
//! Every call creates fresh leaves and a fresh tape, then performs one backward
//! pass. The sum of independent row prices yields each row's own sensitivities;
//! quantities and multipliers are deliberately applied by the portfolio layer.

use crate::{
    ComputeError, OptionRow, OptionSide, Precision, Session, chunked::validate_job,
    pricing::black_scholes_prices,
};
use burn::tensor::{DType, Gradients, Tensor, TensorCreationOptions, TensorData};
use serde::{Deserialize, Serialize};

/// Prices and first derivatives, in input-row order and per underlying unit.
/// Vega, rho, and dividend sensitivity are per 1.0 change in the corresponding
/// annual fraction (multiply by 0.01 for one percentage point). Theta is calendar
/// decay in currency units per year, -dP/dT, with all other inputs held fixed.
/// No higher-order derivative is claimed or returned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AadResult {
    pub prices: Vec<f64>,
    pub delta: Vec<f64>,
    pub vega: Vec<f64>,
    pub rho: Vec<f64>,
    pub theta: Vec<f64>,
    pub dividend_sensitivity: Vec<f64>,
    /// Samples include the retained forward tape and completed backward graph.
    /// They are not guarantees of the instantaneous peak allocation.
    pub max_sampled_allocator_live_bytes: Option<u64>,
    pub max_sampled_allocator_reserved_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AnalyticGreeks {
    pub price: f64,
    pub delta: f64,
    pub vega: f64,
    pub rho: f64,
    pub theta: f64,
    pub dividend_sensitivity: f64,
}

/// Low-level single resident batch. Admission, serialization of concurrent jobs,
/// and chunking belong to the caller. This function validates all inputs before
/// creating any tensors and refuses a batch above the session's row cap.
pub fn differentiate_rows(
    session: &Session,
    rows: &[OptionRow],
) -> Result<AadResult, ComputeError> {
    if rows.is_empty() || rows.len() > session.max_batch_rows() {
        return Err(ComputeError::InvalidInput(format!(
            "AAD batch has {} rows; allowed range is 1..={}",
            rows.len(),
            session.max_batch_rows()
        )));
    }
    validate_job(rows, session.precision())?;
    let device = session.device().clone().autodiff();
    let dtype = match session.precision() {
        Precision::F32 => DType::F32,
        Precision::F64 => DType::F64,
    };
    let vector = |get: fn(&OptionRow) -> f64| {
        Tensor::<1>::from_data(
            TensorData::new(rows.iter().map(get).collect::<Vec<_>>(), [rows.len()]),
            TensorCreationOptions::new(device.clone()).with_dtype(dtype),
        )
    };
    let spot = vector(|row| row.spot).require_grad();
    let volatility = vector(|row| row.volatility).require_grad();
    let rate = vector(|row| row.rate).require_grad();
    let maturity = vector(|row| row.maturity_years).require_grad();
    let dividend = vector(|row| row.dividend_yield).require_grad();
    let strike = vector(|row| row.strike);
    let sign = vector(|row| match row.side {
        OptionSide::Call => 1.0,
        OptionSide::Put => -1.0,
    });
    let price_tensor = black_scholes_prices(
        spot.clone(),
        strike,
        maturity.clone(),
        volatility.clone(),
        rate.clone(),
        dividend.clone(),
        sign,
    );
    session.synchronize()?;
    let mut telemetry = AllocatorTelemetry::default();
    telemetry.sample(session);
    // Materialize values before backward consumes the graph's recorded steps.
    let prices = read_finite(session, &price_tensor, rows.len(), "price")?;
    let objective = price_tensor.sum();
    let mut gradients = objective.backward();
    session.synchronize()?;
    telemetry.sample(session);
    let delta = read_gradient(session, &spot, &mut gradients, rows.len(), "delta")?;
    let vega = read_gradient(session, &volatility, &mut gradients, rows.len(), "vega")?;
    let rho = read_gradient(session, &rate, &mut gradients, rows.len(), "rho")?;
    let mut theta = read_gradient(session, &maturity, &mut gradients, rows.len(), "dP/dT")?;
    for value in &mut theta {
        *value = -*value;
    }
    let dividend_sensitivity = read_gradient(
        session,
        &dividend,
        &mut gradients,
        rows.len(),
        "dividend sensitivity",
    )?;
    telemetry.sample(session);
    Ok(AadResult {
        prices,
        delta,
        vega,
        rho,
        theta,
        dividend_sensitivity,
        max_sampled_allocator_live_bytes: telemetry.live,
        max_sampled_allocator_reserved_bytes: telemetry.reserved,
    })
}

/// Independent closed-form scalar first derivatives. This code does not execute
/// the tensor graph or use its gradients, and uses libm's double-precision CDF.
pub fn analytic_greeks(row: &OptionRow) -> Result<AnalyticGreeks, ComputeError> {
    row.validate()?;
    let time = row.maturity_years;
    let root_time = time.sqrt();
    let sigma_time = row.volatility * root_time;
    let d1 = ((row.spot / row.strike).ln()
        + (row.rate - row.dividend_yield + 0.5 * row.volatility * row.volatility) * time)
        / sigma_time;
    let d2 = d1 - sigma_time;
    let sign = match row.side {
        OptionSide::Call => 1.0,
        OptionSide::Put => -1.0,
    };
    let cdf = |x: f64| 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
    let density = (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let dividend_discount = (-row.dividend_yield * time).exp();
    let discounted_spot = row.spot * dividend_discount;
    let discounted_strike = row.strike * (-row.rate * time).exp();
    let n1 = cdf(sign * d1);
    let n2 = cdf(sign * d2);
    let result = AnalyticGreeks {
        price: sign * (discounted_spot * n1 - discounted_strike * n2),
        delta: sign * dividend_discount * n1,
        vega: discounted_spot * density * root_time,
        rho: sign * time * discounted_strike * n2,
        theta: -discounted_spot * density * row.volatility / (2.0 * root_time)
            - sign * row.rate * discounted_strike * n2
            + sign * row.dividend_yield * discounted_spot * n1,
        dividend_sensitivity: -sign * time * discounted_spot * n1,
    };
    if [
        result.price,
        result.delta,
        result.vega,
        result.rho,
        result.theta,
        result.dividend_sensitivity,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(ComputeError::InvalidInput(format!(
            "{}: analytic Greek overflow",
            row.row_id
        )));
    }
    Ok(result)
}

fn read_gradient(
    session: &Session,
    leaf: &Tensor<1>,
    gradients: &mut Gradients,
    length: usize,
    name: &str,
) -> Result<Vec<f64>, ComputeError> {
    let gradient = leaf.grad_remove(gradients).ok_or_else(|| {
        ComputeError::Execution(format!("AAD graph did not retain {name} gradient"))
    })?;
    read_finite(session, &gradient, length, name)
}

fn read_finite(
    session: &Session,
    tensor: &Tensor<1>,
    length: usize,
    name: &str,
) -> Result<Vec<f64>, ComputeError> {
    // Reading a detached clone leaves the original tracked handle and its tape
    // intact, while making the session's physical-device check unambiguous.
    let readable = tensor.clone().without_autodiff();
    let values = session.read_vector(&readable)?;
    if values.len() != length || values.iter().any(|value| !value.is_finite()) {
        return Err(ComputeError::Execution(format!(
            "nonfinite or malformed AAD {name}"
        )));
    }
    Ok(values)
}

#[derive(Default)]
struct AllocatorTelemetry {
    live: Option<u64>,
    reserved: Option<u64>,
}
impl AllocatorTelemetry {
    fn sample(&mut self, session: &Session) {
        if let Some(usage) = session.device().memory_pool_usage() {
            self.live = Some(self.live.unwrap_or(0).max(usage.bytes_in_use));
            self.reserved = Some(self.reserved.unwrap_or(0).max(usage.bytes_reserved));
        }
    }
}
