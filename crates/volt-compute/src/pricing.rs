use crate::{ComputeError, Session};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionSide {
    Call,
    Put,
}

impl OptionSide {
    fn sign(self) -> f64 {
        match self {
            Self::Call => 1.0,
            Self::Put => -1.0,
        }
    }
}

/// Numeric projection for a single European option mark. Monetary inputs are in
/// currency units, rates are annual continuously compounded rates, volatility is
/// an annual fraction, and maturity is in years. IDs refer to application rows,
/// allowing the same Volt contract in several scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionRow {
    pub row_id: String,
    pub contract_id: String,
    pub side: OptionSide,
    pub spot: f64,
    pub strike: f64,
    pub maturity_years: f64,
    pub volatility: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    pub quantity: i64,
    pub multiplier: u32,
}

impl OptionRow {
    pub fn validate(&self) -> Result<(), ComputeError> {
        if self.row_id.trim().is_empty() || self.contract_id.trim().is_empty() {
            return Err(ComputeError::InvalidInput(
                "row and contract identities must be nonempty".into(),
            ));
        }
        for (name, value) in [
            ("spot", self.spot),
            ("strike", self.strike),
            ("maturity_years", self.maturity_years),
            ("volatility", self.volatility),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ComputeError::InvalidInput(format!(
                    "{}: {name} must be finite and positive",
                    self.row_id
                )));
            }
        }
        if !self.rate.is_finite() || !self.dividend_yield.is_finite() || self.multiplier == 0 {
            return Err(ComputeError::InvalidInput(format!(
                "{}: rates must be finite and multiplier positive",
                self.row_id
            )));
        }
        Ok(())
    }

    pub fn validate_for_precision(&self, precision: crate::Precision) -> Result<(), ComputeError> {
        self.validate()?;
        if precision == crate::Precision::F32 {
            for value in [self.spot, self.strike, self.maturity_years, self.volatility] {
                if !(value as f32).is_finite() || value as f32 == 0.0 {
                    return Err(ComputeError::InvalidInput(format!(
                        "{}: input is outside the finite positive f32 range",
                        self.row_id
                    )));
                }
            }
            for value in [
                self.rate,
                self.dividend_yield,
                self.quantity as f64 * self.multiplier as f64,
            ] {
                if !(value as f32).is_finite() {
                    return Err(ComputeError::InvalidInput(format!(
                        "{}: input overflows f32",
                        self.row_id
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Inputs uploaded once in structure-of-arrays form, then reused on device.
/// This layer computes floating-point marks; it does not mutate Volt's ledger.
pub struct ResidentOptions {
    row_ids: Vec<String>,
    spot: Tensor<1>,
    strike: Tensor<1>,
    maturity: Tensor<1>,
    volatility: Tensor<1>,
    rate: Tensor<1>,
    dividend: Tensor<1>,
    sign: Tensor<1>,
    weights: Tensor<1>,
    spot_bounds: (f64, f64),
    precision: crate::Precision,
}

impl ResidentOptions {
    pub fn upload(session: &Session, rows: &[OptionRow]) -> Result<Self, ComputeError> {
        if rows.is_empty() || rows.len() > session.max_batch_rows() {
            return Err(ComputeError::InvalidInput(format!(
                "batch has {} rows; allowed range is 1..={}",
                rows.len(),
                session.max_batch_rows()
            )));
        }
        let mut identities = HashSet::new();
        for row in rows {
            row.validate_for_precision(session.precision())?;
            if !identities.insert(&row.row_id) {
                return Err(ComputeError::InvalidInput(format!(
                    "duplicate row identity {}",
                    row.row_id
                )));
            }
        }
        let vector = |get: fn(&OptionRow) -> f64| session.vector(rows.iter().map(get).collect());
        let spot_bounds = rows
            .iter()
            .fold((f64::INFINITY, 0.0_f64), |(low, high), row| {
                let value = if session.precision() == crate::Precision::F32 {
                    row.spot as f32 as f64
                } else {
                    row.spot
                };
                (low.min(value), high.max(value))
            });
        Ok(Self {
            row_ids: rows.iter().map(|r| r.row_id.clone()).collect(),
            spot: vector(|r| r.spot),
            strike: vector(|r| r.strike),
            maturity: vector(|r| r.maturity_years),
            volatility: vector(|r| r.volatility),
            rate: vector(|r| r.rate),
            dividend: vector(|r| r.dividend_yield),
            sign: vector(|r| r.side.sign()),
            weights: vector(|r| r.quantity as f64 * r.multiplier as f64),
            spot_bounds,
            precision: session.precision(),
        })
    }

    pub fn row_ids(&self) -> &[String] {
        &self.row_ids
    }

    pub fn prices(&self) -> Tensor<1> {
        self.prices_at_spot(self.spot.clone())
    }

    /// Apply a positive uniform spot shock on device. Contract and market input
    /// arrays remain resident and unchanged between evaluations.
    pub fn prices_with_spot_scale(&self, scale: f64) -> Result<Tensor<1>, ComputeError> {
        if !scale.is_finite()
            || scale <= 0.0
            || (self.spot.dtype() == burn::tensor::DType::F32
                && (!(scale as f32).is_finite() || scale as f32 == 0.0))
        {
            return Err(ComputeError::InvalidInput(
                "spot scale must be finite, positive, and representable at device precision".into(),
            ));
        }
        let valid_bounds = match self.precision {
            crate::Precision::F32 => {
                let low = self.spot_bounds.0 as f32;
                let high = self.spot_bounds.1 as f32;
                low.is_normal()
                    && high.is_normal()
                    && (low * scale as f32).is_normal()
                    && (high * scale as f32).is_normal()
            }
            crate::Precision::F64 => {
                self.spot_bounds.0.is_normal()
                    && self.spot_bounds.1.is_normal()
                    && (self.spot_bounds.0 * scale).is_normal()
                    && (self.spot_bounds.1 * scale).is_normal()
            }
        };
        if !valid_bounds {
            return Err(ComputeError::InvalidInput(
                "scaled spot is outside the positive normal range at device precision".into(),
            ));
        }
        let spot = if scale == 1.0 {
            self.spot.clone()
        } else {
            self.spot.clone() * scale
        };
        Ok(self.prices_at_spot(spot))
    }

    fn prices_at_spot(&self, spot: Tensor<1>) -> Tensor<1> {
        black_scholes_prices(
            spot,
            self.strike.clone(),
            self.maturity.clone(),
            self.volatility.clone(),
            self.rate.clone(),
            self.dividend.clone(),
            self.sign.clone(),
        )
    }

    /// Signed marked portfolio value, including contract multipliers.
    pub fn portfolio_value(&self, prices: Tensor<1>) -> Tensor<1> {
        (prices * self.weights.clone()).sum()
    }
}

/// Shared differentiable pricing graph for valuation and sensitivity engines.
pub(crate) fn black_scholes_prices(
    spot: Tensor<1>,
    strike: Tensor<1>,
    maturity: Tensor<1>,
    volatility: Tensor<1>,
    rate: Tensor<1>,
    dividend: Tensor<1>,
    sign: Tensor<1>,
) -> Tensor<1> {
    let sigma_t = volatility.clone() * maturity.clone().sqrt();
    let drift = rate.clone() - dividend.clone() + volatility.clone() * volatility.clone() * 0.5;
    let d1 = ((spot.clone() / strike.clone()).log() + drift * maturity.clone()) / sigma_t.clone();
    let d2 = d1.clone() - sigma_t;
    let discounted_spot = spot * (-dividend.clone() * maturity.clone()).exp();
    let discounted_strike = strike.clone() * (-rate.clone() * maturity.clone()).exp();
    sign.clone()
        * (discounted_spot * normal_cdf(sign.clone() * d1)
            - discounted_strike * normal_cdf(sign.clone() * d2))
}

fn normal_cdf(x: Tensor<1>) -> Tensor<1> {
    // Burn 0.22 scalar-division backward computes its reciprocal as f32 even
    // for f64 tensors. Multiplication keeps this constant at tensor precision.
    (x.mul_scalar(std::f64::consts::FRAC_1_SQRT_2).erf() + 1.0) * 0.5
}

/// Independent scalar double-precision Black–Scholes reference.
pub fn reference_price(row: &OptionRow) -> Result<f64, ComputeError> {
    reference_price_at_spot_scale(row, 1.0)
}

pub fn reference_price_at_spot_scale(row: &OptionRow, scale: f64) -> Result<f64, ComputeError> {
    row.validate()?;
    let spot = row.spot * scale;
    if !scale.is_finite() || scale <= 0.0 || !spot.is_finite() || spot <= 0.0 {
        return Err(ComputeError::InvalidInput("invalid scaled spot".into()));
    }
    let sigma_t = row.volatility * row.maturity_years.sqrt();
    let d1 = ((spot / row.strike).ln()
        + (row.rate - row.dividend_yield + 0.5 * row.volatility.powi(2)) * row.maturity_years)
        / sigma_t;
    let d2 = d1 - sigma_t;
    let cdf = |x: f64| 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
    let sign = row.side.sign();
    let price = sign
        * (spot * (-row.dividend_yield * row.maturity_years).exp() * cdf(sign * d1)
            - row.strike * (-row.rate * row.maturity_years).exp() * cdf(sign * d2));
    if !price.is_finite() {
        return Err(ComputeError::InvalidInput(format!(
            "{}: price overflow",
            row.row_id
        )));
    }
    Ok(price)
}
