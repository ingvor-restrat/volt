//! Independent analytic checks of stochastic estimates. A multi-standard-error
//! check is a regression control, not a guarantee of confidence-interval coverage.
use crate::{
    ComputeError, OptionRow, Precision,
    aad::analytic_greeks,
    chunked::CompensatedSum,
    monte_carlo::{GreekMoments, MonteCarloResult, OnlineMoments},
    second_order::analytic_gamma,
};
use serde::Serialize;

pub const VALIDATION_STANDARD_ERRORS: f64 = 6.0;

#[derive(Debug, Serialize)]
pub struct MonteCarloCheck {
    pub row_id: Option<String>,
    pub metric: &'static str,
    pub mean: f64,
    pub standard_error: f64,
    pub independent_pairs: u64,
    pub analytic_reference: f64,
    pub absolute_error: f64,
    pub numerical_allowance: f64,
    pub standard_errors_from_reference: Option<f64>,
    pub passed: bool,
}

pub fn validate_monte_carlo(
    rows: &[OptionRow],
    result: &MonteCarloResult,
    precision: Precision,
) -> Result<Vec<MonteCarloCheck>, ComputeError> {
    if result.rows.len() != rows.len() || result.pair_count < 2 {
        return Err(ComputeError::InvalidInput(
            "Monte Carlo validation needs matching rows and at least two pairs".into(),
        ));
    }
    let mut checks = Vec::with_capacity((rows.len() + 1) * 7);
    let mut portfolio: [CompensatedSum; 7] = std::array::from_fn(|_| CompensatedSum::default());
    let mut portfolio_allowance: [CompensatedSum; 7] =
        std::array::from_fn(|_| CompensatedSum::default());
    for (row, observed) in rows.iter().zip(&result.rows) {
        if row.row_id != observed.row_id {
            return Err(ComputeError::InvalidInput(
                "Monte Carlo validation identity mismatch".into(),
            ));
        }
        let analytic = analytic_greeks(row)?;
        let reference = [
            analytic.price,
            analytic.delta,
            analytic.vega,
            analytic.rho,
            analytic.theta,
            analytic.dividend_sensitivity,
            analytic_gamma(row)?,
        ];
        let weighted = row.quantity as f64 * row.multiplier as f64;
        for (i, ((name, moment), value)) in metrics(&observed.moments)
            .into_iter()
            .zip(reference)
            .enumerate()
        {
            let allowance = numerical_allowance(value, precision, i);
            checks.push(check(
                Some(row.row_id.clone()),
                name,
                moment,
                value,
                allowance,
                result.pair_count,
            )?);
            portfolio[i].add(weighted * value);
            portfolio_allowance[i].add(weighted.abs() * allowance);
        }
    }
    let totals = portfolio.map(CompensatedSum::value);
    let allowances = portfolio_allowance.map(CompensatedSum::value);
    for (i, (name, moment)) in metrics(&result.portfolio).into_iter().enumerate() {
        checks.push(check(
            None,
            name,
            moment,
            totals[i],
            allowances[i],
            result.pair_count,
        )?);
    }
    Ok(checks)
}
fn numerical_allowance(value: f64, precision: Precision, metric: usize) -> f64 {
    match precision {
        Precision::F32 if metric == 6 => 1e-6 + 2e-4 * value.abs(),
        Precision::F32 => 2e-4 + 2e-4 * value.abs(),
        Precision::F64 => 1e-9 + 1e-10 * value.abs(),
    }
}
fn check(
    row_id: Option<String>,
    metric: &'static str,
    moment: &OnlineMoments,
    reference: f64,
    allowance: f64,
    pairs: usize,
) -> Result<MonteCarloCheck, ComputeError> {
    if !moment.m2.is_finite() || moment.m2 < 0.0 || moment.count != pairs as u64 {
        return Err(ComputeError::Execution(
            "invalid pair moments supplied to Monte Carlo validation".into(),
        ));
    }
    let estimate = moment.estimate();
    let se = estimate
        .standard_error
        .ok_or_else(|| ComputeError::Execution("Monte Carlo pair SE unavailable".into()))?;
    if estimate.samples != pairs as u64
        || !estimate.mean.is_finite()
        || !se.is_finite()
        || !reference.is_finite()
        || !allowance.is_finite()
    {
        return Err(ComputeError::Execution(
            "nonfinite/count-invalid stochastic validation".into(),
        ));
    }
    let error = (estimate.mean - reference).abs();
    Ok(MonteCarloCheck {
        row_id,
        metric,
        mean: estimate.mean,
        standard_error: se,
        independent_pairs: estimate.samples,
        analytic_reference: reference,
        absolute_error: error,
        numerical_allowance: allowance,
        standard_errors_from_reference: (se > 0.0).then_some(error / se).filter(|v| v.is_finite()),
        passed: error <= VALIDATION_STANDARD_ERRORS * se + allowance,
    })
}
fn metrics(m: &GreekMoments) -> [(&'static str, &OnlineMoments); 7] {
    [
        ("price", &m.price),
        ("delta", &m.delta),
        ("vega_per_unit_volatility", &m.vega),
        ("rho_per_unit_rate", &m.rho),
        ("theta_per_year", &m.theta),
        ("dividend_per_unit_yield", &m.dividend_sensitivity),
        ("gamma_per_squared_spot_unit", &m.gamma),
    ]
}
