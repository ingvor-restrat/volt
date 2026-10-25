//! Bounded CPU-f64 European quote inversion and a transparent volatility surface.
//!
//! Coordinates are log-forward moneyness and total variance. Interpolation is
//! linear in each expiry slice and then linear in maturity, with no extrapolation.
//! This is not SVI and does not guarantee absence of static arbitrage. Diagnostics
//! accompany every fit. Rates/yields are continuously compounded deterministic
//! zero rates at each expiry; integrated carry is interpolated between expiries.
//! Reference for coordinates/arbitrage distinctions: Gatheral & Jacquier,
//! <https://arxiv.org/abs/1204.0646>. No market feed or UI is supplied here.
use crate::{
    ComputeError, OptionRow, OptionSide,
    aad::{AnalyticGreeks, analytic_greeks},
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

fn invalid(message: &str) -> ComputeError {
    ComputeError::InvalidInput(message.into())
}
fn finite(x: f64) -> Result<f64, ComputeError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(invalid("European calculation overflow"))
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EuropeanInputs {
    pub spot: f64,
    pub strike: f64,
    pub maturity_years: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    pub side: OptionSide,
}
impl EuropeanInputs {
    pub fn validate(&self) -> Result<(), ComputeError> {
        if !self.spot.is_finite()
            || self.spot <= 0.0
            || !self.strike.is_finite()
            || self.strike <= 0.0
            || !self.maturity_years.is_finite()
            || self.maturity_years < 0.0
            || !self.rate.is_finite()
            || !self.dividend_yield.is_finite()
        {
            return Err(invalid("invalid European spot/strike/maturity/carry"));
        }
        let (a, b) = self.discounted()?;
        if a <= 0.0 || b <= 0.0 {
            return Err(invalid("discounted notional underflow"));
        }
        Ok(())
    }
    fn discounted(&self) -> Result<(f64, f64), ComputeError> {
        Ok((
            finite(self.spot * (-self.dividend_yield * self.maturity_years).exp())?,
            finite(self.strike * (-self.rate * self.maturity_years).exp())?,
        ))
    }
    pub fn bounds(&self) -> Result<(f64, f64), ComputeError> {
        self.validate()?;
        let (a, b) = self.discounted()?;
        Ok(match self.side {
            OptionSide::Call => ((a - b).max(0.0), a),
            OptionSide::Put => ((b - a).max(0.0), b),
        })
    }
    fn row(&self, volatility: f64) -> OptionRow {
        OptionRow {
            row_id: "surface-query".into(),
            contract_id: "european".into(),
            side: self.side,
            spot: self.spot,
            strike: self.strike,
            maturity_years: self.maturity_years,
            volatility,
            rate: self.rate,
            dividend_yield: self.dividend_yield,
            quantity: 1,
            multiplier: 1,
        }
    }
}
/// Stable OTM evaluation followed by intrinsic value for the ITM side. Inverting
/// a tiny time value still inherits floating-point/quote conditioning limits.
pub fn european_price_and_vega(
    input: &EuropeanInputs,
    volatility: f64,
) -> Result<(f64, f64), ComputeError> {
    input.validate()?;
    if !volatility.is_finite() || volatility < 0.0 {
        return Err(invalid("invalid volatility"));
    }
    let (a, b) = input.discounted()?;
    let signed = match input.side {
        OptionSide::Call => a - b,
        OptionSide::Put => b - a,
    };
    if input.maturity_years == 0.0 || volatility == 0.0 {
        return Ok((signed.max(0.0), 0.0));
    }
    let root = input.maturity_years.sqrt();
    let z = finite(volatility * root)?;
    if z == 0.0 {
        return Ok((signed.max(0.0), 0.0));
    }
    let log_forward = (input.spot.ln() - input.strike.ln())
        + (input.rate - input.dividend_yield) * input.maturity_years;
    let d1 = finite(log_forward / z + 0.5 * z)?;
    let d2 = finite(d1 - z)?;
    let n = |x: f64| 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
    let otm = if a <= b {
        a * n(d1) - b * n(d2)
    } else {
        b * n(-d2) - a * n(-d1)
    };
    // Roundoff can give a negative OTM value of order machine epsilon only.
    let allowance = 32.0 * f64::EPSILON * a.min(b);
    if otm < -allowance {
        return Err(invalid("negative European time value beyond roundoff"));
    }
    let value = finite(otm.max(0.0) + signed.max(0.0))?;
    let vega = finite(a * (-0.5 * d1 * d1).exp() / std::f64::consts::TAU.sqrt() * root)?;
    Ok((value, vega))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvConfig {
    pub price_absolute_tolerance: f64,
    pub price_relative_tolerance: f64,
    pub volatility_tolerance: f64,
    pub minimum_vega: f64,
    pub initial_upper_volatility: f64,
    pub maximum_volatility: f64,
    pub max_iterations: usize,
}
impl Default for IvConfig {
    fn default() -> Self {
        Self {
            price_absolute_tolerance: 1e-10,
            price_relative_tolerance: 1e-12,
            volatility_tolerance: 1e-10,
            minimum_vega: 1e-8,
            initial_upper_volatility: 0.5,
            maximum_volatility: 16.0,
            max_iterations: 128,
        }
    }
}
impl IvConfig {
    pub fn validate(&self) -> Result<(), ComputeError> {
        if [
            self.price_absolute_tolerance,
            self.price_relative_tolerance,
            self.volatility_tolerance,
            self.minimum_vega,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
            || self.volatility_tolerance == 0.0
            || (self.price_absolute_tolerance == 0.0 && self.price_relative_tolerance == 0.0)
            || !self.initial_upper_volatility.is_finite()
            || self.initial_upper_volatility <= 0.0
            || !self.maximum_volatility.is_finite()
            || self.maximum_volatility < self.initial_upper_volatility
            || !(1..=4096).contains(&self.max_iterations)
        {
            return Err(invalid("invalid IV tolerances, bracket or iteration cap"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IvStatus {
    Converged,
    LowVega,
    LowerBoundary,
    UpperBoundary,
    Expired,
    OutsideBounds,
    BracketLimit,
    MaxIterations,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvResult {
    pub status: IvStatus,
    pub volatility: Option<f64>,
    pub price_residual: Option<f64>,
    pub vega: Option<f64>,
    pub bracket: [f64; 2],
    pub iterations: usize,
}
/// Expected financial boundary/nonconvergence states are returned as statuses;
/// malformed inputs/configuration are errors. Boundary IVs are not fitted.
pub fn implied_volatility(
    input: &EuropeanInputs,
    price: f64,
    config: &IvConfig,
) -> Result<IvResult, ComputeError> {
    config.validate()?;
    let (lower, upper) = input.bounds()?;
    if !price.is_finite() {
        return Err(invalid("nonfinite option quote"));
    }
    let status = if price < lower || price > upper {
        Some(IvStatus::OutsideBounds)
    } else if input.maturity_years == 0.0 {
        Some(if price == lower {
            IvStatus::Expired
        } else {
            IvStatus::OutsideBounds
        })
    } else if price == lower {
        Some(IvStatus::LowerBoundary)
    } else if price == upper {
        Some(IvStatus::UpperBoundary)
    } else {
        None
    };
    if let Some(status) = status {
        return Ok(IvResult {
            status,
            volatility: if status == IvStatus::LowerBoundary {
                Some(0.0)
            } else {
                None
            },
            price_residual: if status == IvStatus::Expired {
                Some(lower - price)
            } else {
                None
            },
            vega: None,
            bracket: [0.0, 0.0],
            iterations: 0,
        });
    }
    let tolerance =
        finite(config.price_absolute_tolerance + config.price_relative_tolerance * price.abs())?;
    let mut lo = 0.0;
    let mut hi = config.initial_upper_volatility;
    while european_price_and_vega(input, hi)?.0 < price && hi < config.maximum_volatility {
        hi = (2.0 * hi).min(config.maximum_volatility);
    }
    if european_price_and_vega(input, hi)?.0 < price {
        return Ok(IvResult {
            status: IvStatus::BracketLimit,
            volatility: None,
            price_residual: Some(european_price_and_vega(input, hi)?.0 - price),
            vega: None,
            bracket: [lo, hi],
            iterations: 0,
        });
    }
    let mut x = 0.5 * (lo + hi);
    for iteration in 1..=config.max_iterations {
        let (value, vega) = european_price_and_vega(input, x)?;
        let residual = value - price;
        // Both residual and volatility localization are required. A small vega
        // can make a tiny repricing residual compatible with a very wide root.
        let local_error = if vega > 0.0 {
            residual.abs() / vega
        } else {
            f64::INFINITY
        };
        if residual.abs() <= tolerance
            && (hi - lo <= config.volatility_tolerance
                || local_error <= config.volatility_tolerance)
        {
            return Ok(IvResult {
                status: if vega <= config.minimum_vega {
                    IvStatus::LowVega
                } else {
                    IvStatus::Converged
                },
                volatility: Some(x),
                price_residual: Some(residual),
                vega: Some(vega),
                bracket: [lo, hi],
                iterations: iteration,
            });
        }
        if residual < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let newton = x - residual / vega;
        x = if vega > config.minimum_vega && newton.is_finite() && newton > lo && newton < hi {
            newton
        } else {
            lo + 0.5 * (hi - lo)
        };
    }
    let (value, vega) = european_price_and_vega(input, x)?;
    Ok(IvResult {
        status: IvStatus::MaxIterations,
        volatility: Some(x),
        price_residual: Some(value - price),
        vega: Some(vega),
        bracket: [lo, hi],
        iterations: config.max_iterations,
    })
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteRow {
    pub quote_id: String,
    pub contract_id: String,
    pub source_id: String,
    pub underlying_id: String,
    pub timestamp_ns: i64,
    pub input: EuropeanInputs,
    pub bid: f64,
    pub ask: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainConfig {
    pub cutoff_ns: i64,
    pub max_age_ns: u64,
    pub max_quotes: usize,
    pub max_input_bytes: usize,
    pub iv: IvConfig,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteIssue {
    InvalidInput,
    CrossedOrNegative,
    Future,
    Stale,
    IneligibleBid,
    IneligibleMid,
    IneligibleAsk,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedQuote {
    pub quote: QuoteRow,
    pub bid_iv: Option<IvResult>,
    pub mid_iv: Option<IvResult>,
    pub ask_iv: Option<IvResult>,
    pub eligible: bool,
    pub issues: Vec<QuoteIssue>,
}
/// Source identities are never discarded. A quote is eligible only if all three
/// prices identify finite, adequately conditioned interior roots.
pub fn process_chain(
    rows: &[QuoteRow],
    config: &ChainConfig,
) -> Result<Vec<ProcessedQuote>, ComputeError> {
    config.iv.validate()?;
    if config.max_quotes == 0
        || config.max_quotes > 1_000_000
        || rows.is_empty()
        || rows.len() > config.max_quotes
    {
        return Err(invalid("quote count exceeds declared cap"));
    }
    let bytes = rows
        .iter()
        .try_fold(0usize, |n, q| {
            n.checked_add(std::mem::size_of::<QuoteRow>())
                .and_then(|n| n.checked_add(q.quote_id.len()))
                .and_then(|n| n.checked_add(q.contract_id.len()))
                .and_then(|n| n.checked_add(q.source_id.len()))
                .and_then(|n| n.checked_add(q.underlying_id.len()))
        })
        .ok_or_else(|| invalid("quote input byte overflow"))?;
    if bytes > config.max_input_bytes {
        return Err(invalid("quote identity/input byte cap exceeded"));
    }
    let mut ids = HashSet::new();
    for q in rows {
        if [&q.quote_id, &q.contract_id, &q.source_id, &q.underlying_id]
            .iter()
            .any(|s| s.trim().is_empty())
            || !ids.insert(&q.quote_id)
        {
            return Err(invalid("empty or duplicate quote identity"));
        }
    }
    rows.iter()
        .map(|q| {
            let mut issues = Vec::new();
            if q.input.validate().is_err() || !q.bid.is_finite() || !q.ask.is_finite() {
                issues.push(QuoteIssue::InvalidInput);
            }
            if q.bid < 0.0 || q.ask < q.bid {
                issues.push(QuoteIssue::CrossedOrNegative);
            }
            if q.timestamp_ns > config.cutoff_ns {
                issues.push(QuoteIssue::Future);
            } else if (i128::from(config.cutoff_ns) - i128::from(q.timestamp_ns)) as u128
                > u128::from(config.max_age_ns)
            {
                issues.push(QuoteIssue::Stale);
            }
            let mut results = [None, None, None];
            if !issues.contains(&QuoteIssue::InvalidInput)
                && !issues.contains(&QuoteIssue::CrossedOrNegative)
            {
                for (i, p) in [q.bid, q.bid + 0.5 * (q.ask - q.bid), q.ask]
                    .into_iter()
                    .enumerate()
                {
                    let iv = implied_volatility(&q.input, p, &config.iv)?;
                    if iv.status != IvStatus::Converged {
                        issues.push(
                            [
                                QuoteIssue::IneligibleBid,
                                QuoteIssue::IneligibleMid,
                                QuoteIssue::IneligibleAsk,
                            ][i]
                                .clone(),
                        );
                    }
                    results[i] = Some(iv);
                }
            }
            let [bid_iv, mid_iv, ask_iv] = results;
            Ok(ProcessedQuote {
                quote: q.clone(),
                bid_iv,
                mid_iv,
                ask_iv,
                eligible: issues.is_empty(),
                issues,
            })
        })
        .collect()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceConfig {
    pub max_quotes: usize,
    /// Input rows and variable-length identities, checked before fit copies.
    pub max_input_bytes: usize,
    pub max_knots: usize,
    pub max_diagnostic_checks: usize,
    /// Absolute tolerance for normalized price slopes/convexity/calendar w and
    /// unnormalized put-call price residuals (recorded separately in issues).
    pub diagnostic_tolerance: f64,
}
impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            max_quotes: 100_000,
            max_input_bytes: 64 * 1024 * 1024,
            max_knots: 100_000,
            max_diagnostic_checks: 1_000_000,
            diagnostic_tolerance: 1e-9,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct FitResidual {
    pub quote_id: String,
    pub price_residual: f64,
    pub volatility_residual: f64,
}
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticIssue {
    pub kind: String,
    pub maturity_years: f64,
    pub coordinate: f64,
    pub residual: f64,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct SurfaceDiagnostics {
    pub observations: usize,
    pub knots: usize,
    pub checks: usize,
    pub issues: Vec<DiagnosticIssue>,
    pub residuals: Vec<FitResidual>,
    /// Always false: finite-grid checks cannot prove global no-arbitrage.
    pub arbitrage_free_guaranteed: bool,
}
#[derive(Debug, Clone, Serialize)]
struct Knot {
    k: f64,
    w: f64,
}
#[derive(Debug, Clone, Serialize)]
struct Slice {
    time: f64,
    rate: f64,
    dividend_yield: f64,
    knots: Vec<Knot>,
}
#[derive(Debug, Clone, Serialize)]
pub struct SourceObservation {
    pub quote_id: String,
    pub contract_id: String,
    pub source_id: String,
    pub timestamp_ns: i64,
}
#[derive(Debug, Clone, Serialize)]
pub struct VolatilitySurface {
    pub underlying_id: String,
    pub source_quote_ids: Vec<String>,
    pub source_observations: Vec<SourceObservation>,
    pub interpolation_method: String,
    pub carry_convention: String,
    pub extrapolation_policy: String,
    pub diagnostics: SurfaceDiagnostics,
    spot: f64,
    slices: Vec<Slice>,
}
#[derive(Debug, Clone, Serialize)]
pub struct SurfacePoint {
    pub spot: f64,
    pub strike: f64,
    pub maturity_years: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    pub log_forward_moneyness: f64,
    pub total_variance: f64,
    pub volatility: f64,
    pub at_interpolation_knot: bool,
}
fn interpolate(knots: &[Knot], k: f64) -> Result<(f64, bool), ComputeError> {
    let first = knots
        .first()
        .ok_or_else(|| invalid("empty surface slice"))?;
    let last = knots.last().unwrap();
    // Exact knot queries use the supplied data; all extrapolation is refused.
    if k < first.k || k > last.k {
        return Err(invalid("surface moneyness extrapolation refused"));
    }
    let i = knots.partition_point(|p| p.k < k);
    if i < knots.len() && knots[i].k == k {
        return Ok((knots[i].w, true));
    }
    let a = &knots[i - 1];
    let b = &knots[i];
    Ok((a.w + (b.w - a.w) * ((k - a.k) / (b.k - a.k)), false))
}
impl VolatilitySurface {
    pub fn spot(&self) -> f64 {
        self.spot
    }
    pub fn expiry_count(&self) -> usize {
        self.slices.len()
    }
    fn bracket(&self, t: f64) -> Result<(usize, usize, f64), ComputeError> {
        if !t.is_finite()
            || t <= 0.0
            || t < self.slices[0].time
            || t > self.slices.last().unwrap().time
        {
            return Err(invalid("surface maturity extrapolation refused"));
        }
        let i = self.slices.partition_point(|s| s.time < t);
        if self.slices[i].time == t {
            return Ok((i, i, 0.0));
        }
        let a = i - 1;
        Ok((
            a,
            i,
            (t - self.slices[a].time) / (self.slices[i].time - self.slices[a].time),
        ))
    }
    fn carry(&self, t: f64) -> Result<(f64, f64), ComputeError> {
        let (a, b, x) = self.bracket(t)?;
        let f = &self.slices[a];
        let g = &self.slices[b];
        Ok((
            (f.rate * f.time + x * (g.rate * g.time - f.rate * f.time)) / t,
            (f.dividend_yield * f.time
                + x * (g.dividend_yield * g.time - f.dividend_yield * f.time))
                / t,
        ))
    }
    pub fn query(&self, strike: f64, maturity_years: f64) -> Result<SurfacePoint, ComputeError> {
        self.query_at_spot(self.spot, strike, maturity_years)
    }
    /// Changes spot/forward while keeping the original fitted (k,w) surface
    /// frozen. No quotes are recalibrated by this operation.
    pub fn query_at_spot(
        &self,
        spot: f64,
        strike: f64,
        t: f64,
    ) -> Result<SurfacePoint, ComputeError> {
        if !spot.is_finite() || spot <= 0.0 || !strike.is_finite() || strike <= 0.0 {
            return Err(invalid("invalid surface spot/strike"));
        }
        let (a, b, x) = self.bracket(t)?;
        let (rate, dividend_yield) = self.carry(t)?;
        let k = finite(strike.ln() - spot.ln() - (rate - dividend_yield) * t)?;
        let (w0, k0) = interpolate(&self.slices[a].knots, k)?;
        let (w1, k1) = interpolate(&self.slices[b].knots, k)?;
        let w = finite(w0 + x * (w1 - w0))?;
        if w <= 0.0 {
            return Err(invalid("nonpositive surface variance"));
        }
        Ok(SurfacePoint {
            spot,
            strike,
            maturity_years: t,
            rate,
            dividend_yield,
            log_forward_moneyness: k,
            total_variance: w,
            volatility: finite((w / t).sqrt())?,
            at_interpolation_knot: k0 || k1,
        })
    }
    pub fn greeks(
        &self,
        strike: f64,
        t: f64,
        side: OptionSide,
        convention: GreekConvention,
    ) -> Result<SurfaceGreeks, ComputeError> {
        let point = self.query(strike, t)?;
        let input = EuropeanInputs {
            spot: self.spot,
            strike,
            maturity_years: t,
            rate: point.rate,
            dividend_yield: point.dividend_yield,
            side,
        };
        let partial = analytic_greeks(&input.row(point.volatility))?;
        let gamma = finite(partial.vega / self.spot / self.spot / point.volatility / t)?;
        let (spot_delta, spot_gamma) = match convention {
            GreekConvention::FixedVolatility => (partial.delta, gamma),
            GreekConvention::StickyLogMoneynessFiniteBump { relative_spot_bump } => {
                if !relative_spot_bump.is_finite() || !(1e-7..=0.1).contains(&relative_spot_bump) {
                    return Err(invalid("relative Greek bump must be1e-7..=0.1"));
                }
                let h = self.spot * relative_spot_bump;
                let price = |s: f64| -> Result<f64, ComputeError> {
                    let p = self.query_at_spot(s, strike, t)?;
                    european_price_and_vega(&EuropeanInputs { spot: s, ..input }, p.volatility)
                        .map(|p| p.0)
                };
                let plus = price(self.spot + h)?;
                let minus = price(self.spot - h)?;
                let center = european_price_and_vega(&input, point.volatility)?.0;
                (
                    finite((plus - minus) / (2.0 * h))?,
                    finite(((plus - center) + (minus - center)) / h / h)?,
                )
            }
        };
        Ok(SurfaceGreeks {
            point,
            convention,
            fixed_volatility_partials: partial,
            fixed_volatility_gamma: gamma,
            spot_delta,
            spot_gamma,
        })
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GreekConvention {
    FixedVolatility,
    StickyLogMoneynessFiniteBump { relative_spot_bump: f64 },
}
#[derive(Debug, Clone, Serialize)]
pub struct SurfaceGreeks {
    pub point: SurfacePoint,
    pub convention: GreekConvention,
    /// Theta/rho/vega in this field remain fixed-volatility partials; only the
    /// separately labelled spot_delta/spot_gamma follow the requested convention.
    pub fixed_volatility_partials: AnalyticGreeks,
    pub fixed_volatility_gamma: f64,
    pub spot_delta: f64,
    pub spot_gamma: f64,
}
/// Average total variance at duplicate (expiry,strike) knots (e.g. call/put),
/// interpolate, and expose fit residuals plus finite-grid arbitrage diagnostics.
/// Input order does not choose which side of a duplicate survives.
pub fn fit_surface(
    rows: &[ProcessedQuote],
    config: &SurfaceConfig,
) -> Result<VolatilitySurface, ComputeError> {
    if config.max_quotes == 0
        || config.max_quotes > 1_000_000
        || config.max_input_bytes == 0
        || config.max_input_bytes > 1024 * 1024 * 1024
        || config.max_knots == 0
        || config.max_knots > 1_000_000
        || config.max_diagnostic_checks == 0
        || config.max_diagnostic_checks > 10_000_000
        || rows.len() > config.max_quotes
        || !config.diagnostic_tolerance.is_finite()
        || config.diagnostic_tolerance < 0.0
    {
        return Err(invalid("invalid surface bounds/tolerance or quote count"));
    }
    let input_bytes = rows
        .iter()
        .try_fold(0usize, |n, p| {
            n.checked_add(std::mem::size_of::<ProcessedQuote>())
                .and_then(|n| n.checked_add(p.quote.quote_id.len()))
                .and_then(|n| n.checked_add(p.quote.contract_id.len()))
                .and_then(|n| n.checked_add(p.quote.source_id.len()))
                .and_then(|n| n.checked_add(p.quote.underlying_id.len()))
                .and_then(|n| {
                    n.checked_add(
                        p.issues
                            .len()
                            .checked_mul(std::mem::size_of::<QuoteIssue>())?,
                    )
                })
        })
        .ok_or_else(|| invalid("surface input byte overflow"))?;
    if input_bytes > config.max_input_bytes {
        return Err(invalid("surface input byte cap exceeded"));
    }
    let first = rows.first().ok_or_else(|| invalid("empty quote chain"))?;
    let spot = first.quote.input.spot;
    let underlying = &first.quote.underlying_id;
    let mut observations = Vec::new();
    let mut ids = HashSet::new();
    for row in rows {
        if row.quote.underlying_id != *underlying
            || row.quote.input.spot.to_bits() != spot.to_bits()
            || !ids.insert(&row.quote.quote_id)
        {
            return Err(invalid(
                "surface requires one underlying, common spot and unique quote IDs",
            ));
        }
        if row.eligible {
            if !row.issues.is_empty()
                || !row.quote.bid.is_finite()
                || !row.quote.ask.is_finite()
                || row.quote.bid < 0.0
                || row.quote.ask < row.quote.bid
                || [
                    &row.quote.quote_id,
                    &row.quote.contract_id,
                    &row.quote.source_id,
                    &row.quote.underlying_id,
                ]
                .iter()
                .any(|s| s.trim().is_empty())
            {
                return Err(invalid("malformed eligible quote"));
            }
            row.quote.input.validate()?;
            let iv = row
                .mid_iv
                .as_ref()
                .ok_or_else(|| invalid("eligible quote missing midpoint IV"))?;
            if iv.status != IvStatus::Converged {
                return Err(invalid("eligible quote has unusable IV status"));
            }
            let v = iv
                .volatility
                .ok_or_else(|| invalid("eligible quote missing volatility"))?;
            if !v.is_finite() || v <= 0.0 || row.quote.input.maturity_years <= 0.0 {
                return Err(invalid("invalid eligible IV"));
            }
            let q = &row.quote;
            let k = finite(
                q.input.strike.ln()
                    - spot.ln()
                    - (q.input.rate - q.input.dividend_yield) * q.input.maturity_years,
            )?;
            let w = finite(v * v * q.input.maturity_years)?;
            observations.push((row, k, w));
        }
    }
    if observations.is_empty() {
        return Err(invalid("no eligible interior IV observations"));
    }
    observations.sort_by(|a, b| {
        a.0.quote
            .input
            .maturity_years
            .total_cmp(&b.0.quote.input.maturity_years)
            .then(a.1.total_cmp(&b.1))
            .then(a.0.quote.quote_id.cmp(&b.0.quote.quote_id))
    });
    let mut slices: Vec<Slice> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut knots = 0usize;
    for (row, k, w) in &observations {
        let q = &row.quote.input;
        if slices.last().is_none_or(|s| s.time != q.maturity_years) {
            slices.push(Slice {
                time: q.maturity_years,
                rate: q.rate,
                dividend_yield: q.dividend_yield,
                knots: Vec::new(),
            });
            counts.clear();
        }
        let slice = slices.last_mut().unwrap();
        if slice.rate != q.rate || slice.dividend_yield != q.dividend_yield {
            return Err(invalid(
                "same-expiry quotes disagree on deterministic carry",
            ));
        }
        if slice.knots.last().is_some_and(|p| p.k == *k) {
            let n = counts.last_mut().unwrap();
            *n += 1;
            let p = slice.knots.last_mut().unwrap();
            p.w += (*w - p.w) / (*n as f64);
        } else {
            knots += 1;
            if knots > config.max_knots {
                return Err(invalid("surface knot cap exceeded"));
            }
            slice.knots.push(Knot { k: *k, w: *w });
            counts.push(1);
        }
    }
    // Linear-size strike/calendar diagnostics plus actual same-coordinate
    // call/put cross comparisons. Refuse duplicate-heavy quadratic work before
    // allocating issue/residual output; unrelated contracts are never paired.
    let mut upper = observations
        .len()
        .checked_mul(9)
        .ok_or_else(|| invalid("surface diagnostic shape overflow"))?;
    let mut start = 0;
    while start < observations.len() {
        let mut end = start + 1;
        while end < observations.len()
            && observations[end].0.quote.input.maturity_years
                == observations[start].0.quote.input.maturity_years
            && observations[end].1 == observations[start].1
        {
            end += 1;
        }
        let n = end - start;
        upper = upper
            .checked_add(
                n.checked_mul(n - 1)
                    .ok_or_else(|| invalid("parity comparison overflow"))?
                    / 2,
            )
            .ok_or_else(|| invalid("surface diagnostic shape overflow"))?;
        start = end;
    }
    if upper > config.max_diagnostic_checks {
        return Err(invalid("surface diagnostic work exceeds cap"));
    }
    let mut surface = VolatilitySurface {
        underlying_id: underlying.clone(),
        source_quote_ids: observations
            .iter()
            .map(|p| p.0.quote.quote_id.clone())
            .collect(),
        source_observations: observations
            .iter()
            .map(|p| SourceObservation {
                quote_id: p.0.quote.quote_id.clone(),
                contract_id: p.0.quote.contract_id.clone(),
                source_id: p.0.quote.source_id.clone(),
                timestamp_ns: p.0.quote.timestamp_ns,
            })
            .collect(),
        interpolation_method: "piecewise_linear_total_variance_log_forward_moneyness_then_maturity"
            .into(),
        carry_convention:
            "piecewise_linear_integrated_continuously_compounded_zero_rate_and_dividend_yield"
                .into(),
        extrapolation_policy: "refuse".into(),
        diagnostics: SurfaceDiagnostics {
            observations: observations.len(),
            knots,
            ..Default::default()
        },
        spot,
        slices,
    };
    diagnose(&mut surface, &observations, config)?;
    Ok(surface)
}
fn diagnose(
    surface: &mut VolatilitySurface,
    observations: &[(&ProcessedQuote, f64, f64)],
    config: &SurfaceConfig,
) -> Result<(), ComputeError> {
    let tol = config.diagnostic_tolerance;
    for (row, _, _) in observations {
        let q = &row.quote;
        let point = surface.query(q.input.strike, q.input.maturity_years)?;
        let value = european_price_and_vega(&q.input, point.volatility)?.0;
        surface.diagnostics.residuals.push(FitResidual {
            quote_id: q.quote_id.clone(),
            price_residual: value - (q.bid + 0.5 * (q.ask - q.bid)),
            volatility_residual: point.volatility
                - row.mid_iv.as_ref().unwrap().volatility.unwrap(),
        });
        surface.diagnostics.checks += 1;
    }
    for (i, (a, k, _)) in observations.iter().enumerate() {
        for (b, other_k, _) in &observations[i + 1..] {
            if a.quote.input.maturity_years != b.quote.input.maturity_years || k != other_k {
                break;
            }
            surface.diagnostics.checks += 1;
            let a = &a.quote;
            let b = &b.quote;
            if a.input.strike == b.input.strike
                && matches!(
                    (a.input.side, b.input.side),
                    (OptionSide::Call, OptionSide::Put) | (OptionSide::Put, OptionSide::Call)
                )
            {
                let (call, put) = if matches!(a.input.side, OptionSide::Call) {
                    (a, b)
                } else {
                    (b, a)
                };
                let (sd, kd) = call.input.discounted()?;
                let residual = (call.bid + 0.5 * (call.ask - call.bid))
                    - (put.bid + 0.5 * (put.ask - put.bid))
                    - (sd - kd);
                if residual.abs() > tol {
                    surface.diagnostics.issues.push(DiagnosticIssue {
                        kind: "put_call_parity".into(),
                        maturity_years: a.input.maturity_years,
                        coordinate: a.input.strike,
                        residual,
                    });
                }
            }
        }
    }
    for slice in &surface.slices {
        let sd = surface.spot * (-slice.dividend_yield * slice.time).exp();
        let mut prev: Option<(f64, f64)> = None;
        let mut slope: Option<f64> = None;
        // Add midpoints: knot-only convexity does not examine interpolation.
        let mut grid: Vec<f64> = slice.knots.iter().map(|p| p.k).collect();
        grid.extend(
            slice
                .knots
                .windows(2)
                .map(|p| p[0].k + 0.5 * (p[1].k - p[0].k)),
        );
        grid.sort_by(f64::total_cmp);
        for k in grid {
            let x = finite(k.exp())?;
            let strike = finite(
                surface.spot * ((slice.rate - slice.dividend_yield) * slice.time + k).exp(),
            )?;
            let w = interpolate(&slice.knots, k)?.0;
            let c = european_price_and_vega(
                &EuropeanInputs {
                    spot: surface.spot,
                    strike,
                    maturity_years: slice.time,
                    rate: slice.rate,
                    dividend_yield: slice.dividend_yield,
                    side: OptionSide::Call,
                },
                (w / slice.time).sqrt(),
            )?
            .0 / sd;
            if let Some((px, pc)) = prev {
                let s = finite((c - pc) / (x - px))?;
                surface.diagnostics.checks += 1;
                if s > tol || s < -1.0 - tol {
                    surface.diagnostics.issues.push(DiagnosticIssue {
                        kind: "normalized_call_slope".into(),
                        maturity_years: slice.time,
                        coordinate: k,
                        residual: s,
                    });
                }
                if let Some(old) = slope {
                    surface.diagnostics.checks += 1;
                    if s < old - tol {
                        surface.diagnostics.issues.push(DiagnosticIssue {
                            kind: "strike_convexity".into(),
                            maturity_years: slice.time,
                            coordinate: k,
                            residual: s - old,
                        });
                    }
                }
                slope = Some(s);
            }
            prev = Some((x, c));
        }
    }
    for pair in surface.slices.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        for p in a.knots.iter().chain(&b.knots) {
            if let (Ok((wa, _)), Ok((wb, _))) =
                (interpolate(&a.knots, p.k), interpolate(&b.knots, p.k))
            {
                surface.diagnostics.checks += 1;
                if wb < wa - tol {
                    surface.diagnostics.issues.push(DiagnosticIssue {
                        kind: "calendar_total_variance".into(),
                        maturity_years: b.time,
                        coordinate: p.k,
                        residual: wb - wa,
                    });
                }
            }
        }
    }
    if surface.diagnostics.checks > config.max_diagnostic_checks {
        return Err(invalid("surface diagnostic cap exceeded"));
    }
    Ok(())
}
