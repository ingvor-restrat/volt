//! Quote-to-risk application boundary. A complete input frame is calculated and
//! published as one Hypercube generation; failed or stale updates do not replace it.
use crate::{
    ComputeError, OptionRow, OptionSide,
    aad::analytic_greeks,
    chunked::CompensatedSum,
    volatility_surface::{
        self as surface, ChainConfig, EuropeanInputs, GreekConvention, IvConfig, ProcessedQuote,
        QuoteRow, SurfaceConfig, VolatilitySurface,
    },
};
use hypercube::{ExecutionMode, HypercubeEngine, InputRow, NodeSpec, Snapshot, Transform, Update};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
fn invalid(s: &str) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn error(e: impl std::fmt::Display) -> ComputeError {
    ComputeError::Execution(e.to_string())
}
struct HashWriter {
    digest: Sha256,
    remaining: usize,
}
impl std::io::Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.remaining {
            return Err(std::io::Error::other("quote frame byte cap exceeded"));
        }
        self.remaining -= bytes.len();
        self.digest.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn hash<T: Serialize>(x: &T, limit: usize) -> Result<String, ComputeError> {
    let mut w = HashWriter {
        digest: Sha256::new(),
        remaining: limit,
    };
    serde_json::to_writer(&mut w, x).map_err(error)?;
    Ok(format!("{:x}", w.digest.finalize()))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LivePosition {
    pub contract_id: String,
    pub quantity: i64,
    pub multiplier: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuoteFrame {
    pub schema: String,
    pub generation: u64,
    /// Every remaining maturity in this frame is measured at this valuation cutoff.
    pub cutoff_ns: i64,
    pub currency: String,
    pub position_revision: u64,
    pub synthetic: bool,
    pub quotes: Vec<QuoteRow>,
    pub positions: Vec<LivePosition>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveRiskConfig {
    pub max_quotes: usize,
    pub max_request_bytes: usize,
    pub max_age_ns: u64,
    pub convention: GreekConvention,
    pub relative_spot_scenario: f64,
    pub volatility_scenario: f64,
}
impl Default for LiveRiskConfig {
    fn default() -> Self {
        Self {
            max_quotes: 2048,
            max_request_bytes: 2 * 1024 * 1024,
            max_age_ns: 5_000_000_000,
            convention: GreekConvention::FixedVolatility,
            relative_spot_scenario: 0.01,
            volatility_scenario: 0.01,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct LiveGreeks {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega_per_vol_point: f64,
    pub theta_per_day: f64,
    pub rho_per_bp: f64,
}
#[derive(Debug, Clone, Serialize)]
pub struct RiskRow {
    pub quote: ProcessedQuote,
    pub position_units: f64,
    pub pricing_basis: String,
    pub greeks: Option<LiveGreeks>,
    pub calculation_error: Option<String>,
    pub position_value: Option<f64>,
    pub position_delta: Option<f64>,
    pub position_vega: Option<f64>,
}
#[derive(Debug, Clone, Serialize)]
pub struct PortfolioRisk {
    pub complete: bool,
    pub unpriced_contracts: Vec<String>,
    pub position_count: usize,
    pub value: Option<f64>,
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub vega_per_vol_point: Option<f64>,
    pub theta_per_day: Option<f64>,
    pub rho_per_bp: Option<f64>,
}
#[derive(Debug, Clone, Serialize)]
pub struct RiskScenario {
    pub name: String,
    pub spot_change: f64,
    pub volatility_change: f64,
    pub pnl: Option<f64>,
    pub local_pnl: Option<f64>,
    pub unavailable_contracts: Vec<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct SurfaceGridPoint {
    pub maturity_years: f64,
    pub log_moneyness: f64,
    pub strike: f64,
    pub volatility: f64,
}
#[derive(Debug, Clone, Serialize)]
pub struct RiskTimings {
    pub inversion_ms: f64,
    pub surface_ms: f64,
    pub greeks_and_scenarios_ms: f64,
    pub cube_ms: f64,
    pub total_ms: f64,
}
#[derive(Debug, Clone, Serialize)]
pub struct LiveRiskSnapshot {
    pub schema: String,
    pub generation: u64,
    pub cutoff_ns: i64,
    pub calculated_at_ns: i64,
    pub currency: String,
    pub underlying_id: String,
    pub spot: f64,
    pub position_revision: u64,
    pub synthetic: bool,
    pub convention: GreekConvention,
    pub model_partial_units: String,
    pub input_sha256: String,
    pub positions_sha256: String,
    pub rows: Vec<RiskRow>,
    pub portfolio: PortfolioRisk,
    pub scenarios: Vec<RiskScenario>,
    pub surface: Option<VolatilitySurface>,
    pub surface_error: Option<String>,
    pub grid: Vec<SurfaceGridPoint>,
    pub timings: RiskTimings,
    pub cube: Snapshot,
}
/// The engine accepts only strictly newer complete quote/book frames. Keeping
/// source time and portfolio revision together avoids joining unrelated widgets.
pub struct LiveRiskEngine {
    config: LiveRiskConfig,
    cube: HypercubeEngine,
    last_generation: u64,
    last_cutoff_ns: i64,
    last_position_revision: u64,
    last_positions_sha256: Option<String>,
}
impl LiveRiskEngine {
    pub fn new(config: LiveRiskConfig) -> Result<Self, ComputeError> {
        if config.max_quotes == 0
            || config.max_quotes > 100_000
            || config.max_request_bytes == 0
            || config.max_request_bytes > 64 * 1024 * 1024
            || !config.relative_spot_scenario.is_finite()
            || !(0.0..0.25).contains(&config.relative_spot_scenario)
            || !config.volatility_scenario.is_finite()
            || !(0.0..0.5).contains(&config.volatility_scenario)
        {
            return Err(invalid("invalid live risk limits/scenario shocks"));
        }
        if let GreekConvention::StickyLogMoneynessFiniteBump { relative_spot_bump } =
            config.convention
            && (!relative_spot_bump.is_finite() || !(1e-7..=0.1).contains(&relative_spot_bump))
        {
            return Err(invalid("invalid surface spot bump"));
        }
        Ok(Self {
            config,
            cube: HypercubeEngine::new(),
            last_generation: 0,
            last_cutoff_ns: 0,
            last_position_revision: 0,
            last_positions_sha256: None,
        })
    }
    /// Publish a fully calculated frame transactionally. On publication failure,
    /// the previously accepted generation/book remains current. A compiled cube
    /// plan can be rebuilt on retry; numerical inputs and outputs are unchanged.
    pub fn update_and_publish<F>(
        &mut self,
        frame: QuoteFrame,
        publish: F,
    ) -> Result<LiveRiskSnapshot, ComputeError>
    where
        F: FnOnce(&LiveRiskSnapshot) -> Result<(), ComputeError>,
    {
        let previous = (
            self.last_generation,
            self.last_cutoff_ns,
            self.last_position_revision,
            self.last_positions_sha256.clone(),
        );
        let next = self.update(frame)?;
        if let Err(e) = publish(&next) {
            self.cube = HypercubeEngine::new();
            (
                self.last_generation,
                self.last_cutoff_ns,
                self.last_position_revision,
                self.last_positions_sha256,
            ) = previous;
            return Err(e);
        }
        Ok(next)
    }
    pub fn config(&self) -> &LiveRiskConfig {
        &self.config
    }
    pub fn update(&mut self, frame: QuoteFrame) -> Result<LiveRiskSnapshot, ComputeError> {
        let started = Instant::now();
        if frame.schema != "volt.compute.quote_frame.v1"
            || frame.generation <= self.last_generation
            || frame.cutoff_ns < 0
            || frame.cutoff_ns < self.last_cutoff_ns
            || frame.position_revision < self.last_position_revision
            || frame.quotes.is_empty()
            || frame.quotes.len() > self.config.max_quotes
            || frame.positions.len() > self.config.max_quotes
            || frame.currency.len() != 3
            || !frame.currency.bytes().all(|b| b.is_ascii_uppercase())
        {
            return Err(invalid("invalid or stale quote frame"));
        }
        let bytes = frame
            .quotes
            .iter()
            .try_fold(0usize, |n, q| {
                n.checked_add(q.quote_id.len())?
                    .checked_add(q.contract_id.len())?
                    .checked_add(q.source_id.len())?
                    .checked_add(q.underlying_id.len())
            })
            .and_then(|n| {
                frame
                    .positions
                    .iter()
                    .try_fold(n, |n, p| n.checked_add(p.contract_id.len()))
            })
            .ok_or_else(|| invalid("quote identity byte overflow"))?;
        if bytes > self.config.max_request_bytes {
            return Err(invalid("quote frame exceeds byte limit"));
        }
        let mut contracts = HashSet::new();
        for q in &frame.quotes {
            if !contracts.insert(q.contract_id.as_str()) {
                return Err(invalid(
                    "quote frame must contain one normalized quote per contract",
                ));
            }
        }
        let mut positions = BTreeMap::new();
        for p in &frame.positions {
            if p.multiplier == 0
                || p.multiplier > 1_000_000
                || p.quantity.unsigned_abs() > 1_000_000_000
                || !contracts.contains(p.contract_id.as_str())
                || positions
                    .insert(
                        p.contract_id.clone(),
                        p.quantity as f64 * p.multiplier as f64,
                    )
                    .is_some()
            {
                return Err(invalid("invalid, duplicate or unquoted portfolio contract"));
            }
        }
        let input_sha256 = hash(&frame, self.config.max_request_bytes)?;
        let positions_sha256 = hash(&frame.positions, self.config.max_request_bytes)?;
        if self
            .last_positions_sha256
            .as_ref()
            .is_some_and(|h| h != &positions_sha256)
            && frame.position_revision <= self.last_position_revision
        {
            return Err(invalid(
                "changed portfolio requires a newer position revision",
            ));
        }
        let quote_start = Instant::now();
        let processed = surface::process_chain(
            &frame.quotes,
            &ChainConfig {
                cutoff_ns: frame.cutoff_ns,
                max_age_ns: self.config.max_age_ns,
                max_quotes: self.config.max_quotes,
                max_input_bytes: self.config.max_request_bytes,
                iv: IvConfig::default(),
            },
        )?;
        // A frame has one underlying and spot, including quotes rejected for age.
        let first = &frame.quotes[0];
        if !first.input.spot.is_finite()
            || first.input.spot <= 0.0
            || first.underlying_id.trim().is_empty()
            || frame.quotes.iter().any(|q| {
                q.underlying_id != first.underlying_id
                    || q.input.spot.to_bits() != first.input.spot.to_bits()
            })
        {
            return Err(invalid(
                "live frame requires a single underlying and common spot snapshot",
            ));
        }
        let inversion_ms = quote_start.elapsed().as_secs_f64() * 1000.0;
        let fit_start = Instant::now();
        let (fitted, surface_error) = match surface::fit_surface(
            &processed,
            &SurfaceConfig {
                max_quotes: self.config.max_quotes,
                max_knots: self.config.max_quotes,
                max_input_bytes: self.config.max_request_bytes,
                max_diagnostic_checks: self.config.max_quotes.saturating_mul(128).min(10_000_000),
                ..SurfaceConfig::default()
            },
        ) {
            Ok(s) => (Some(s), None),
            Err(e) => (None, Some(e.to_string())),
        };
        let surface_ms = fit_start.elapsed().as_secs_f64() * 1000.0;
        let risk_start = Instant::now();
        let mut rows = Vec::with_capacity(processed.len());
        for p in processed {
            let units = positions.get(&p.quote.contract_id).copied().unwrap_or(0.0);
            let (mut greeks, pricing_basis, mut calculation_error) =
                match quote_risk(&p, fitted.as_ref(), self.config.convention) {
                    Ok((g, b)) => (Some(g), b.to_string(), None),
                    Err(e) => (None, "unavailable".into(), Some(e.to_string())),
                };
            if greeks.as_ref().is_some_and(|g| {
                [
                    g.price,
                    g.delta,
                    g.gamma,
                    g.vega_per_vol_point,
                    g.theta_per_day,
                    g.rho_per_bp,
                ]
                .iter()
                .any(|x| !x.is_finite() || !(x * units).is_finite())
            }) {
                greeks = None;
                calculation_error = Some("risk calculation or weighted position overflow".into());
            }
            rows.push(RiskRow {
                position_value: greeks.as_ref().map(|g| g.price * units),
                position_delta: greeks.as_ref().map(|g| g.delta * units),
                position_vega: greeks.as_ref().map(|g| g.vega_per_vol_point * units),
                quote: p,
                position_units: units,
                greeks,
                pricing_basis,
                calculation_error,
            });
        }
        let portfolio = aggregate(&rows, frame.positions.len())?;
        let scenarios = make_scenarios(&rows, fitted.as_ref(), &self.config)?;
        let grid = make_grid(&rows, fitted.as_ref());
        let greeks_and_scenarios_ms = risk_start.elapsed().as_secs_f64() * 1000.0;
        let cube_start = Instant::now();
        let node_names = [
            "bid",
            "ask",
            "mid_iv",
            "delta",
            "gamma",
            "vega_per_vol_point",
            "theta_per_day",
            "position_value",
            "position_delta",
            "position_vega",
            "age_seconds",
        ];
        let oldest_surface_time = fitted
            .as_ref()
            .and_then(|s| s.source_observations.iter().map(|x| x.timestamp_ns).min());
        let inputs = rows
            .iter()
            .map(|r| {
                let q = &r.quote.quote;
                let lineage_time = if r.pricing_basis == "surface" {
                    oldest_surface_time
                        .unwrap_or(q.timestamp_ns)
                        .min(q.timestamp_ns)
                } else {
                    q.timestamp_ns
                };
                let mut row = InputRow::new(&q.contract_id, lineage_time / 1_000_000);
                for (k, v) in [
                    ("bid", Some(q.bid)),
                    ("ask", Some(q.ask)),
                    (
                        "mid_iv",
                        r.quote.mid_iv.as_ref().and_then(|iv| iv.volatility),
                    ),
                    ("position_value", r.position_value),
                    ("position_delta", r.position_delta),
                    ("position_vega", r.position_vega),
                    (
                        "age_seconds",
                        (frame.cutoff_ns >= q.timestamp_ns).then_some(
                            (i128::from(frame.cutoff_ns) - i128::from(q.timestamp_ns)) as f64 / 1e9,
                        ),
                    ),
                ] {
                    if let Some(v) = v.filter(|x| x.is_finite()) {
                        row.fields.insert(k.into(), v);
                    }
                }
                if let Some(g) = &r.greeks {
                    for (k, v) in [
                        ("delta", g.delta),
                        ("gamma", g.gamma),
                        ("vega_per_vol_point", g.vega_per_vol_point),
                        ("theta_per_day", g.theta_per_day),
                    ] {
                        row.fields.insert(k.into(), v);
                    }
                }
                row
            })
            .collect();
        let cube = self
            .cube
            .update(Update {
                generation: frame.generation,
                observed_at_ms: frame.cutoff_ns / 1_000_000,
                mode: ExecutionMode::Live,
                rows: inputs,
                nodes: node_names
                    .into_iter()
                    .map(|n| NodeSpec::field(n, n, Transform::Identity))
                    .collect(),
            })
            .map_err(error)?;
        let cube_ms = cube_start.elapsed().as_secs_f64() * 1000.0;
        self.last_generation = frame.generation;
        self.last_cutoff_ns = frame.cutoff_ns;
        self.last_position_revision = frame.position_revision;
        self.last_positions_sha256 = Some(positions_sha256.clone());
        Ok(LiveRiskSnapshot {schema:"volt.compute.live_risk.v1".into(),generation:frame.generation,cutoff_ns:frame.cutoff_ns,
            calculated_at_ns:now_ns(),currency:frame.currency,underlying_id:first.underlying_id.clone(),spot:first.input.spot,
            position_revision:frame.position_revision,synthetic:frame.synthetic,convention:self.config.convention,
            model_partial_units:"Vega per +1 volatility percentage point; theta per calendar day; rho per +1 basis point. Only spot delta/gamma follow the selected surface-motion convention.".into(),
            input_sha256,positions_sha256,rows,portfolio,scenarios,surface:fitted,surface_error,grid,
            timings:RiskTimings {inversion_ms,surface_ms,greeks_and_scenarios_ms,cube_ms,total_ms:started.elapsed().as_secs_f64()*1000.0},cube})
    }
}
fn quote_risk(
    p: &ProcessedQuote,
    s: Option<&VolatilitySurface>,
    convention: GreekConvention,
) -> Result<(LiveGreeks, &'static str), ComputeError> {
    if !p.eligible {
        return Err(invalid("quote is ineligible for risk"));
    }
    let q = &p.quote;
    if let Some(s) = s {
        let g = s.greeks(
            q.input.strike,
            q.input.maturity_years,
            q.input.side,
            convention,
        )?;
        let x = g.fixed_volatility_partials;
        let input = EuropeanInputs {
            spot: g.point.spot,
            strike: g.point.strike,
            maturity_years: g.point.maturity_years,
            rate: g.point.rate,
            dividend_yield: g.point.dividend_yield,
            side: q.input.side,
        };
        let price = surface::european_price_and_vega(&input, g.point.volatility)?.0;
        return Ok((
            LiveGreeks {
                price,
                delta: g.spot_delta,
                gamma: g.spot_gamma,
                vega_per_vol_point: x.vega * 0.01,
                theta_per_day: x.theta / 365.0,
                rho_per_bp: x.rho * 0.0001,
            },
            "surface",
        ));
    }
    if !matches!(convention, GreekConvention::FixedVolatility) {
        return Err(invalid("surface-motion risk needs a fitted surface"));
    }
    let iv = p
        .mid_iv
        .as_ref()
        .and_then(|x| x.volatility)
        .ok_or_else(|| invalid("missing implied volatility"))?;
    let x = analytic_greeks(&OptionRow {
        row_id: q.quote_id.clone(),
        contract_id: q.contract_id.clone(),
        side: q.input.side,
        spot: q.input.spot,
        strike: q.input.strike,
        maturity_years: q.input.maturity_years,
        rate: q.input.rate,
        dividend_yield: q.input.dividend_yield,
        volatility: iv,
        quantity: 1,
        multiplier: 1,
    })?;
    let price = surface::european_price_and_vega(&q.input, iv)?.0;
    Ok((
        LiveGreeks {
            price,
            delta: x.delta,
            gamma: x.vega / q.input.spot / q.input.spot / iv / q.input.maturity_years,
            vega_per_vol_point: x.vega * 0.01,
            theta_per_day: x.theta / 365.0,
            rho_per_bp: x.rho * 0.0001,
        },
        "quote_iv",
    ))
}
fn aggregate(rows: &[RiskRow], position_count: usize) -> Result<PortfolioRisk, ComputeError> {
    let mut sums: Vec<CompensatedSum> = (0..6).map(|_| CompensatedSum::default()).collect();
    let mut missing = Vec::new();
    for r in rows {
        if r.position_units == 0.0 {
            continue;
        }
        if let Some(g) = &r.greeks {
            for (s, v) in sums.iter_mut().zip([
                g.price,
                g.delta,
                g.gamma,
                g.vega_per_vol_point,
                g.theta_per_day,
                g.rho_per_bp,
            ]) {
                s.add(v * r.position_units);
            }
        } else {
            missing.push(r.quote.quote.contract_id.clone());
        }
    }
    let v: Vec<f64> = sums.into_iter().map(CompensatedSum::value).collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err(invalid("portfolio aggregation overflow"));
    }
    let complete = missing.is_empty();
    Ok(PortfolioRisk {
        complete,
        unpriced_contracts: missing,
        position_count,
        value: complete.then_some(v[0]),
        delta: complete.then_some(v[1]),
        gamma: complete.then_some(v[2]),
        vega_per_vol_point: complete.then_some(v[3]),
        theta_per_day: complete.then_some(v[4]),
        rho_per_bp: complete.then_some(v[5]),
    })
}
fn make_scenarios(
    rows: &[RiskRow],
    surface: Option<&VolatilitySurface>,
    c: &LiveRiskConfig,
) -> Result<Vec<RiskScenario>, ComputeError> {
    let mut out = Vec::new();
    for (name, ds, dv) in [
        ("Spot up", c.relative_spot_scenario, 0.0),
        ("Spot down", -c.relative_spot_scenario, 0.0),
        ("Volatility up", 0.0, c.volatility_scenario),
        ("Volatility down", 0.0, -c.volatility_scenario),
    ] {
        let mut pnl = CompensatedSum::default();
        let mut local = CompensatedSum::default();
        let mut unavailable = Vec::new();
        for r in rows.iter().filter(|r| r.position_units != 0.0) {
            let q = &r.quote.quote;
            let result = (|| -> Result<(f64, f64), ComputeError> {
                let g = r
                    .greeks
                    .as_ref()
                    .ok_or_else(|| invalid("unpriced scenario position"))?;
                let (vol, rate, dividend_yield) = if let Some(s) = surface {
                    let point = if matches!(c.convention, GreekConvention::FixedVolatility) {
                        s.query(q.input.strike, q.input.maturity_years)?
                    } else {
                        s.query_at_spot(
                            q.input.spot * (1.0 + ds),
                            q.input.strike,
                            q.input.maturity_years,
                        )?
                    };
                    (point.volatility, point.rate, point.dividend_yield)
                } else {
                    (
                        r.quote
                            .mid_iv
                            .as_ref()
                            .and_then(|iv| iv.volatility)
                            .ok_or_else(|| invalid("missing IV"))?,
                        q.input.rate,
                        q.input.dividend_yield,
                    )
                };
                if vol + dv <= 0.0 {
                    return Err(invalid("nonpositive shocked volatility"));
                }
                let input = EuropeanInputs {
                    spot: q.input.spot * (1.0 + ds),
                    rate,
                    dividend_yield,
                    ..q.input
                };
                let price = surface::european_price_and_vega(&input, vol + dv)?.0;
                let delta_s = q.input.spot * ds;
                Ok((
                    r.position_units * (price - g.price),
                    r.position_units
                        * (g.delta * delta_s
                            + 0.5 * g.gamma * delta_s * delta_s
                            + g.vega_per_vol_point * dv / 0.01),
                ))
            })();
            match result {
                Ok((p, a)) if p.is_finite() && a.is_finite() => {
                    pnl.add(p);
                    local.add(a);
                }
                _ => unavailable.push(q.contract_id.clone()),
            }
        }
        let p = pnl.value();
        let a = local.value();
        let complete = unavailable.is_empty() && p.is_finite() && a.is_finite();
        out.push(RiskScenario {
            name: name.into(),
            spot_change: ds,
            volatility_change: dv,
            pnl: complete.then_some(p),
            local_pnl: complete.then_some(a),
            unavailable_contracts: unavailable,
        });
    }
    Ok(out)
}
fn make_grid(rows: &[RiskRow], surface: Option<&VolatilitySurface>) -> Vec<SurfaceGridPoint> {
    let Some(s) = surface else {
        return Vec::new();
    };
    let mut expiries: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.quote.eligible) {
        let i = r.quote.quote.input;
        let k = (i.strike / i.spot).ln() - (i.rate - i.dividend_yield) * i.maturity_years;
        expiries
            .entry(i.maturity_years.to_bits())
            .or_default()
            .push(k);
    }
    let Some((&lo, _)) = expiries.first_key_value() else {
        return Vec::new();
    };
    let hi = *expiries.last_key_value().unwrap().0;
    let kmin = expiries
        .values()
        .map(|xs| xs.iter().copied().fold(f64::INFINITY, f64::min))
        .fold(f64::NEG_INFINITY, f64::max);
    let kmax = expiries
        .values()
        .map(|xs| xs.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .fold(f64::INFINITY, f64::min);
    if kmin > kmax {
        return Vec::new();
    }
    let t0 = f64::from_bits(lo);
    let t1 = f64::from_bits(hi);
    let mut out = Vec::new();
    for j in 0..=12 {
        let t = t0 + (t1 - t0) * j as f64 / 12.0;
        for i in 0..=24 {
            let k = kmin + (kmax - kmin) * i as f64 / 24.0;
            // Carry integrals are learned from the surface; get forward using one
            // supported strike, then map the common k grid back to strike.
            let anchor = rows
                .iter()
                .filter(|r| r.quote.eligible)
                .map(|r| r.quote.quote.input.strike)
                .find_map(|strike| s.query(strike, t).ok());
            if let Some(a) = anchor {
                let strike = s.spot() * ((a.rate - a.dividend_yield) * t + k).exp();
                if let Ok(p) = s.query(strike, t) {
                    out.push(SurfaceGridPoint {
                        maturity_years: t,
                        log_moneyness: k,
                        strike,
                        volatility: p.volatility,
                    });
                }
            }
        }
    }
    out
}
pub fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(i64::MAX as u128) as i64
}
/// Deterministic evolving quotes for a demonstrator, never labelled live market data.
pub fn demo_frame(generation: u64, cutoff_ns: i64) -> Result<QuoteFrame, ComputeError> {
    let day_ns = 86_400_000_000_000_i64;
    let day = cutoff_ns.div_euclid(day_ns);
    let anchor = day
        .checked_mul(day_ns)
        .ok_or_else(|| invalid("demo clock overflow"))?;
    let year_ns = 365.0 * day_ns as f64;
    let phase = generation as f64 * 0.07;
    let spot = 100.0 + phase.sin();
    let mut quotes = Vec::new();
    for (e, nominal_t) in [0.25_f64, 0.5, 1.0].into_iter().enumerate() {
        let t = nominal_t - (cutoff_ns - anchor) as f64 / year_ns;
        for strike in [80., 85., 90., 95., 100., 105., 110., 115., 120.] {
            for side in [OptionSide::Call, OptionSide::Put] {
                let kind = if matches!(side, OptionSide::Call) {
                    "C"
                } else {
                    "P"
                };
                let contract_id = format!("DEMO-{day}-{e}-{kind}-{strike}");
                let input = EuropeanInputs {
                    spot,
                    strike,
                    maturity_years: t,
                    rate: 0.03,
                    dividend_yield: 0.01,
                    side,
                };
                let k = (strike / spot).ln() - 0.02 * t;
                let sigma = 0.20 - 0.12 * k + 0.10 * k * k + 0.005 * t + 0.003 * phase.cos();
                let mid = surface::european_price_and_vega(&input, sigma)?.0;
                let width = (mid * 0.002).max(0.005);
                quotes.push(QuoteRow {
                    quote_id: format!("{contract_id}-{generation}"),
                    contract_id,
                    source_id: "deterministic-demo".into(),
                    underlying_id: "DEMO".into(),
                    timestamp_ns: cutoff_ns,
                    input,
                    bid: (mid - width).max(0.0),
                    ask: mid + width,
                });
            }
        }
    }
    Ok(QuoteFrame {
        schema: "volt.compute.quote_frame.v1".into(),
        generation,
        cutoff_ns,
        currency: "USD".into(),
        position_revision: 1,
        synthetic: true,
        quotes,
        positions: vec![
            LivePosition {
                contract_id: format!("DEMO-{day}-1-C-100"),
                quantity: -10,
                multiplier: 100,
            },
            LivePosition {
                contract_id: format!("DEMO-{day}-1-P-100"),
                quantity: -10,
                multiplier: 100,
            },
        ],
    })
}
