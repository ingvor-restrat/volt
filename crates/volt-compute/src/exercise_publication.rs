//! Exercise-aware risk publication through Hypercube, without repricing.
//!
//! Inputs are completed `volt.options_risk_snapshot.v1` worker snapshots. Each
//! underlying may have a different risk cutoff after an incremental update.
//! Hypercube observation lineage is the oldest quote/stock/calibration input;
//! explicit nanosecond risk cutoffs remain in `underlyings`. If the calibration
//! stock timestamp is absent, observation lineage conservatively uses epoch zero
//! and marks the underlying lineage incomplete. Theta can be absent
//! at an exercise/dividend event and produces a missing cell and total.
use crate::{ComputeError, chunked::CompensatedSum, exercise_pricing::ExerciseRisk};
use hypercube::{ExecutionMode, HypercubeEngine, InputRow, NodeSpec, Snapshot, Transform, Update};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

const FIELDS: [&str; 14] = [
    "spot",
    "position_units",
    "price",
    "delta",
    "gamma",
    "vega_per_vol_point",
    "theta_per_day",
    "rho_per_bp",
    "value",
    "delta_dollars",
    "gamma_1pct",
    "position_vega_per_vol_point",
    "position_theta_per_day",
    "position_rho_per_bp",
];
const TOTALS: [&str; 6] = [
    "value",
    "delta_dollars",
    "gamma_1pct",
    "vega_per_vol_point",
    "theta_per_day",
    "rho_per_bp",
];
fn invalid(s: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn ns(s: &str) -> Result<i64, ComputeError> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid(
            "timestamps must be nonnegative decimal nanoseconds",
        ));
    }
    s.parse::<i64>()
        .ok()
        .filter(|n| *n >= 0)
        .ok_or_else(|| invalid("timestamps must be nonnegative decimal nanoseconds"))
}
fn identity(s: &str) -> bool {
    !s.is_empty() && s.len() <= 256 && !s.chars().any(char::is_control)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationRow {
    pub contract_id: String,
    pub underlying: String,
    pub spot: f64,
    pub quantity: i64,
    pub multiplier: u32,
    pub position_units: f64,
    pub quote_ns: String,
    pub spot_ns: String,
    pub calibration_ns: String,
    #[serde(default)]
    pub calibration_spot_ns: Option<String>,
    pub risk_asof_ns: String,
    pub risk: Option<ExerciseRisk>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationInput {
    pub schema: String,
    pub generation: u64,
    pub as_of_ns: String,
    pub currency: String,
    pub source: String,
    pub rows: Vec<PublicationRow>,
}
#[derive(Debug, Clone, Serialize)]
pub struct UnderlyingRiskClock {
    pub risk_asof_ns: String,
    pub risk_age_at_publication_seconds: f64,
    pub contracts: usize,
    pub calibration_stock_lineage_complete: bool,
}
#[derive(Debug, Serialize)]
pub struct ExercisePublication {
    pub schema: &'static str,
    pub generation: u64,
    pub as_of_ns: String,
    pub currency: String,
    pub source: String,
    pub underlyings: BTreeMap<String, UnderlyingRiskClock>,
    pub portfolio: BTreeMap<String, Option<f64>>,
    pub unpriced_positions: Vec<String>,
    pub cube: Snapshot,
}
/// A bounded, monotone native publisher. Failed validation leaves it unchanged.
/// Complete combined snapshots permit upstream incremental pricing while every
/// accepted Hypercube generation remains coherent and independently readable.
#[derive(Default)]
pub struct ExercisePublisher {
    cube: HypercubeEngine,
    last_cutoff: Option<i64>,
    currency: Option<String>,
    source: Option<String>,
}
impl ExercisePublisher {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn publish(
        &mut self,
        input: &PublicationInput,
    ) -> Result<ExercisePublication, ComputeError> {
        let cutoff = ns(&input.as_of_ns)?;
        if input.schema != "volt.options_risk_snapshot.v1"
            || input.generation == 0
            || input.generation > (1_u64 << 53) - 1
            || self.last_cutoff.is_some_and(|v| cutoff < v)
            || input.currency.len() != 3
            || !input.currency.bytes().all(|b| b.is_ascii_uppercase())
            || !["synthetic", "historical", "live"].contains(&input.source.as_str())
            || self.currency.as_ref().is_some_and(|v| v != &input.currency)
            || self.source.as_ref().is_some_and(|v| v != &input.source)
            || input.rows.is_empty()
            || input.rows.len() > 2048
        {
            return Err(invalid(
                "invalid publication schema, generation, cutoff, source, currency or row cap",
            ));
        }
        let mut seen = HashSet::new();
        let mut clocks: BTreeMap<String, UnderlyingRiskClock> = BTreeMap::new();
        let mut underlying_spots = BTreeMap::new();
        let mut rows = Vec::with_capacity(input.rows.len());
        let mut sums: [CompensatedSum; 6] = std::array::from_fn(|_| CompensatedSum::default());
        let mut theta_missing = false;
        let mut unpriced = Vec::new();
        for r in &input.rows {
            let risk_time = ns(&r.risk_asof_ns)?;
            let quote_time = ns(&r.quote_ns)?;
            let spot_time = ns(&r.spot_ns)?;
            let calibration_time = ns(&r.calibration_ns)?;
            let calibration_stock = r.calibration_spot_ns.as_deref().map(ns).transpose()?;
            let units = i128::from(r.quantity) * i128::from(r.multiplier);
            if !identity(&r.contract_id)
                || !identity(&r.underlying)
                || !seen.insert(&r.contract_id)
                || r.multiplier == 0
                || r.multiplier > 1_000_000
                || units.abs() > (1_i128 << 53) - 1
                || r.position_units != units as f64
                || !r.spot.is_finite()
                || r.spot <= 0.
                || risk_time > cutoff
                || quote_time > calibration_time
                || calibration_time > risk_time
                || calibration_stock.is_some_and(|v| v > calibration_time)
                || spot_time > risk_time
            {
                return Err(invalid(
                    "invalid row identity, units, spot or input/risk chronology",
                ));
            }
            if let Some(previous) = underlying_spots.insert(&r.underlying, (r.spot, spot_time))
                && previous != (r.spot, spot_time)
            {
                return Err(invalid(
                    "underlying stock observations must agree across contracts",
                ));
            }
            let clock = clocks
                .entry(r.underlying.clone())
                .or_insert(UnderlyingRiskClock {
                    risk_asof_ns: r.risk_asof_ns.clone(),
                    risk_age_at_publication_seconds: (cutoff - risk_time) as f64 / 1_000_000_000.,
                    contracts: 0,
                    calibration_stock_lineage_complete: true,
                });
            if ns(&clock.risk_asof_ns)? != risk_time {
                return Err(invalid("one underlying must share a risk cutoff"));
            }
            clock.contracts += 1;
            clock.calibration_stock_lineage_complete &= calibration_stock.is_some();
            let mut row = InputRow::new(
                &r.contract_id,
                quote_time
                    .min(spot_time)
                    .min(calibration_stock.unwrap_or(0))
                    / 1_000_000,
            )
            .with_field("spot", r.spot)
            .with_field("position_units", r.position_units);
            if let Some(g) = &r.risk {
                if g.valuation.contract_id != r.contract_id {
                    return Err(invalid("risk contract identity differs"));
                }
                let values = [
                    g.valuation.price,
                    g.delta,
                    g.gamma,
                    g.vega_per_vol_point,
                    g.rho_per_bp,
                ];
                if values.iter().any(|x| !x.is_finite())
                    || g.theta_per_day.is_some_and(|v| !v.is_finite())
                {
                    return Err(invalid("nonfinite risk"));
                }
                for (key, value) in [
                    ("price", g.valuation.price),
                    ("delta", g.delta),
                    ("gamma", g.gamma),
                    ("vega_per_vol_point", g.vega_per_vol_point),
                    ("rho_per_bp", g.rho_per_bp),
                ] {
                    row.fields.insert(key.into(), value);
                }
                if let Some(t) = g.theta_per_day {
                    row.fields.insert("theta_per_day".into(), t);
                }
                let weighted = [
                    r.position_units * g.valuation.price,
                    r.position_units * g.delta * r.spot,
                    0.5 * r.position_units * g.gamma * (r.spot * 0.01).powi(2),
                    r.position_units * g.vega_per_vol_point,
                    r.position_units * g.theta_per_day.unwrap_or(0.),
                    r.position_units * g.rho_per_bp,
                ];
                if weighted.iter().any(|x| !x.is_finite()) {
                    return Err(invalid("position risk overflow"));
                }
                for (i, v) in weighted.into_iter().enumerate() {
                    if i != 4 || g.theta_per_day.is_some() {
                        row.fields.insert(FIELDS[8 + i].into(), v);
                    }
                    sums[i].add(v);
                }
                theta_missing |= r.position_units != 0. && g.theta_per_day.is_none();
            } else if r.position_units != 0. {
                unpriced.push(r.contract_id.clone());
            }
            rows.push(row);
        }
        let mut portfolio = BTreeMap::new();
        for (i, (name, sum)) in TOTALS.into_iter().zip(sums).enumerate() {
            let v = sum.value();
            if !v.is_finite() {
                return Err(invalid("portfolio risk overflow"));
            }
            portfolio.insert(
                name.into(),
                if !unpriced.is_empty() || (i == 4 && theta_missing) {
                    None
                } else {
                    Some(v)
                },
            );
        }
        let cube = self
            .cube
            .update(Update {
                generation: input.generation,
                observed_at_ms: cutoff / 1_000_000,
                mode: ExecutionMode::Live,
                rows,
                nodes: FIELDS
                    .iter()
                    .map(|f| NodeSpec::field(*f, *f, Transform::Identity))
                    .collect(),
            })
            .map_err(|e| invalid(e.to_string()))?;
        self.last_cutoff = Some(cutoff);
        self.currency = Some(input.currency.clone());
        self.source = Some(input.source.clone());
        Ok(ExercisePublication {
            schema: "volt.exercise_hypercube_publication.v1",
            generation: input.generation,
            as_of_ns: input.as_of_ns.clone(),
            currency: input.currency.clone(),
            source: input.source.clone(),
            underlyings: clocks,
            portfolio,
            unpriced_positions: unpriced,
            cube,
        })
    }
}
