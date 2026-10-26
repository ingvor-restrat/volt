//! Financial local-risk lot allocation, distinct from rollout selection.
//!
//! Each bit selects a prequoted signed contract lot. The exported upper-
//! triangular QUBO encodes normalized squared delta/gamma/vega residuals plus
//! linear fees. Cash/inventory/one-choice constraints are hard-filtered by the
//! bounded exact enumerator, separately from the objective matrix.
use crate::ComputeError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
fn invalid(s: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn finite(v: f64, s: &str) -> Result<f64, ComputeError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(invalid(format!("nonfinite lot allocation {s}")))
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskVector {
    pub delta: f64,
    pub gamma: f64,
    pub vega_per_vol_point: f64,
}
impl RiskVector {
    fn values(self) -> [f64; 3] {
        [self.delta, self.gamma, self.vega_per_vol_point]
    }
    fn from(v: [f64; 3]) -> Self {
        Self {
            delta: v[0],
            gamma: v[1],
            vega_per_vol_point: v[2],
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskContext {
    pub underlying_id: String,
    pub currency: String,
    pub generation: u64,
    pub as_of_ns: i64,
    pub max_quote_age_ns: u64,
    pub greek_convention: String,
    pub provenance: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuotedInstrument {
    pub instrument_id: String,
    pub quote_id: String,
    pub underlying_id: String,
    pub currency: String,
    pub generation: u64,
    pub observed_at_ns: i64,
    pub greek_convention: String,
    pub eligible: bool,
    /// Price and Greeks are per underlying unit; multiplier converts one contract.
    pub mark: f64,
    pub per_unit_risk: RiskVector,
    pub multiplier: u64,
    pub current_contracts: i64,
    pub min_contracts: i64,
    pub max_contracts: i64,
    pub proportional_fee: f64,
    pub fixed_fee_per_selected_lot: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedLot {
    pub lot_id: String,
    pub instrument: usize,
    pub contracts: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HedgeLotProblem {
    pub context: RiskContext,
    pub instruments: Vec<QuotedInstrument>,
    pub lots: Vec<SignedLot>,
    pub current_risk: RiskVector,
    pub target_risk: RiskVector,
    /// Scales carry the same respective units as delta/gamma/vega-per-vol-point.
    pub risk_scales: [f64; 3],
    pub risk_weights: [f64; 3],
    pub fee_weight: f64,
    pub cash_before: f64,
    pub cash_borrowing_limit: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuboObjective {
    pub constant: f64,
    pub upper: Vec<Vec<f64>>,
    pub lot_ids: Vec<String>,
    pub problem_sha256: String,
    pub feasibility_constraints_encoded: bool,
    pub convention: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableLot {
    pub lot_id: String,
    pub instrument_id: String,
    pub source_quote_id: String,
    pub contracts: i64,
    pub underlying_units: f64,
    pub signed_purchase_notional: f64,
    pub fee: f64,
    pub cash_after: f64,
    pub inventory_after: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LotEvaluation {
    pub mask: u64,
    pub feasible: bool,
    pub reasons: Vec<String>,
    pub objective: f64,
    pub post_trade_risk: RiskVector,
    pub residual_risk: RiskVector,
    pub fees: f64,
    pub cash_after: f64,
    pub final_contracts: Vec<i64>,
    pub execution: Vec<ExecutableLot>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LotSolution {
    pub objective_model: QuboObjective,
    pub selected: LotEvaluation,
    pub enumerated_masks: u64,
    pub feasible_masks: u64,
    pub max_expansion_discrepancy: f64,
    pub method: String,
}
fn clean(s: &str) -> bool {
    !s.trim().is_empty() && s.len() <= 512 && !s.chars().any(char::is_control)
}
impl HedgeLotProblem {
    pub fn validate(&self) -> Result<(), ComputeError> {
        let n = self.lots.len();
        let c = &self.context;
        if !(1..=16).contains(&n)
            || self.instruments.is_empty()
            || self.instruments.len() > 16
            || ![
                &c.underlying_id,
                &c.currency,
                &c.greek_convention,
                &c.provenance,
            ]
            .into_iter()
            .all(|s| clean(s))
            || c.generation == 0
        {
            return Err(invalid(
                "lot problem requires 1..16 instruments/lots and complete context",
            ));
        }
        if self
            .current_risk
            .values()
            .into_iter()
            .chain(self.target_risk.values())
            .any(|x| !x.is_finite())
            || self.risk_scales.iter().any(|x| !x.is_finite() || *x <= 0.)
            || self.risk_weights.iter().any(|x| !x.is_finite() || *x < 0.)
            || !self.fee_weight.is_finite()
            || self.fee_weight < 0.
            || !self.cash_before.is_finite()
            || !self.cash_borrowing_limit.is_finite()
            || self.cash_borrowing_limit < 0.
            || self.cash_before < -self.cash_borrowing_limit
        {
            return Err(invalid(
                "invalid risk scales/weights or initial cash allowance",
            ));
        }
        let mut ids = std::collections::HashSet::new();
        let mut quote_ids = std::collections::HashSet::new();
        for i in &self.instruments {
            if !clean(&i.instrument_id)
                || !clean(&i.quote_id)
                || !ids.insert(&i.instrument_id)
                || !quote_ids.insert(&i.quote_id)
                || i.underlying_id != c.underlying_id
                || i.currency != c.currency
                || i.generation != c.generation
                || i.greek_convention != c.greek_convention
                || !i.eligible
                || i.observed_at_ns > c.as_of_ns
                || c.as_of_ns
                    .checked_sub(i.observed_at_ns)
                    .is_none_or(|age| age as u64 > c.max_quote_age_ns)
                || !i.mark.is_finite()
                || i.mark < 0.
                || i.per_unit_risk.values().iter().any(|x| !x.is_finite())
                || i.multiplier == 0
                || i.multiplier > 1_000_000
                || i.min_contracts > i.current_contracts
                || i.current_contracts > i.max_contracts
                || i.min_contracts < -1_000_000
                || i.max_contracts > 1_000_000
                || !i.proportional_fee.is_finite()
                || i.proportional_fee < 0.
                || !i.fixed_fee_per_selected_lot.is_finite()
                || i.fixed_fee_per_selected_lot < 0.
            {
                return Err(invalid(
                    "invalid/stale/incoherent instrument quote or inventory bound",
                ));
            }
        }
        let mut ids = std::collections::HashSet::new();
        for l in &self.lots {
            if !clean(&l.lot_id)
                || !ids.insert(&l.lot_id)
                || l.instrument >= self.instruments.len()
                || l.contracts == 0
                || l.contracts.unsigned_abs() > 1_000_000
            {
                return Err(invalid("invalid signed contract lot"));
            }
            self.lot_data(l)?;
        }
        Ok(())
    }
    fn lot_data(&self, l: &SignedLot) -> Result<([f64; 3], f64, f64), ComputeError> {
        let q = &self.instruments[l.instrument];
        let units = l.contracts as f64 * q.multiplier as f64;
        let purchase = finite(units * q.mark, "purchase notional")?;
        let fee = finite(
            purchase.abs() * q.proportional_fee + q.fixed_fee_per_selected_lot,
            "fee",
        )?;
        let raw = q.per_unit_risk.values();
        let mut risk = [0.; 3];
        for i in 0..3 {
            risk[i] = finite(raw[i] * units, "lot risk")?;
        }
        Ok((risk, purchase, fee))
    }
}
fn add(sum: &mut f64, c: &mut f64, x: f64) {
    let t = *sum + x;
    *c += if sum.abs() >= x.abs() {
        (*sum - t) + x
    } else {
        (x - t) + *sum
    };
    *sum = t;
}
fn hash(p: &HedgeLotProblem) -> Result<String, ComputeError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(p).map_err(|e| invalid(e.to_string()))?)
    ))
}
/// Objective constant + sum_i Qii zi + sum_(i<j) Qij zi zj. Lower triangle is0.
pub fn build(p: &HedgeLotProblem) -> Result<QuboObjective, ComputeError> {
    p.validate()?;
    let n = p.lots.len();
    let current = p.current_risk.values();
    let target = p.target_risk.values();
    let mut residual = [0.; 3];
    for k in 0..3 {
        residual[k] = finite(
            (current[k] - target[k]) / p.risk_scales[k],
            "normalized initial residual",
        )?;
    }
    let data = p
        .lots
        .iter()
        .map(|l| p.lot_data(l))
        .collect::<Result<Vec<_>, _>>()?;
    let mut q = vec![vec![0.; n]; n];
    let mut constant = 0.;
    for k in 0..3 {
        if p.risk_weights[k] > 0. {
            constant = finite(
                constant + p.risk_weights[k] * residual[k].powi(2),
                "constant",
            )?;
        }
    }
    for i in 0..n {
        let mut diagonal = finite(p.fee_weight * data[i].2, "weighted fee")?;
        for k in 0..3 {
            if p.risk_weights[k] > 0. {
                let g = finite(data[i].0[k] / p.risk_scales[k], "normalized lot risk")?;
                diagonal = finite(
                    diagonal + p.risk_weights[k] * (2. * residual[k] * g + g * g),
                    "diagonal coefficient",
                )?;
            }
        }
        q[i][i] = diagonal;
        for j in i + 1..n {
            let mut pair = 0.;
            for k in 0..3 {
                if p.risk_weights[k] > 0. {
                    pair = finite(
                        pair + 2.
                            * p.risk_weights[k]
                            * (data[i].0[k] / p.risk_scales[k])
                            * (data[j].0[k] / p.risk_scales[k]),
                        "pair coefficient",
                    )?;
                }
            }
            q[i][j] = pair;
        }
    }
    Ok(QuboObjective{constant,upper:q,lot_ids:p.lots.iter().map(|l|l.lot_id.clone()).collect(),problem_sha256:hash(p)?,feasibility_constraints_encoded:false,convention:"Upper triangular objective only; z_i²=z_i. Delta: underlying units; gamma: underlying units per spot-price unit; vega: currency per volatility percentage point. Hard cash/inventory/one-lot-per-instrument filters are enforced separately by solve.".into()})
}
impl QuboObjective {
    pub fn value(&self, mask: u64) -> Result<f64, ComputeError> {
        let n = self.upper.len();
        if n > 16
            || self.lot_ids.len() != n
            || self.upper.iter().any(|r| r.len() != n)
            || mask >= (1u64 << n)
            || !self.constant.is_finite()
            || self.upper.iter().flatten().any(|x| !x.is_finite())
            || self
                .upper
                .iter()
                .enumerate()
                .any(|(i, row)| row.iter().take(i).any(|x| *x != 0.))
        {
            return Err(invalid("invalid QUBO shape/mask/coefficients"));
        }
        let (mut value, mut correction) = (self.constant, 0.);
        for i in 0..n {
            if mask & (1 << i) != 0 {
                for j in i..n {
                    if mask & (1 << j) != 0 {
                        add(&mut value, &mut correction, self.upper[i][j]);
                    }
                }
            }
        }
        finite(value + correction, "QUBO objective")
    }
    fn norm(&self, mask: u64) -> f64 {
        let mut n = self.constant.abs();
        for i in 0..self.upper.len() {
            if mask & (1 << i) != 0 {
                for j in i..self.upper.len() {
                    if mask & (1 << j) != 0 {
                        n += self.upper[i][j].abs();
                    }
                }
            }
        }
        n
    }
}
/// Direct financial objective and feasibility, independent of QUBO expansion.
pub fn evaluate(p: &HedgeLotProblem, mask: u64) -> Result<LotEvaluation, ComputeError> {
    p.validate()?;
    evaluate_checked(p, mask)
}
fn evaluate_checked(p: &HedgeLotProblem, mask: u64) -> Result<LotEvaluation, ComputeError> {
    let n = p.lots.len();
    if mask >= (1u64 << n) {
        return Err(invalid("lot mask exceeds binary dimension"));
    }
    let mut risk = p.current_risk.values();
    let mut rc = [0.; 3];
    let (mut fees, mut fc) = (0., 0.);
    let (mut cash, mut cc) = (p.cash_before, 0.);
    let mut inventory = p
        .instruments
        .iter()
        .map(|i| i.current_contracts)
        .collect::<Vec<_>>();
    let mut used = vec![false; p.instruments.len()];
    let mut chosen = Vec::new();
    let mut reasons = Vec::new();
    for i in 0..n {
        if mask & (1 << i) == 0 {
            continue;
        }
        let l = &p.lots[i];
        if used[l.instrument] {
            reasons.push(format!(
                "multiple lots for instrument {}",
                p.instruments[l.instrument].instrument_id
            ));
        }
        used[l.instrument] = true;
        let (g, purchase, fee) = p.lot_data(l)?;
        for k in 0..3 {
            add(&mut risk[k], &mut rc[k], g[k]);
        }
        add(&mut fees, &mut fc, fee);
        add(
            &mut cash,
            &mut cc,
            finite(-purchase - fee, "trade cash change")?,
        );
        inventory[l.instrument] = inventory[l.instrument]
            .checked_add(l.contracts)
            .ok_or_else(|| invalid("inventory overflow"))?;
        chosen.push((i, purchase, fee));
    }
    let mut residual = [0.; 3];
    let target = p.target_risk.values();
    let mut objective = 0.;
    for k in 0..3 {
        risk[k] = finite(risk[k] + rc[k], "post-trade risk")?;
        residual[k] = finite(risk[k] - target[k], "risk residual")?;
        if p.risk_weights[k] > 0. {
            objective = finite(
                objective + p.risk_weights[k] * (residual[k] / p.risk_scales[k]).powi(2),
                "direct risk objective",
            )?;
        }
    }
    fees = finite(fees + fc, "total fees")?;
    cash = finite(cash + cc, "cash balance")?;
    objective = finite(objective + p.fee_weight * fees, "direct objective")?;
    if cash < -p.cash_borrowing_limit {
        reasons.push("cash borrowing allowance exceeded".into());
    }
    for (i, q) in inventory.iter().enumerate() {
        if *q < p.instruments[i].min_contracts || *q > p.instruments[i].max_contracts {
            reasons.push(format!(
                "inventory bound: {}",
                p.instruments[i].instrument_id
            ));
        }
    }
    let mut execution = Vec::new();
    if reasons.is_empty() {
        // Cash-increasing trades first; at most one selected lot per instrument.
        chosen.sort_by(|a, b| (-b.1 - b.2).total_cmp(&(-a.1 - a.2)).then(a.0.cmp(&b.0)));
        let (mut running, mut correction) = (p.cash_before, 0.);
        for (i, purchase, fee) in chosen {
            let lot = &p.lots[i];
            let q = &p.instruments[lot.instrument];
            add(&mut running, &mut correction, -purchase - fee);
            let after = finite(running + correction, "execution cash")?;
            if after < -p.cash_borrowing_limit {
                reasons.push("cash allowance fails during ordered execution".into());
                execution.clear();
                break;
            }
            execution.push(ExecutableLot {
                lot_id: lot.lot_id.clone(),
                instrument_id: q.instrument_id.clone(),
                source_quote_id: q.quote_id.clone(),
                contracts: lot.contracts,
                underlying_units: lot.contracts as f64 * q.multiplier as f64,
                signed_purchase_notional: purchase,
                fee,
                cash_after: after,
                inventory_after: inventory[lot.instrument],
            });
        }
    }
    Ok(LotEvaluation {
        mask,
        feasible: reasons.is_empty(),
        reasons,
        objective,
        post_trade_risk: RiskVector::from(risk),
        residual_risk: RiskVector::from(residual),
        fees,
        cash_after: cash,
        final_contracts: inventory,
        execution,
    })
}
/// Exhaustive finite-domain optimum, limited to 16 binary choices (65536 masks).
/// The authoritative objective is recomputed from financial lots. QUBO values
/// are cross-checked with a conservative floating-point expansion bound.
pub fn solve(p: &HedgeLotProblem) -> Result<LotSolution, ComputeError> {
    let q = build(p)?;
    let count = 1u64 << p.lots.len();
    let mut best: Option<LotEvaluation> = None;
    let mut feasible = 0;
    let mut max_error = 0_f64;
    for mask in 0..count {
        let value = evaluate_checked(p, mask)?;
        let from_q = q.value(mask)?;
        let error = (from_q - value.objective).abs();
        let bound = 1024. * f64::EPSILON * (1. + q.norm(mask) + value.objective.abs());
        if !bound.is_finite() || error > bound {
            return Err(invalid(
                "QUBO expansion exceeds floating-point bound; rescale risk units",
            ));
        }
        max_error = max_error.max(error);
        if value.feasible {
            feasible += 1;
            if best.as_ref().is_none_or(|b| value.objective < b.objective) {
                best = Some(value);
            }
        }
    }
    Ok(LotSolution{objective_model:q,selected:best.ok_or_else(||invalid("no feasible financial lot mask"))?,enumerated_masks:count,feasible_masks:feasible,max_expansion_discrepancy:max_error,method:"Exact enumeration of all bounded signed-lot choices, direct normalized Greek-residual/fee objective, hard financial feasibility, lowest mask on exact ties. Local quoted-Greek risk proxy; exhaustive CPU search.".into()})
}
