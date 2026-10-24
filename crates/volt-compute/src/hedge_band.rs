//! Deterministic no-trade bands around a causal model-hedge target.
//!
//! Widths are physical inventory units, separately declared for every asset.
//! On each decision the previous holding is projected to the closest point of
//! `[target-width,target+width]`. Liquidation and its fees belong to the ledger.
//! Selection is exclusively on caller-supplied validation paths; the selected
//! widths, objective, cost and validation source identity form an audit record.
use crate::{
    ComputeError,
    hedge_application::{HedgeCostConfig, HedgePathBatch},
    model_hedge::scalar_hedge_pnl,
    risk_objective::{RiskObjective, weighted_risk},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoTradeBandSelection {
    pub schema: String,
    pub widths: Vec<f64>,
    pub objective: RiskObjective,
    pub proportional_cost: f64,
    pub terminal_proportional_cost: f64,
    pub premium: f64,
    pub validation_source_hash: String,
    pub validation_paths: usize,
    pub candidate_risks: Vec<f64>,
    pub selected_candidate: usize,
}

/// Transform path-major `[paths,steps,assets]` targets, starting at zero inventory.
/// No future target or mark is read while processing a current decision.
pub fn banded_holdings(
    targets: &[f64],
    paths: usize,
    steps: usize,
    widths: &[f64],
    max_working_bytes: u64,
) -> Result<Vec<f64>, ComputeError> {
    let assets = widths.len();
    let count = paths
        .checked_mul(steps)
        .and_then(|n| n.checked_mul(assets))
        .ok_or_else(|| invalid("no-trade-band shape overflow"))?;
    let bytes = count
        .checked_add(assets)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| invalid("no-trade-band storage overflow"))?;
    if paths == 0 || steps == 0 || assets == 0 || targets.len() != count {
        return Err(invalid("no-trade-band shape mismatch"));
    }
    if bytes as u128 > max_working_bytes as u128 {
        return Err(ComputeError::BudgetExceeded(
            "no-trade-band output and inventory exceed working allowance".into(),
        ));
    }
    if widths.iter().any(|w| !w.is_finite() || *w < 0.0) || targets.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "no-trade-band targets/widths must be finite; widths nonnegative",
        ));
    }
    // Validate every boundary before allocating or producing a partial output.
    for row in targets.chunks_exact(assets) {
        for (target, width) in row.iter().zip(widths) {
            if !(target - width).is_finite() || !(target + width).is_finite() {
                return Err(invalid("no-trade-band inventory boundary overflow"));
            }
        }
    }
    let mut result = vec![0.0; count];
    let mut previous = vec![0.0_f64; assets];
    for path in 0..paths {
        previous.fill(0.0);
        for step in 0..steps {
            let offset = (path * steps + step) * assets;
            for asset in 0..assets {
                let target = targets[offset + asset];
                let next = previous[asset].clamp(target - widths[asset], target + widths[asset]);
                result[offset + asset] = next;
                previous[asset] = next;
            }
        }
    }
    Ok(result)
}

/// Tune a predeclared grid on validation only. Exact ties choose the first row.
/// The allowance covers new outputs/risk workspace, not caller-owned inputs.
pub fn select_no_trade_band(
    validation: &HedgePathBatch,
    targets: &[f64],
    cost: HedgeCostConfig,
    premium: f64,
    objective: RiskObjective,
    candidates: &[Vec<f64>],
    max_working_bytes: u64,
) -> Result<NoTradeBandSelection, ComputeError> {
    validation.validate()?;
    if validation.source_hash.trim().is_empty()
        || candidates.is_empty()
        || candidates.len() > 128
        || candidates
            .iter()
            .any(|row| row.len() != validation.hedge_count)
    {
        return Err(invalid("invalid validation identity or no-trade-band grid"));
    }
    // Scalar accounting vectors and nonlinear empirical-risk work, concurrently
    // with the transformed holdings. Reserve before any candidate execution.
    let risk_bytes = (validation.paths as u64)
        .checked_mul(192)
        .and_then(|n| n.checked_add(candidates.len() as u64 * 16))
        .ok_or_else(|| invalid("no-trade-band tuning storage overflow"))?;
    let holdings_allowance = max_working_bytes.checked_sub(risk_bytes).ok_or_else(|| {
        ComputeError::BudgetExceeded("no-trade-band risk workspace exceeds allowance".into())
    })?;
    let mut risks = Vec::with_capacity(candidates.len());
    let mut selected = 0;
    for (index, widths) in candidates.iter().enumerate() {
        let holdings = banded_holdings(
            targets,
            validation.paths,
            validation.steps,
            widths,
            holdings_allowance,
        )?;
        let pnl = scalar_hedge_pnl(validation, &holdings, cost, premium)?;
        let risk = weighted_risk(&pnl.pnl, None, objective)
            .map_err(|e| invalid(&e.to_string()))?
            .value;
        if index == 0 || risk < risks[selected] {
            selected = index;
        }
        risks.push(risk);
    }
    Ok(NoTradeBandSelection {
        schema: "volt.compute.no_trade_band_selection.v1".into(),
        widths: candidates[selected].clone(),
        objective,
        proportional_cost: cost.proportional,
        terminal_proportional_cost: cost.terminal_proportional,
        premium,
        validation_source_hash: validation.source_hash.clone(),
        validation_paths: validation.paths,
        candidate_risks: risks,
        selected_candidate: selected,
    })
}

fn invalid(message: &str) -> ComputeError {
    ComputeError::InvalidInput(message.into())
}
