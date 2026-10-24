//! Model-independent path/hedge interface. The paper adapter supplies Heston or
//! BS states; later return-signal generators can supply the same contract without
//! modifying policy accounting. This module does not add a predictive signal.
use crate::{
    ComputeError, Precision, Session,
    budget::MemoryBudget,
    hedge_policy::{HedgePolicy, HedgePolicyConfig},
    market_paths::{FUNDED_VARIANCE_CLAIM, MarketModel, MarketPaths, STOCK, VARIANCE},
    risk_objective::RiskObjective,
};
use burn::tensor::{DType, Tensor, TensorCreationOptions, TensorData, activation::relu};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HedgeLiability {
    Call { strike: f64 },
    NormalizedCallSpread { lower: f64, upper: f64 },
    None,
}
impl HedgeLiability {
    pub fn payoff(&self, spot: f64) -> Result<f64, ComputeError> {
        let value = match *self {
            Self::Call { strike } if strike.is_finite() && strike > 0.0 => (spot - strike).max(0.0),
            Self::NormalizedCallSpread { lower, upper }
                if lower.is_finite() && upper.is_finite() && lower > 0.0 && upper > lower =>
            {
                if spot <= lower {
                    0.0
                } else if spot >= upper {
                    1.0
                } else {
                    (spot - lower) / (upper - lower)
                }
            }
            Self::None => 0.0,
            _ => return Err(invalid("invalid European liability strikes")),
        };
        if !spot.is_finite() || spot <= 0.0 || !value.is_finite() {
            return Err(invalid("invalid liability spot/payoff"));
        }
        Ok(value)
    }
}

#[derive(Debug, Clone)]
pub struct HedgePathBatch {
    pub paths: usize,
    pub steps: usize,
    pub hedge_count: usize,
    pub feature_count: usize,
    /// `[path, time, instrument]`, includes terminal marks.
    pub marks: Vec<f64>,
    /// `[path, decision, feature]`, excludes terminal time.
    pub features: Vec<f64>,
    pub payoffs: Vec<f64>,
    pub source_hash: String,
}
impl HedgePathBatch {
    /// Independent market systems must have the same path IDs and grid. Heston
    /// uses(logS,V) and(S,H); BS useslogS andS. Liabilities add across systems.
    pub fn from_markets(
        markets: &[MarketPaths],
        liability: HedgeLiability,
    ) -> Result<Self, ComputeError> {
        for market in markets {
            market.validate()?;
        }
        let first = markets
            .first()
            .ok_or_else(|| invalid("empty market system set"))?;
        if first.config.rate != 0.0 || first.config.dividend_yield != 0.0 {
            return Err(ComputeError::Unsupported(
                "paper hedge accounting currently requires r=q=0".into(),
            ));
        }
        let heston = matches!(first.config.model, MarketModel::Heston { .. });
        let per = if heston { 2 } else { 1 };
        let paths = first.config.paths;
        let steps = first.config.steps;
        let count = per * markets.len();
        let mark_len = paths
            .checked_mul(steps + 1)
            .and_then(|v| v.checked_mul(count))
            .ok_or_else(|| invalid("hedge mark shape overflow"))?;
        let feature_len = paths
            .checked_mul(steps)
            .and_then(|v| v.checked_mul(count))
            .ok_or_else(|| invalid("hedge feature shape overflow"))?;
        for market in markets {
            if market.config.paths != paths
                || market.config.steps != steps
                || market.config.first_path_id != first.config.first_path_id
                || market.config.dt.to_bits() != first.config.dt.to_bits()
                || market.config.rate != 0.0
                || market.config.dividend_yield != 0.0
                || matches!(market.config.model, MarketModel::Heston { .. }) != heston
                || market.values.len() != paths * (steps + 1) * 4
            {
                return Err(invalid("market grids, dimensions or model families differ"));
            }
        }
        let mut marks = Vec::with_capacity(mark_len);
        let mut features = Vec::with_capacity(feature_len);
        let mut payoffs = Vec::with_capacity(paths);
        for p in 0..paths {
            for t in 0..=steps {
                for market in markets {
                    let stock = market
                        .value(p, t, STOCK)
                        .ok_or_else(|| invalid("missing stock mark"))?;
                    marks.push(stock);
                    if heston {
                        marks.push(
                            market
                                .value(p, t, FUNDED_VARIANCE_CLAIM)
                                .ok_or_else(|| invalid("missing variance mark"))?,
                        );
                    }
                    if t < steps {
                        features.push(stock.ln());
                        if heston {
                            features.push(
                                market
                                    .value(p, t, VARIANCE)
                                    .ok_or_else(|| invalid("missing variance state"))?,
                            );
                        }
                    }
                }
            }
            let mut value = 0.0;
            for market in markets {
                value += liability.payoff(
                    market
                        .value(p, steps, STOCK)
                        .ok_or_else(|| invalid("missing terminal stock"))?,
                )?;
            }
            payoffs.push(value);
        }
        let mut hash = Sha256::new();
        for market in markets {
            hash.update(market.content_hash.as_bytes());
        }
        hash.update(serde_json::to_vec(&liability).map_err(execution)?);
        let batch = Self {
            paths,
            steps,
            hedge_count: count,
            feature_count: count,
            marks,
            features,
            payoffs,
            source_hash: format!("{:x}", hash.finalize()),
        };
        batch.validate()?;
        Ok(batch)
    }
    pub fn validate(&self) -> Result<(), ComputeError> {
        let expected = self
            .steps
            .checked_add(1)
            .and_then(|times| self.paths.checked_mul(times))
            .and_then(|n| n.checked_mul(self.hedge_count));
        let features = self
            .paths
            .checked_mul(self.steps)
            .and_then(|n| n.checked_mul(self.feature_count));
        if self.paths == 0
            || self.steps == 0
            || self.hedge_count == 0
            || self.feature_count == 0
            || expected != Some(self.marks.len())
            || features != Some(self.features.len())
            || self.payoffs.len() != self.paths
            || self.source_hash.is_empty()
            || self.marks.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self
                .features
                .iter()
                .chain(&self.payoffs)
                .any(|v| !v.is_finite())
        {
            return Err(invalid("invalid hedge batch shapes/values/source"));
        }
        Ok(())
    }
}

pub struct PreparedHedgePaths {
    paths: usize,
    steps: usize,
    hedge_count: usize,
    feature_count: usize,
    source_hash: String,
    marks: Vec<Tensor<2>>,
    features: Vec<Tensor<2>>,
    payoffs: Tensor<1>,
}
impl PreparedHedgePaths {
    pub fn shape(&self) -> [usize; 4] {
        [self.paths, self.steps, self.hedge_count, self.feature_count]
    }
    pub fn source_hash(&self) -> &str {
        &self.source_hash
    }
    pub fn upload(session: &Session, batch: &HedgePathBatch) -> Result<Self, ComputeError> {
        batch.validate()?;
        if session.precision() == Precision::F32
            && batch
                .marks
                .iter()
                .chain(&batch.features)
                .chain(&batch.payoffs)
                .any(|v| !(*v as f32).is_finite())
        {
            return Err(invalid("hedge batch values cannot be represented in f32"));
        }
        if batch.paths > session.max_batch_rows() {
            return Err(invalid("hedge path batch exceeds session cap"));
        }
        let mut marks = Vec::with_capacity(batch.steps + 1);
        let mut features = Vec::with_capacity(batch.steps);
        for t in 0..=batch.steps {
            let data = (0..batch.paths)
                .flat_map(|p| {
                    let s = (p * (batch.steps + 1) + t) * batch.hedge_count;
                    batch.marks[s..s + batch.hedge_count].iter().copied()
                })
                .collect();
            marks.push(tensor2(session, data, [batch.paths, batch.hedge_count]));
            if t < batch.steps {
                let data = (0..batch.paths)
                    .flat_map(|p| {
                        let s = (p * batch.steps + t) * batch.feature_count;
                        batch.features[s..s + batch.feature_count].iter().copied()
                    })
                    .collect();
                features.push(tensor2(session, data, [batch.paths, batch.feature_count]));
            }
        }
        let payoffs = session.vector(batch.payoffs.clone());
        session.synchronize()?;
        Ok(Self {
            paths: batch.paths,
            steps: batch.steps,
            hedge_count: batch.hedge_count,
            feature_count: batch.feature_count,
            source_hash: batch.source_hash.clone(),
            marks,
            features,
            payoffs,
        })
    }
}

pub struct HedgeRollout {
    pub pnl: Tensor<1>,
    pub costs: Tensor<1>,
    pub turnover: Tensor<1>,
    /// Only populated on explicit request; retaining all holdings costs memory.
    pub holdings: Vec<Tensor<2>>,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HedgeCostConfig {
    pub proportional: f64,
    pub terminal_proportional: f64,
}
impl Default for HedgeCostConfig {
    fn default() -> Self {
        Self {
            proportional: 0.0,
            terminal_proportional: 0.0,
        }
    }
}

pub fn policy_rollout(
    session: &Session,
    paths: &PreparedHedgePaths,
    policy: &mut HedgePolicy,
    cost: HedgeCostConfig,
    premium: f64,
    train: bool,
    retain_holdings: bool,
) -> Result<HedgeRollout, ComputeError> {
    if policy.config().time_steps != paths.steps
        || policy.config().hedge_count != paths.hedge_count
        || policy.config().feature_count != paths.feature_count
        || !premium.is_finite()
        || !cost.proportional.is_finite()
        || cost.proportional < 0.0
        || !cost.terminal_proportional.is_finite()
        || cost.terminal_proportional < 0.0
    {
        return Err(invalid(
            "policy/path shape, premium or transaction-cost mismatch",
        ));
    }
    let mut previous = tensor2(
        session,
        vec![0.0; paths.paths * paths.hedge_count],
        [paths.paths, paths.hedge_count],
    );
    let mut pnl = session.vector(vec![premium; paths.paths]);
    let mut costs = session.vector(vec![0.0; paths.paths]);
    let mut turnover = costs.clone();
    let mut holdings = Vec::new();
    for t in 0..paths.steps {
        let target = policy.forward_at(t, paths.features[t].clone(), previous.clone(), train)?;
        let notional = (target.clone() - previous).abs() * paths.marks[t].clone();
        let traded = notional.sum_dim(1).reshape([paths.paths]);
        let charge = traded.clone() * cost.proportional;
        pnl = pnl
            + (target.clone() * (paths.marks[t + 1].clone() - paths.marks[t].clone()))
                .sum_dim(1)
                .reshape([paths.paths])
            - charge.clone();
        costs = costs + charge.without_autodiff();
        turnover = turnover + traded.without_autodiff();
        previous = target.clone();
        if retain_holdings {
            holdings.push(target.without_autodiff());
        }
    }
    let terminal_notional = (previous.abs() * paths.marks[paths.steps].clone())
        .sum_dim(1)
        .reshape([paths.paths]);
    let terminal = terminal_notional.clone() * cost.terminal_proportional;
    turnover = turnover + terminal_notional.without_autodiff();
    pnl = pnl - terminal.clone() - paths.payoffs.clone();
    costs = costs + terminal.without_autodiff();
    Ok(HedgeRollout {
        pnl,
        costs,
        turnover,
        holdings,
    })
}

/// Refresh fixed-weight running statistics on separately declared training
/// scenarios. Never call this with validation/test paths.
pub fn calibrate_policy_paths(
    session: &Session,
    paths: &PreparedHedgePaths,
    policy: &mut HedgePolicy,
) -> Result<(), ComputeError> {
    if policy.config().time_steps != paths.steps
        || policy.config().hedge_count != paths.hedge_count
        || policy.config().feature_count != paths.feature_count
    {
        return Err(invalid("calibration policy/path dimensions differ"));
    }
    let mut previous = tensor2(
        session,
        vec![0.0; paths.paths * paths.hedge_count],
        [paths.paths, paths.hedge_count],
    );
    for step in 0..paths.steps {
        previous = policy.calibrate_at(step, paths.features[step].clone(), previous)?;
    }
    Ok(())
}

/// CVaR uses a jointly trained scalar eta; empirical scalar evaluation performs
/// exact fractional-tail integration independently. All objectives minimize risk.
pub fn tensor_risk(
    pnl: Tensor<1>,
    objective: RiskObjective,
    eta: Option<Tensor<1>>,
) -> Result<Tensor<1>, ComputeError> {
    let n = pnl.dims()[0];
    if n == 0 {
        return Err(invalid("empty tensor risk batch"));
    }
    let mean = |x: Tensor<1>| (x * (1.0 / n as f64)).sum();
    match objective {
        RiskObjective::SquaredLoss => Ok(mean(pnl.clone() * pnl)),
        RiskObjective::Cvar { alpha } if alpha.is_finite() && (0.0..1.0).contains(&alpha) => {
            let eta = eta.ok_or_else(|| invalid("CVaR training requires eta"))?;
            if eta.dims() != [1]
                || eta.clone().without_autodiff().device()
                    != pnl.clone().without_autodiff().device()
                || eta.dtype() != pnl.dtype()
            {
                return Err(invalid("CVaR eta must be scalar"));
            }
            Ok(eta.clone() + mean(relu(-pnl - eta)) * (1.0 / (1.0 - alpha)))
        }
        RiskObjective::Entropic {
            risk_aversion: lambda,
        } if lambda.is_finite() && lambda > 0.0 => {
            let loss = -pnl;
            let anchor = loss.clone().without_autodiff().max();
            let scaled = (loss.clone() - anchor.clone()) * lambda;
            let min = scaled
                .clone()
                .without_autodiff()
                .min()
                .try_to_vec_as::<f64>()
                .map_err(execution)?[0];
            if min >= -0.125 {
                // expm1 via a converged series avoids exp(x)-1 cancellation;
                // degree12 remainder is below4e-22 on this bounded interval.
                let center = (anchor.clone() + mean(loss.clone() - anchor)).without_autodiff();
                let x = (loss - center.clone()) * lambda;
                let mut term = x.clone();
                let mut total = term.clone();
                for k in 2..=12 {
                    term = term * x.clone() * (1.0 / k as f64);
                    total = total + term.clone();
                }
                // Some device backends implement log1p as log(1+y), losing
                // the small variance correction before division by lambda.
                // Here |x|<=0.125, hence |mean(expm1(x))|<0.134; evaluate
                // log1p directly without forming 1+y. The actual centered
                // mean is O(lambda^2), so the mean-risk limit stays accurate.
                let y = mean(total);
                let mut power = y.clone();
                let mut logarithm = y.clone();
                for k in 2..=16 {
                    power = power * y.clone();
                    let coefficient = if k % 2 == 0 { -1.0 } else { 1.0 } / k as f64;
                    logarithm = logarithm + power.clone() * coefficient;
                }
                Ok(center + logarithm * (1.0 / lambda))
            } else {
                Ok(anchor + mean(scaled.exp()).log() * (1.0 / lambda))
            }
        }
        _ => Err(invalid("invalid tensor risk objective")),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgeMemoryPlan {
    pub schema: String,
    pub batch_paths: usize,
    pub training: bool,
    pub estimated_compute_bytes: u64,
    pub estimated_host_working_bytes: u64,
    pub compute_admission_limit_bytes: u64,
    pub host_admission_limit_bytes: u64,
    pub parameter_count: u64,
    pub modeled_scalar_slots: u64,
}
/// Conservative complete-policy admission, including strict broadcast products,
/// recurrent activations/adjoints, Adam candidates and market buffers. No quotas.
pub fn plan_hedging(
    config: &HedgePolicyConfig,
    paths: usize,
    precision: Precision,
    target: crate::ExecutionTarget,
    budget: &MemoryBudget,
    training: bool,
) -> Result<HedgeMemoryPlan, ComputeError> {
    config.validate()?;
    if paths == 0
        || paths > config.max_batch_rows
        || paths > budget.max_batch_rows
        || budget.headroom_percent >= 100
        || budget.compute_bytes == 0
        || budget.host_working_bytes == 0
    {
        return Err(invalid("invalid hedge job batch/budget/headroom"));
    }
    if target == crate::ExecutionTarget::Metal && precision == Precision::F64 {
        return Err(ComputeError::Unsupported("Metal has no native f64".into()));
    }
    let p = paths as u64;
    let t = config.time_steps as u64;
    let h = config.hidden_width as u64;
    let d = config.hedge_count as u64;
    let f = config.feature_count as u64;
    let input = config.input_width() as u64;
    let parameters = t
        .checked_mul(input * h + h * h + h * d + 6 * h + d)
        .ok_or_else(overflow)?;
    let expanded = input * h + h * h + h * d;
    if p.checked_mul((input * h).max(h * h).max(h * d))
        .is_none_or(|n| n > crate::hedge_policy::MAX_EXPANDED_PRODUCT_ELEMENTS as u64)
    {
        return Err(ComputeError::BudgetExceeded(
            "hedge strict product exceeds shape cap".into(),
        ));
    }
    let tape = if training {
        t.checked_mul(12 * (expanded + 32 * h + 8 * d + 8 * f))
            .ok_or_else(overflow)?
    } else {
        12 * expanded + t * (4 * d + 2 * f + 16)
    };
    let slots = p
        .checked_mul(tape)
        .and_then(|v| v.checked_add(parameters * 12 + t * 8 * h))
        .ok_or_else(overflow)?;
    let scalar = if precision == Precision::F64 { 8 } else { 4 };
    let compute = slots
        .checked_mul(scalar)
        .and_then(|v| v.checked_add(budget.compute_reserve_bytes))
        .ok_or_else(overflow)?;
    let source = p
        .checked_mul((t + 1) * (4 * 8 + 4 * d * 8 + 2 * f * 8) + 1024)
        .ok_or_else(overflow)?;
    let shared = matches!(
        target,
        crate::ExecutionTarget::Cpu | crate::ExecutionTarget::Metal
    );
    let host = source
        .checked_add(parameters * 96)
        .and_then(|v| v.checked_add(if shared { compute } else { 0 }))
        .ok_or_else(overflow)?;
    let usable = |bytes: u64| {
        let pct = 100 - u64::from(budget.headroom_percent);
        (bytes / 100) * pct + (bytes % 100) * pct / 100
    };
    let cl = usable(budget.compute_bytes);
    let hl = usable(budget.host_working_bytes);
    if compute > cl || host > hl {
        return Err(ComputeError::BudgetExceeded(format!(
            "hedge job needs estimatedcompute {compute}/{cl}, host {host}/{hl}; reduce batch or increase working allowance"
        )));
    }
    Ok(HedgeMemoryPlan {
        schema: "volt.compute.hedge_memory_plan.v1".into(),
        batch_paths: paths,
        training,
        estimated_compute_bytes: compute,
        estimated_host_working_bytes: host,
        compute_admission_limit_bytes: cl,
        host_admission_limit_bytes: hl,
        parameter_count: parameters,
        modeled_scalar_slots: slots,
    })
}

pub(crate) fn tensor2(session: &Session, values: Vec<f64>, shape: [usize; 2]) -> Tensor<2> {
    let dtype = if session.precision() == Precision::F64 {
        DType::F64
    } else {
        DType::F32
    };
    Tensor::from_data(
        TensorData::new(values, shape),
        TensorCreationOptions::new(session.device().clone()).with_dtype(dtype),
    )
}
fn invalid(s: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn execution(e: impl std::fmt::Display) -> ComputeError {
    ComputeError::Execution(e.to_string())
}
fn overflow() -> ComputeError {
    invalid("hedge memory estimate overflow")
}
