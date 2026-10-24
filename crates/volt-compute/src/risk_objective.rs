//! Independent scalar held-out objectives. P&L is positive for gains and includes
//! any declared premium already: the evaluator never adds cash implicitly.
//! CVaR and entropy measure losses, so smaller objective values are preferable.
//! Squared loss is `E[P&L²]` and is not a cash-invariant monetary risk measure.
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum RiskError {
    #[error("invalid risk input: {0}")]
    InvalidInput(String),
    #[error("risk arithmetic is not representable: {0}")]
    Arithmetic(&'static str),
}
type Result<T> = std::result::Result<T, RiskError>;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RiskObjective {
    Cvar { alpha: f64 },
    Entropic { risk_aversion: f64 },
    SquaredLoss,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskEvaluation {
    pub objective: RiskObjective,
    pub value: f64,
    pub mean_pnl: f64,
    /// Population dispersion under the normalized empirical probability weights.
    pub pnl_standard_deviation: f64,
    pub observations: usize,
    pub positive_weight_observations: usize,
    pub probability_weight_ess: f64,
    /// ESS of weights proportional to p_i exp(-lambda*pnl_i), not a confidence interval.
    pub entropic_tilt_ess: Option<f64>,
    pub cvar_tail_probability: Option<f64>,
    /// Loss at the fractional boundary of the worst-loss tail.
    pub cvar_boundary_loss: Option<f64>,
}

/// Positive finite weights need not sum to one. Zero weights are allowed, but
/// all observations must still be finite and at least one weight must be positive.
/// CVaR integrates exactly the worst (1-alpha) empirical probability mass,
/// including a fraction of the boundary observation rather than a rounded count.
pub fn weighted_risk(
    pnl: &[f64],
    weights: Option<&[f64]>,
    objective: RiskObjective,
) -> Result<RiskEvaluation> {
    validate_objective(objective)?;
    let probabilities = probabilities(pnl, weights)?;
    let mean = finite_sum(pnl.iter().zip(&probabilities).map(|(x, p)| x * p))?;
    let scale = pnl
        .iter()
        .zip(&probabilities)
        .filter(|(_, p)| **p > 0.0)
        .map(|(x, _)| x.abs())
        .fold(0.0, f64::max);
    let deviation = if scale == 0.0 {
        0.0
    } else {
        let variance_scaled = finite_sum(pnl.iter().zip(&probabilities).map(|(x, p)| {
            if *p == 0.0 {
                0.0
            } else {
                p * (x / scale - mean / scale).powi(2)
            }
        }))?;
        finite(
            scale * variance_scaled.max(0.0).sqrt(),
            "P&L standard deviation",
        )?
    };
    let weight_square = finite_sum(probabilities.iter().map(|p| p * p))?;
    let mut result = RiskEvaluation {
        objective,
        value: 0.0,
        mean_pnl: mean,
        pnl_standard_deviation: deviation,
        observations: pnl.len(),
        positive_weight_observations: probabilities.iter().filter(|p| **p > 0.0).count(),
        probability_weight_ess: 1.0 / weight_square,
        entropic_tilt_ess: None,
        cvar_tail_probability: None,
        cvar_boundary_loss: None,
    };
    match objective {
        RiskObjective::Cvar { alpha } => {
            let tail = 1.0 - alpha;
            let mut loss: Vec<(f64, f64)> = pnl
                .iter()
                .zip(&probabilities)
                .filter(|(_, p)| **p > 0.0)
                .map(|(x, p)| (-x, *p))
                .collect();
            loss.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut remaining = tail;
            let mut parts = Vec::with_capacity(loss.len());
            for (i, (x, p)) in loss.iter().enumerate() {
                if remaining <= 0.0 {
                    break;
                }
                let take = if i + 1 == loss.len() {
                    remaining
                } else {
                    p.min(remaining)
                };
                parts.push(x * (take / tail));
                remaining = (remaining - take).max(0.0);
                result.cvar_boundary_loss = Some(*x);
            }
            result.value = finite_sum(parts.into_iter())?;
            result.cvar_tail_probability = Some(tail);
        }
        RiskObjective::Entropic {
            risk_aversion: lambda,
        } => {
            let max_loss = pnl
                .iter()
                .zip(&probabilities)
                .filter(|(_, p)| **p > 0.0)
                .map(|(x, _)| -x)
                .fold(f64::NEG_INFINITY, f64::max);
            let exponent = |x: f64| lambda * ((-x) - max_loss);
            let terms: Vec<f64> = pnl
                .iter()
                .zip(&probabilities)
                .map(|(x, p)| {
                    if *p == 0.0 {
                        0.0
                    } else {
                        p * exponent(*x).exp()
                    }
                })
                .collect();
            let total = finite_sum(terms.iter().copied())?;
            if total <= 0.0 {
                return Err(RiskError::Arithmetic("entropic exponential mass"));
            }
            // expm1/log1p avoid losing the small-lambda mean-risk limit. For
            // concentrated tails, direct positive summation avoids cancellation.
            let minus_one = finite_sum(pnl.iter().zip(&probabilities).map(|(x, p)| {
                if *p == 0.0 {
                    0.0
                } else {
                    p * exponent(*x).exp_m1()
                }
            }))?;
            let log_mass = if minus_one > -0.5 {
                minus_one.ln_1p()
            } else {
                total.ln()
            };
            // At tiny lambda, even expm1's subnormal mass can lose its last bit
            // before division by lambda. The second cumulant expansion has
            // omitted terms of order (lambda*scale)^2*scale, below f64 relative
            // precision in this narrowly selected regime.
            result.value = if lambda * scale <= 1e-8 {
                finite(
                    -mean + 0.5 * (lambda * deviation) * deviation,
                    "entropic mean limit",
                )?
            } else {
                finite(max_loss + log_mass / lambda, "entropic risk")?
            };
            result.entropic_tilt_ess =
                Some(1.0 / finite_sum(terms.iter().map(|x| (x / total).powi(2)))?);
        }
        RiskObjective::SquaredLoss => {
            result.value = finite_sum(pnl.iter().zip(&probabilities).map(|(x, p)| (p * x) * x))?;
        }
    }
    Ok(result)
}

/// Monetary indifference premium = optimized risk(with liability, zero added
/// premium) - optimized risk(no liability, zero added premium). Both optimizations
/// must use the same market, objective, frictions and feasible policy class.
/// Reject squared loss: subtracting its optimal values is not a monetary price.
pub fn indifference_price(
    objective: RiskObjective,
    optimized_with_liability: f64,
    optimized_without_liability: f64,
) -> Result<f64> {
    validate_objective(objective)?;
    if matches!(objective, RiskObjective::SquaredLoss) {
        return invalid("squared loss does not define a cash-invariant indifference price");
    }
    if !optimized_with_liability.is_finite() || !optimized_without_liability.is_finite() {
        return invalid("optimized risks must be finite");
    }
    finite(
        optimized_with_liability - optimized_without_liability,
        "indifference premium",
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapConfig {
    pub replicates: usize,
    pub confidence: f64,
    pub seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PercentileInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedPathComparison {
    pub schema_version: String,
    pub objective: RiskObjective,
    pub paths: usize,
    pub bootstrap: BootstrapConfig,
    pub candidate_risk: f64,
    pub baseline_risk: f64,
    /// Negative favors candidate; resamples recompute both nonlinear objectives.
    pub risk_difference_candidate_minus_baseline: f64,
    pub risk_difference_interval: PercentileInterval,
    /// Positive favors candidate. Equal initial premiums are caller's obligation.
    pub mean_pnl_difference_candidate_minus_baseline: f64,
    pub mean_pnl_difference_standard_error: f64,
    pub mean_pnl_difference_interval: PercentileInterval,
    /// Unbiased sample covariance across paired path P&Ls.
    pub paired_sample_covariance: f64,
}

/// IID empirical PATH-PAIR bootstrap: draw n uniform indices with replacement,
/// and use each index for candidate and baseline together. This estimates held-out
/// path sampling uncertainty conditional on the two frozen trained policies.
/// It does not estimate training-seed or simulator/model uncertainty. Weighted
/// scenario probabilities are deliberately not accepted as IID observations.
pub fn paired_path_comparison(
    candidate: &[f64],
    baseline: &[f64],
    objective: RiskObjective,
    bootstrap: BootstrapConfig,
) -> Result<PairedPathComparison> {
    if candidate.len() != baseline.len() || candidate.len() < 2 {
        return invalid("need at least two aligned candidate and baseline paths");
    }
    if bootstrap.replicates < 2
        || !bootstrap.confidence.is_finite()
        || bootstrap.confidence <= 0.0
        || bootstrap.confidence >= 1.0
    {
        return invalid(
            "bootstrap requires at least two replicates and confidence strictly between zero and one",
        );
    }
    let a = weighted_risk(candidate, None, objective)?;
    let b = weighted_risk(baseline, None, objective)?;
    let n = candidate.len() as f64;
    let differences: Vec<f64> = candidate.iter().zip(baseline).map(|(a, b)| a - b).collect();
    if differences.iter().any(|x| !x.is_finite()) {
        return Err(RiskError::Arithmetic("paired differences"));
    }
    let mean_difference = finite_sum(differences.iter().map(|x| x / n))?;
    let sum_square = finite_sum(
        differences
            .iter()
            .map(|x| ((x - mean_difference) / (n - 1.0)) * (x - mean_difference)),
    )?;
    let covariance = finite_sum(
        candidate
            .iter()
            .zip(baseline)
            .map(|(x, y)| ((x - a.mean_pnl) / (n - 1.0)) * (y - b.mean_pnl)),
    )?;
    let mut rng = SplitMix64(bootstrap.seed);
    let mut sample_a = vec![0.0; candidate.len()];
    let mut sample_b = vec![0.0; candidate.len()];
    let mut risks = Vec::with_capacity(bootstrap.replicates);
    let mut means = Vec::with_capacity(bootstrap.replicates);
    for _ in 0..bootstrap.replicates {
        for i in 0..candidate.len() {
            let index = rng.index(candidate.len());
            sample_a[i] = candidate[index];
            sample_b[i] = baseline[index];
        }
        let aa = weighted_risk(&sample_a, None, objective)?;
        let bb = weighted_risk(&sample_b, None, objective)?;
        risks.push(finite(aa.value - bb.value, "bootstrap risk difference")?);
        means.push(finite_sum(
            sample_a.iter().zip(&sample_b).map(|(x, y)| (x - y) / n),
        )?);
    }
    let risk_interval = percentile_interval(&mut risks, bootstrap.confidence);
    let mean_interval = percentile_interval(&mut means, bootstrap.confidence);
    Ok(PairedPathComparison {
        schema_version: "volt.compute.paired_path_comparison.v1".into(),
        objective,
        paths: candidate.len(),
        bootstrap,
        candidate_risk: a.value,
        baseline_risk: b.value,
        risk_difference_candidate_minus_baseline: finite(a.value - b.value, "risk difference")?,
        risk_difference_interval: risk_interval,
        mean_pnl_difference_candidate_minus_baseline: mean_difference,
        mean_pnl_difference_standard_error: (sum_square.max(0.0) / n).sqrt(),
        mean_pnl_difference_interval: mean_interval,
        paired_sample_covariance: covariance,
    })
}

fn probabilities(pnl: &[f64], weights: Option<&[f64]>) -> Result<Vec<f64>> {
    if pnl.is_empty() || pnl.iter().any(|x| !x.is_finite()) {
        return invalid("P&L must be nonempty and finite");
    }
    if let Some(w) = weights {
        if w.len() != pnl.len() || w.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return invalid("weights must match paths and be finite/nonnegative");
        }
        let maximum = w.iter().copied().fold(0.0, f64::max);
        if maximum == 0.0 {
            return invalid("at least one weight must be positive");
        }
        let total = finite_sum(w.iter().map(|x| x / maximum))?;
        let normalized: Vec<f64> = w.iter().map(|x| (x / maximum) / total).collect();
        if w.iter()
            .zip(&normalized)
            .any(|(original, p)| *original > 0.0 && *p == 0.0)
        {
            return Err(RiskError::Arithmetic(
                "positive probability underflow; rescale/reformulate scenario weights",
            ));
        }
        Ok(normalized)
    } else {
        Ok(vec![1.0 / pnl.len() as f64; pnl.len()])
    }
}
fn validate_objective(objective: RiskObjective) -> Result<()> {
    match objective {
        RiskObjective::Cvar { alpha } if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) => {
            invalid("CVaR alpha must be in [0,1)")
        }
        RiskObjective::Entropic { risk_aversion }
            if !risk_aversion.is_finite() || risk_aversion <= 0.0 =>
        {
            invalid("entropic risk aversion must be finite and positive")
        }
        _ => Ok(()),
    }
}
fn finite(value: f64, what: &'static str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RiskError::Arithmetic(what))
    }
}
fn invalid<T>(message: &str) -> Result<T> {
    Err(RiskError::InvalidInput(message.into()))
}
fn finite_sum(values: impl Iterator<Item = f64>) -> Result<f64> {
    let (mut sum, mut correction) = (0.0_f64, 0.0_f64);
    for x in values {
        finite(x, "summand")?;
        let next = finite(sum + x, "sum")?;
        correction += if sum.abs() >= x.abs() {
            (sum - next) + x
        } else {
            (x - next) + sum
        };
        sum = next;
    }
    finite(sum + correction, "compensated sum")
}
fn percentile_interval(values: &mut [f64], confidence: f64) -> PercentileInterval {
    values.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        let pos = p * (values.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        values[lo] * (1.0 - (pos - lo as f64)) + values[hi] * (pos - lo as f64)
    };
    let tail = (1.0 - confidence) / 2.0;
    PercentileInterval {
        lower: quantile(tail),
        upper: quantile(1.0 - tail),
    }
}
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn index(&mut self, len: usize) -> usize {
        let n = len as u64;
        let threshold = n.wrapping_neg() % n;
        loop {
            let x = self.next();
            if x >= threshold {
                return (x % n) as usize;
            }
        }
    }
}
