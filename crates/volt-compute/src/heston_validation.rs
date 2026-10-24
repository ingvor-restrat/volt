//! Independent node and cutoff refinement of all Heston output metrics.
//! Passing finite refinements is a numerical diagnostic, not a global accuracy
//! certificate or a substitute for independent prices/Greeks and domain tests.
use crate::{
    ComputeError, Precision,
    heston::QuadratureConfig,
    heston_job::{
        HestonJobConfig, HestonJobRequest, HestonJobResult, metrics, plan_heston, run_heston,
    },
};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct RefinementCheck {
    pub direction: &'static str,
    pub metric: &'static str,
    pub max_absolute_difference: f64,
    pub max_tolerance_ratio: f64,
    pub mismatched_rows: usize,
}
#[derive(Debug, Serialize)]
pub struct CheckedHestonResult {
    /// (N,U), (2N,U), then (2N,2U): independent node and cutoff directions.
    /// The final stage supplies the returned risk estimates.
    pub stages: Vec<HestonJobResult>,
    pub checks: Vec<RefinementCheck>,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub passed: bool,
}

pub fn quadrature_stages(q: QuadratureConfig) -> Result<[QuadratureConfig; 3], ComputeError> {
    if q.nodes > 1024
        || q.nodes < 16
        || !q.upper_bound.is_finite()
        || q.upper_bound > 5_000_000.0
        || q.upper_bound < 1.0
    {
        return Err(ComputeError::InvalidInput("checked Heston base quadrature requires16..=1024nodes andcutoff1..=5000000 to admit independent refinement".into()));
    }
    Ok([
        q,
        QuadratureConfig {
            nodes: q.nodes * 2,
            ..q
        },
        QuadratureConfig {
            nodes: q.nodes * 2,
            upper_bound: q.upper_bound * 2.0,
        },
    ])
}
pub fn run_checked_heston(
    request: &HestonJobRequest,
    config: &HestonJobConfig,
) -> Result<CheckedHestonResult, ComputeError> {
    let quadratures = quadrature_stages(config.quadrature)?;
    // Verify all three admissions and all input rows before the first dispatch.
    for quadrature in quadratures {
        plan_heston(
            request,
            &HestonJobConfig {
                quadrature,
                ..config.clone()
            },
        )?;
    }
    let mut stages = Vec::with_capacity(3);
    for quadrature in quadratures {
        stages.push(run_heston(
            request,
            &HestonJobConfig {
                quadrature,
                ..config.clone()
            },
        )?);
    }
    let (absolute_tolerance, relative_tolerance) = match config.precision {
        Precision::F32 => (3e-4, 5e-5),
        Precision::F64 => (1e-8, 1e-9),
    };
    let mut checks = Vec::with_capacity(22);
    for (i, direction) in ["nodes", "cutoff"].into_iter().enumerate() {
        for ((name, coarse), (_, fine)) in metrics(&stages[i].values)
            .into_iter()
            .zip(metrics(&stages[i + 1].values))
        {
            let mut check = RefinementCheck {
                direction,
                metric: name,
                max_absolute_difference: 0.0,
                max_tolerance_ratio: 0.0,
                mismatched_rows: 0,
            };
            if coarse.len() != fine.len() {
                return Err(ComputeError::Execution(
                    "Heston refinement result shape mismatch".into(),
                ));
            }
            for (a, b) in coarse.iter().zip(fine) {
                let difference = (a - b).abs();
                let ratio = difference / (absolute_tolerance + relative_tolerance * b.abs());
                check.max_absolute_difference = check.max_absolute_difference.max(difference);
                check.max_tolerance_ratio = check.max_tolerance_ratio.max(ratio);
                check.mismatched_rows += usize::from(!ratio.is_finite() || ratio > 1.0);
            }
            checks.push(check);
        }
    }
    let passed = checks.iter().all(|c| c.mismatched_rows == 0);
    Ok(CheckedHestonResult {
        stages,
        checks,
        absolute_tolerance,
        relative_tolerance,
        passed,
    })
}
