//! Independent scalar-f64 Heston reference with adaptive, phase-resolved panels.
//!
//! Characteristic function: Albrecher/Mayer/Schoutens/Tistaert, *The Little
//! Heston Trap*, Eq.(2), negative-d representation:
//! <https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/HestonTrap.pdf>.
//! The Lewis half-contour payoff identity and exact BS control variate are
//! transcribed in `heston`. Here J integrates Re(exp(i*u*x)*(phiH-phiBS))/h,
//! Jx integrates Re(i*u*exp(i*u*x)*(phiH-phiBS))/h, and Jv integrates
//! Re(exp(i*u*x)*phiH*D)/h, h=u²+1/4. Consequently
//! C=CBS-A*J, delta=deltaBS-A/S*(J/2+Jx), Cv=-A*Jv.
//! Puts use parity. No price finite differences, clipping, or proxy model occur.
//!
//! The 15/7 Gauss-Kronrod mathematical nodes/weights and the embedded-error
//! rescaling specification are from Netlib QUADPACK dqk15:
//! <https://www.netlib.org/quadpack/dqk15.f>. This is a separate Rust panel
//! implementation, not a binding or a port of a third-party pricing engine.
//! Each panel is additionally bounded in width by the known Fourier phase;
//! adaptive error agreement alone cannot certify an undersampled oscillation.
//! Tail panels double the cutoff. In the domain documented by analytic_tail_bounds,
//! a positive characteristic-function modulus bound controls the remaining tail.
//! Otherwise TWO successive small shell changes and decaying positive envelopes
//! are required, with empirical error estimates and a safety factor. The chosen
//! tail method is reported. Embedded quadrature remains a numerical estimate;
//! independent model controls remain necessary. Reaching ANY work/cutoff limit returns an error, never a fallback.

use crate::{
    ComputeError, OptionRow, OptionSide, Precision,
    aad::analytic_greeks,
    heston::{HestonOption, HestonParams, QuadratureConfig, validate_inputs},
};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

const MAX_DEPTH: usize = 64;
/// DFS retains at most 65 interval records, with a 15x6 scratch sample array.
/// Per concurrently executing scalar query; excludes caller inputs/results and
/// thread/runtime stacks. This is a logical working estimate, not an OS quota.
pub const MODELED_WORKSPACE_BYTES: u64 = 32 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveHestonConfig {
    pub price_absolute_tolerance: f64,
    pub delta_absolute_tolerance: f64,
    pub variance_absolute_tolerance: f64,
    pub initial_cutoff: f64,
    pub max_cutoff: f64,
    pub max_evaluations: usize,
    /// Total evaluated panels, not simultaneously retained interval records.
    pub max_panels: usize,
}
impl Default for AdaptiveHestonConfig {
    fn default() -> Self {
        Self {
            price_absolute_tolerance: 1e-7,
            delta_absolute_tolerance: 1e-7,
            variance_absolute_tolerance: 1e-5,
            initial_cutoff: 64.0,
            max_cutoff: 10_000_000.0,
            max_evaluations: 2_000_000,
            max_panels: 131_072,
        }
    }
}
impl AdaptiveHestonConfig {
    pub fn validate(&self) -> Result<(), ComputeError> {
        if [
            self.price_absolute_tolerance,
            self.delta_absolute_tolerance,
            self.variance_absolute_tolerance,
            self.initial_cutoff,
            self.max_cutoff,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.0)
            || self.initial_cutoff < 1.0
            || self.max_cutoff > 100_000_000.0
            || self.initial_cutoff > self.max_cutoff * 0.25
            || self.max_evaluations < 45
            || self.max_panels < 3
        {
            return Err(invalid(
                "invalid adaptive tolerances, cutoffs or work limits; at least two cutoff doublings required",
            ));
        }
        Ok(())
    }
    pub fn modeled_workspace_bytes(&self) -> u64 {
        MODELED_WORKSPACE_BYTES
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveErrorEstimates {
    pub price: f64,
    pub delta: f64,
    pub initial_variance_sensitivity: f64,
    /// Financial units in order price, delta, dPrice/dV0.
    pub quadrature: [f64; 3],
    /// Positive-envelope and successive-shell estimates, same financial units.
    pub tail: [f64; 3],
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveTailMethod {
    #[default]
    ObservedEnvelope,
    AnalyticCharacteristicBound,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveHestonResult {
    pub price: f64,
    pub delta: f64,
    pub initial_variance_sensitivity: f64,
    pub error_estimates: AdaptiveErrorEstimates,
    pub evaluations: usize,
    pub panels: usize,
    pub cutoff: f64,
    #[serde(default)]
    pub tail_method: AdaptiveTailMethod,
    pub converged: bool,
}

#[derive(Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}
impl Complex {
    fn add(self, b: Self) -> Self {
        Self {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
    fn sub(self, b: Self) -> Self {
        Self {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }
    fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }
    fn div(self, b: Self) -> Self {
        let d = b.re * b.re + b.im * b.im;
        Self {
            re: (self.re * b.re + self.im * b.im) / d,
            im: (self.im * b.re - self.re * b.im) / d,
        }
    }
    fn sqrt(self) -> Self {
        let re = ((self.re.hypot(self.im) + self.re) * 0.5).sqrt();
        Self {
            re,
            im: self.im / (2.0 * re),
        }
    }
    fn exp(self) -> Self {
        let m = self.re.exp();
        let (s, c) = self.im.sin_cos();
        Self {
            re: m * c,
            im: m * s,
        }
    }
    fn log1p(self) -> Self {
        Self {
            re: (2.0 * self.re + self.re * self.re + self.im * self.im).ln_1p() * 0.5,
            im: self.im.atan2(1.0 + self.re),
        }
    }
    fn modulus(self) -> f64 {
        self.re.hypot(self.im)
    }
}
struct Integrand<'a> {
    option: &'a HestonOption,
    params: &'a HestonParams,
    x: f64,
    a: f64,
    phase_width: f64,
}
impl Integrand<'_> {
    fn values(&self, u: f64) -> Result<[f64; 6], ComputeError> {
        let p = self.params;
        let time = self.option.maturity_years;
        let h = u * u + 0.25;
        let aa = Complex {
            re: p.kappa - 0.5 * p.correlation * p.vol_of_variance,
            im: -p.correlation * p.vol_of_variance * u,
        };
        let square = aa.mul(aa);
        let d = Complex {
            re: square.re + p.vol_of_variance * p.vol_of_variance * h,
            im: square.im,
        }
        .sqrt();
        let apd = aa.add(d);
        let t = Complex { re: -h, im: 0.0 }.div(apd);
        let g = t.scale(p.vol_of_variance * p.vol_of_variance).div(apd);
        let e = d.scale(-time).exp();
        let ge = g.mul(e);
        let logarithm = ge.scale(-1.0).log1p().sub(g.scale(-1.0).log1p());
        let c = t
            .scale(time)
            .sub(logarithm.scale(2.0 / (p.vol_of_variance * p.vol_of_variance)))
            .scale(p.kappa * p.long_run_variance);
        let variance_coefficient = t
            .mul(Complex {
                re: 1.0 - e.re,
                im: -e.im,
            })
            .div(Complex {
                re: 1.0 - ge.re,
                im: -ge.im,
            });
        let phi = c.add(variance_coefficient.scale(p.initial_variance)).exp();
        let reference = (-0.5 * p.long_run_variance * time * h).exp();
        let difference = phi.sub(Complex {
            re: reference,
            im: 0.0,
        });
        let (s, c) = (u * self.x).sin_cos();
        let phase = Complex { re: c, im: s };
        let rotated = phase.mul(difference);
        let variance = phi.mul(variance_coefficient);
        let rotated_variance = phase.mul(variance);
        let envelope = difference.modulus() / h;
        let values = [
            rotated.re / h,
            -u * rotated.im / h,
            rotated_variance.re / h,
            envelope,
            u * envelope,
            variance.modulus() / h,
        ];
        if values.iter().any(|x| !x.is_finite()) {
            return Err(ComputeError::Execution(
                "nonfinite adaptive Heston integrand".into(),
            ));
        }
        Ok(values)
    }
    fn financial(&self, raw: [f64; 3]) -> [f64; 3] {
        [
            self.a * raw[0],
            self.a / self.option.spot * (0.5 * raw[0] + raw[1]),
            self.a * raw[2],
        ]
    }
}
#[derive(Default, Clone, Copy)]
struct Sum {
    sum: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, x: f64) {
        let y = x - self.correction;
        let t = self.sum + y;
        self.correction = (t - self.sum) - y;
        self.sum = t;
    }
    fn value(self) -> f64 {
        self.sum
    }
}
struct Counters {
    evaluations: usize,
    panels: usize,
}
struct Shell {
    value: [f64; 6],
    error: [f64; 6],
}

pub fn adaptive_greeks(
    option: &HestonOption,
    params: &HestonParams,
    config: AdaptiveHestonConfig,
) -> Result<AdaptiveHestonResult, ComputeError> {
    config.validate()?;
    // Reuse the domain/identity contract of the tensor engine, without tensors or
    // its finite quadrature node choice participating in this independent solver.
    validate_inputs(
        std::slice::from_ref(option),
        std::slice::from_ref(params),
        QuadratureConfig {
            nodes: 16,
            upper_bound: 1.0,
        },
        Precision::F64,
    )?;
    let reference_row = OptionRow {
        row_id: option.row_id.clone(),
        contract_id: "heston-reference-component".into(),
        side: OptionSide::Call,
        spot: option.spot,
        strike: option.strike,
        maturity_years: option.maturity_years,
        volatility: params.long_run_variance.sqrt(),
        rate: option.rate,
        dividend_yield: option.dividend_yield,
        quantity: 1,
        multiplier: 1,
    };
    let reference = analytic_greeks(&reference_row)?;
    let x = (option.spot / option.strike).ln()
        + (option.rate - option.dividend_yield) * option.maturity_years;
    let a = (option.spot * option.strike).sqrt()
        * (-0.5 * (option.rate + option.dividend_yield) * option.maturity_years).exp()
        / PI;
    // High-frequency CF phase includes rho*(V0+kappa*theta*T)/xi. The extra
    // positive bound guards transitional phase curvature; panel quadrature then
    // adapts further wherever the embedded error requires it.
    let phase_frequency = x.abs()
        + (params.initial_variance
            + params.kappa * params.long_run_variance * option.maturity_years)
            * (1.0 + params.correlation.abs())
            / params.vol_of_variance;
    let phase_width = PI / (2.0 * phase_frequency.max(1e-12));
    if !a.is_finite()
        || a <= 0.0
        || !x.is_finite()
        || !phase_frequency.is_finite()
        || !phase_width.is_finite()
        || phase_width <= 0.0
    {
        return Err(invalid(
            "adaptive Heston scales/phase are not representable",
        ));
    }
    let integrand = Integrand {
        option,
        params,
        x,
        a,
        phase_width,
    };
    let tolerances = [
        config.price_absolute_tolerance,
        config.delta_absolute_tolerance,
        config.variance_absolute_tolerance,
    ];
    let mut totals = [Sum::default(); 3];
    let mut errors = [Sum::default(); 3];
    let mut counters = Counters {
        evaluations: 0,
        panels: 0,
    };
    let mut previous_envelope: Option<[f64; 3]> = None;
    let mut previous_ratio = [f64::INFINITY; 3];
    let mut previous_change = [f64::INFINITY; 3];
    let mut lower = 0.0;
    let mut cutoff = config.initial_cutoff;
    for stage in 0..64usize {
        let shell = integrate_shell(
            &integrand,
            lower,
            cutoff,
            stage,
            tolerances,
            &config,
            &mut counters,
        )?;
        for i in 0..3 {
            totals[i].add(shell.value[i]);
            errors[i].add(shell.error[i]);
        }
        let envelope = [
            shell.value[3] + shell.error[3],
            shell.value[4] + shell.error[4],
            shell.value[5] + shell.error[5],
        ];
        let change = integrand.financial([
            shell.value[0].abs(),
            shell.value[1].abs(),
            shell.value[2].abs(),
        ]);
        let ratio = std::array::from_fn(|i| match previous_envelope {
            Some(p) if p[i] > 0.0 => envelope[i] / p[i],
            Some(_) if envelope[i] == 0.0 => 0.0,
            _ => f64::INFINITY,
        });
        let raw_tail: [f64; 3] = std::array::from_fn(|i| {
            let r = ratio[i].max(previous_ratio[i]);
            if r.is_finite() && r < 0.5 {
                4.0 * envelope[i] * r / (1.0 - r)
            } else {
                f64::INFINITY
            }
        });
        let empirical_tail = integrand.financial(raw_tail);
        let raw_errors = errors.map(Sum::value);
        let mut quadrature = integrand.financial(raw_errors);
        let raw_total = totals.map(Sum::value);
        // Account for finite precision in reference arithmetic and accumulated sums,
        // independently of the embedded integration error.
        let rounding = 128.0 * f64::EPSILON;
        quadrature[0] += rounding
            * (reference.price.abs() + (a * raw_total[0]).abs() + option.spot + option.strike);
        quadrature[1] += rounding
            * (reference.delta.abs()
                + a / option.spot * (0.5 * raw_total[0].abs() + raw_total[1].abs())
                + 1.0);
        quadrature[2] += rounding * ((a * raw_total[2]).abs() + 1.0);
        let analytic_tail = analytic_tail_bounds(params, option.maturity_years, cutoff)
            .map(|raw| integrand.financial(raw));
        let analytic_acceptable = stage >= 2
            && analytic_tail
                .is_some_and(|tail| (0..3).all(|i| quadrature[i] + tail[i] <= tolerances[i]));
        let empirical_acceptable = stage >= 2
            && (0..3).all(|i| {
                quadrature[i] + empirical_tail[i] <= tolerances[i]
                    && change[i] <= 0.25 * tolerances[i]
                    && previous_change[i] <= 0.25 * tolerances[i]
            });
        // A proven modulus tail bound replaces empirical shell-decay inference
        // only in its stated domain. Quadrature/phase checks remain mandatory.
        let (tail, tail_method) = if analytic_acceptable {
            (
                analytic_tail.unwrap(),
                AdaptiveTailMethod::AnalyticCharacteristicBound,
            )
        } else {
            (empirical_tail, AdaptiveTailMethod::ObservedEnvelope)
        };
        let acceptable = analytic_acceptable || empirical_acceptable;
        if acceptable {
            let mut price = reference.price - a * raw_total[0];
            let mut delta = reference.delta - a / option.spot * (0.5 * raw_total[0] + raw_total[1]);
            let cv = -a * raw_total[2];
            if matches!(option.side, OptionSide::Put) {
                price += option.strike * (-option.rate * option.maturity_years).exp()
                    - option.spot * (-option.dividend_yield * option.maturity_years).exp();
                delta -= (-option.dividend_yield * option.maturity_years).exp();
            }
            if [price, delta, cv].iter().any(|x| !x.is_finite()) {
                return Err(ComputeError::Execution(
                    "nonfinite adaptive Heston result".into(),
                ));
            }
            return Ok(AdaptiveHestonResult {
                price,
                delta,
                initial_variance_sensitivity: cv,
                error_estimates: AdaptiveErrorEstimates {
                    price: quadrature[0] + tail[0],
                    delta: quadrature[1] + tail[1],
                    initial_variance_sensitivity: quadrature[2] + tail[2],
                    quadrature,
                    tail,
                },
                evaluations: counters.evaluations,
                panels: counters.panels,
                cutoff,
                tail_method,
                converged: true,
            });
        }
        previous_envelope = Some(envelope);
        previous_ratio = ratio;
        previous_change = change;
        lower = cutoff;
        if cutoff > config.max_cutoff * 0.5 {
            return Err(ComputeError::Execution(format!(
                "adaptive Heston tail did not converge by cutoff {cutoff}; estimates quadrature={quadrature:?}, tail={tail:?}, evaluations={}",
                counters.evaluations
            )));
        }
        cutoff *= 2.0;
    }
    Err(ComputeError::Execution(
        "adaptive Heston exceeded tail stage limit".into(),
    ))
}

/// Absolute tails of (J,Jx,Jv) BEFORE multiplying by A or A/S. Returns None
/// outside the proved domain or when bound arithmetic is not representable.
///
/// Proof (real arithmetic, principal square root): write a=a0-i*rho*xi*u,
/// a0=kappa-rho*xi/2 >=0, b=xi*sqrt(1-rho²), h=u²+1/4.
/// Re(d²)=a0²+xi²/4+b²*u², hence Re(d)>=b*u. Also
/// Re(a*conj(d))=a0*(Re(d)+(rho*xi*u)²/Re(d))>=0, so |g|<=1.
/// For t=(a-d)/xi²=-h/(a+d), Re(t)<=(a0-b*u)/xi² and
/// |t|<=h/(b*u). With e=exp(-d*T), D=t+t*(g-1)*e/(1-g*e).
/// At U*b*T>=1, u*exp(-b*T*u) decreases for all u>=U, giving
/// Re(D)<=(a0-b*u)/xi²+R(U),
/// R(U)=2*(U+1/(4U))*exp(-b*T*U)/(b*(1-exp(-b*T*U))).
/// Therefore |phiH(u)|<=K(U)*exp(-alpha*u), where
/// alpha=(V0+kappa*theta*T)*b/xi² and
/// log K=(V0+kappa*theta*T)*a0/xi²+V0*R(U)
///       +(2*kappa*theta/xi²)*log(2/(1-exp(-b*T*U))).
/// Furthermore |D|/h<=coth(b*T*U/2)/(b*u). Integrating these positive
/// exponential envelopes using 1/u<=1/U bounds all three tails. The BS
/// reference has the independent Gaussian envelope exp(-theta*T*u²/2).
///
/// The implementation inflates constants/denominators for f64 roundoff; it is
/// not an interval-arithmetic proof of the entire price or embedded quadrature.
pub fn analytic_tail_bounds(p: &HestonParams, time: f64, cutoff: f64) -> Option<[f64; 3]> {
    let values = [
        p.initial_variance,
        p.kappa,
        p.long_run_variance,
        p.vol_of_variance,
        p.correlation,
        time,
        cutoff,
    ];
    if values.iter().any(|v| !v.is_finite())
        || p.initial_variance < 0.0
        || p.kappa <= 0.0
        || p.long_run_variance <= 0.0
        || p.vol_of_variance <= 0.0
        || p.correlation.abs() >= 1.0
        || time <= 0.0
        || cutoff <= 0.0
    {
        return None;
    }
    let eps = 128.0 * f64::EPSILON;
    let xi2 = p.vol_of_variance * p.vol_of_variance;
    let a0 = p.kappa - p.correlation * p.vol_of_variance * 0.5;
    let a_error = eps * (p.kappa + (p.correlation * p.vol_of_variance * 0.5).abs());
    if a0 < a_error || !xi2.is_normal() {
        return None;
    }
    let b =
        p.vol_of_variance * ((1.0 - p.correlation) * (1.0 + p.correlation)).sqrt() * (1.0 - eps);
    let decay = b * time * (1.0 - eps);
    let z = decay * cutoff;
    let mass = p.initial_variance + p.kappa * p.long_run_variance * time;
    let alpha = mass * (1.0 - eps) * b / (xi2 * (1.0 + eps)) * (1.0 - eps);
    if !b.is_normal() || !decay.is_normal() || z < 1.0 || !alpha.is_normal() || !mass.is_finite() {
        return None;
    }
    let denominator = -(-z).exp_m1() * (1.0 - eps);
    let log_r = 2.0_f64.ln() + (cutoff + 0.25 / cutoff).ln() - b.ln() - z - denominator.ln();
    let variance_correction = if p.initial_variance == 0.0 {
        0.0
    } else {
        (p.initial_variance.ln() + log_r).exp()
    };
    let log_k = (mass * (1.0 + eps) * (a0 + a_error) / (xi2 * (1.0 - eps))
        + variance_correction * (1.0 + eps)
        + (2.0 * p.kappa * p.long_run_variance / (xi2 * (1.0 - eps)))
            * (1.0 + eps)
            * (2.0 / denominator).ln())
        * (1.0 + eps);
    if !log_k.is_finite() {
        return None;
    }
    let log_h = log_k - alpha * cutoff - alpha.ln();
    let coth = (1.0 + (-z).exp()) / denominator;
    let beta = 0.5 * p.long_run_variance * time * (1.0 - eps);
    if !beta.is_normal() {
        return None;
    }
    let gaussian = -beta * cutoff * cutoff - (2.0 * beta).ln();
    let exp_upper = |log: f64| -> Option<f64> {
        let value = log.exp() * (1.0 + 4.0 * eps);
        if value.is_finite() {
            Some(value.max(f64::MIN_POSITIVE))
        } else {
            None
        }
    };
    let j = exp_upper(log_h - 2.0 * cutoff.ln())? + exp_upper(gaussian - 3.0 * cutoff.ln())?;
    let jx = exp_upper(log_h - cutoff.ln())? + exp_upper(gaussian - 2.0 * cutoff.ln())?;
    let jv = exp_upper(log_h - cutoff.ln() - b.ln() + coth.ln())?;
    let result = [j * (1.0 + eps), jx * (1.0 + eps), jv * (1.0 + eps)];
    result.iter().all(|v| v.is_finite()).then_some(result)
}

fn integrate_shell(
    f: &Integrand<'_>,
    lower: f64,
    upper: f64,
    stage: usize,
    tolerances: [f64; 3],
    config: &AdaptiveHestonConfig,
    counters: &mut Counters,
) -> Result<Shell, ComputeError> {
    let width = upper - lower;
    let count = (width / f.phase_width).ceil().max(1.0);
    if !count.is_finite()
        || count > config.max_panels.saturating_sub(counters.panels) as f64
        || count > config.max_evaluations.saturating_sub(counters.evaluations) as f64 / 15.0
    {
        return Err(work_limit(counters));
    }
    let count = count as usize;
    let mut value = [Sum::default(); 6];
    let mut error = [Sum::default(); 6];
    // Sum over all stages <=1/16 of each requested error budget.
    let allocation = (6.0 / (PI * PI)) / 16.0 / ((stage + 1) as f64).powi(2);
    let mut stack = Vec::with_capacity(MAX_DEPTH + 1);
    for panel in 0..count {
        let a = lower + width * (panel as f64 / count as f64);
        let b = if panel + 1 == count {
            upper
        } else {
            lower + width * ((panel + 1) as f64 / count as f64)
        };
        stack.push((a, b, 0usize));
        while let Some((a, b, depth)) = stack.pop() {
            if counters.panels >= config.max_panels
                || counters.evaluations > config.max_evaluations.saturating_sub(15)
            {
                return Err(work_limit(counters));
            }
            let result = panel_gk15(f, a, b)?;
            counters.panels += 1;
            counters.evaluations += 15;
            let allowed = tolerances.map(|t| t * allocation * (b - a) / width);
            let signed = f.financial([result.error[0], result.error[1], result.error[2]]);
            let envelope = f.financial([result.error[3], result.error[4], result.error[5]]);
            if (0..3).all(|i| signed[i] <= allowed[i] && envelope[i] <= allowed[i]) {
                for i in 0..6 {
                    value[i].add(result.value[i]);
                    error[i].add(result.error[i]);
                }
            } else {
                let middle = a + (b - a) * 0.5;
                if depth >= MAX_DEPTH || middle <= a || middle >= b {
                    return Err(ComputeError::Execution(
                        "adaptive Heston panel cannot meet tolerance at f64 resolution".into(),
                    ));
                }
                stack.push((middle, b, depth + 1));
                stack.push((a, middle, depth + 1));
            }
        }
    }
    Ok(Shell {
        value: value.map(Sum::value),
        error: error.map(Sum::value),
    })
}

// Mathematical constants rounded to binary64, Netlib dqk15 (Fullerton's table).
const X: [f64; 8] = [
    0.9914553711208126,
    0.9491079123427585,
    0.8648644233597691,
    0.7415311855993945,
    0.5860872354676911,
    0.4058451513773972,
    0.2077849550078985,
    0.0,
];
const WK: [f64; 8] = [
    0.022935322010529225,
    0.06309209262997855,
    0.10479001032225018,
    0.14065325971552592,
    0.1690047266392679,
    0.19035057806478542,
    0.20443294007529889,
    0.20948214108472782,
];
const WG: [f64; 4] = [
    0.1294849661688697,
    0.27970539148927667,
    0.3818300505051189,
    0.4179591836734694,
];
fn panel_gk15(f: &Integrand<'_>, a: f64, b: f64) -> Result<Shell, ComputeError> {
    let center = a + (b - a) * 0.5;
    let half = (b - a) * 0.5;
    let middle = f.values(center)?;
    let mut kronrod = middle.map(|v| v * WK[7]);
    let mut gauss = middle.map(|v| v * WG[3]);
    let mut abs = middle.map(|v| v.abs() * WK[7]);
    let mut samples = [[[0.0; 6]; 2]; 7];
    for j in 0..7 {
        let pair = [
            f.values(center - half * X[j])?,
            f.values(center + half * X[j])?,
        ];
        samples[j] = pair;
        for i in 0..6 {
            let sum = pair[0][i] + pair[1][i];
            kronrod[i] += WK[j] * sum;
            abs[i] += WK[j] * (pair[0][i].abs() + pair[1][i].abs());
            if j % 2 == 1 {
                gauss[i] += WG[j / 2] * sum;
            }
        }
    }
    let mut value = [0.0; 6];
    let mut error = [0.0; 6];
    for i in 0..6 {
        let mean = kronrod[i] * 0.5;
        let mut asc = WK[7] * (middle[i] - mean).abs();
        for j in 0..7 {
            asc += WK[j] * ((samples[j][0][i] - mean).abs() + (samples[j][1][i] - mean).abs());
        }
        asc *= half;
        let absolute = abs[i] * half;
        let mut err = ((kronrod[i] - gauss[i]) * half).abs();
        if asc > 0.0 && err > 0.0 {
            err = asc * (200.0 * err / asc).powf(1.5).min(1.0);
        }
        if absolute > f64::MIN_POSITIVE / (50.0 * f64::EPSILON) {
            err = err.max(50.0 * f64::EPSILON * absolute);
        }
        value[i] = kronrod[i] * half;
        error[i] = err;
    }
    Ok(Shell { value, error })
}
fn invalid(s: &str) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn work_limit(c: &Counters) -> ComputeError {
    ComputeError::BudgetExceeded(format!(
        "adaptive Heston work limit: {} evaluations, {} panels; no result certified",
        c.evaluations, c.panels
    ))
}

#[cfg(test)]
mod tail_bound_tests {
    use super::*;

    /// Independent positive-envelope integrals over [U,16U] are below the
    /// algebraic bounds for the whole infinite tail. This exercises the bound
    /// over zero/positive V0, both correlation signs and short/long maturities.
    #[test]
    fn modulus_tail_bound_dominates_numerical_positive_envelopes() {
        for correlation in [-0.9, -0.2, 0.5] {
            for initial_variance in [0.0, 0.04] {
                for time in [1.0 / 365.0, 0.5, 10.0] {
                    let p = HestonParams {
                        initial_variance,
                        kappa: 1.0,
                        long_run_variance: 0.04,
                        vol_of_variance: 2.0,
                        correlation,
                    };
                    let b = p.vol_of_variance * (1.0 - correlation * correlation).sqrt();
                    let alpha = (initial_variance + p.kappa * p.long_run_variance * time) * b
                        / (p.vol_of_variance * p.vol_of_variance);
                    let u = (4.0 / (b * time)).max(8.0 / alpha).max(1.0);
                    let bound = analytic_tail_bounds(&p, time, u).unwrap();
                    let option = HestonOption {
                        row_id: "tail-envelope".into(),
                        side: OptionSide::Call,
                        spot: 100.0,
                        strike: 100.0,
                        maturity_years: time,
                        rate: 0.0,
                        dividend_yield: 0.0,
                    };
                    let f = Integrand {
                        option: &option,
                        params: &p,
                        x: 0.0,
                        a: 1.0,
                        phase_width: 1.0,
                    };
                    let mut integral = [Sum::default(); 3];
                    for panel in 0..256 {
                        let a = u * 16.0_f64.powf(panel as f64 / 256.0);
                        let b = u * 16.0_f64.powf((panel + 1) as f64 / 256.0);
                        let estimate = panel_gk15(&f, a, b).unwrap();
                        for (i, total) in integral.iter_mut().enumerate() {
                            total.add(estimate.value[i + 3] + estimate.error[i + 3]);
                        }
                    }
                    for (actual, bound) in integral.map(Sum::value).into_iter().zip(bound) {
                        assert!(
                            actual <= bound * (1.0 + 1e-10),
                            "tail {actual} exceeds bound {bound}"
                        );
                    }
                }
            }
        }
    }
}
