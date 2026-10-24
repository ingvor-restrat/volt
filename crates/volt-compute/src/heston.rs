//! Semi-analytic European Heston prices and risk controls, not Monte Carlo.
//!
//! Characteristic function: Albrecher, Mayer, Schoutens & Tistaert,
//! *The Little Heston Trap*, Eq. (2), negative-d representation:
//! <https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/HestonTrap.pdf>.
//! Write a=kappa-rho*xi*i*z, d=sqrt(a*a+xi*xi*(z*z+i*z)),
//! g=(a-d)/(a+d), C=kappa*theta/xi²*((a-d)*T-2*log((1-g*exp(-d*T))/(1-g))),
//! D=(a-d)/xi²*(1-exp(-d*T))/(1-g*exp(-d*T)), phi=exp(C+D*v0).
//! The square root has positive real part. Cancellation in a-d is avoided by
//! rationalization, and the two logarithms are evaluated with complex log1p.
//!
//! The half-contour Fourier payoff inversion (Lewis, 2001,
//! <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=282110>) used here is
//! C= S exp(-qT) - A integral Re[exp(i*u*x)*phi(u-i/2)]/(u²+1/4) du,
//! A=sqrt(S*K)*exp(-(r+q)*T/2)/pi, x=ln(S/K)+(r-q)*T.
//! We evaluate the exact control-variate identity
//! C_H=C_BS(theta)-A integral Re[exp(i*u*x)*(phi_H-phi_BS)]/(u²+1/4) du,
//! phi_BS=exp(-theta*T*(u²+1/4)/2). The reference call uses the shared
//! Black-Scholes graph with volatility sqrt(theta); no market BS volatility is
//! supplied or substituted for stochastic variance. Subtracting characteristic
//! functions inside the integrand avoids cancellation of two values near S.
//! Differentiating twice gives gamma=gamma_BS+A/S² integral Re[exp(i*u*x)*
//! (phi_H-phi_BS)] du. This also differentiates the finite quadrature exactly;
//! the analytic gamma_BS is only the canceled reference component, not a
//! substituted Heston Greek. Both reference and correction participate in AD.
//! Puts use exact put-call parity. Reverse AD differentiates the same quadrature
//! graph for all first derivatives. Only first-order reverse AD is supported.
//!
//! Gauss-Legendre nodes t on `[0,1]` are mapped to u=cutoff*t², with
//! Jacobian 2*cutoff*t. This clusters frequencies near zero while retaining the
//! long tail needed at zero current variance. This fixed quadrature is NOT an accuracy certificate.
//! Both node count and cutoff must be refined independently, especially for
//! short maturity and large vol-of-variance. Extending the allowed cutoff to
//! 1e7 admits one-day zero-variance tails; it does not certify oscillatory
//! far-moneyness states at the fixed 2048-node cap. No Feller restriction is imposed.

use crate::{ComputeError, OptionSide, Precision, Session, pricing::black_scholes_prices};
use burn::tensor::{DType, Gradients, Tensor, TensorCreationOptions, TensorData};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// No Black-Scholes volatility field: all stochastic variance inputs are explicit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HestonOption {
    pub row_id: String,
    pub side: OptionSide,
    pub spot: f64,
    pub strike: f64,
    pub maturity_years: f64,
    pub rate: f64,
    pub dividend_yield: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HestonParams {
    /// Current CIR variance may equal zero; its Greek is the right derivative.
    pub initial_variance: f64,
    pub kappa: f64,
    pub long_run_variance: f64,
    pub vol_of_variance: f64,
    pub correlation: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuadratureConfig {
    pub nodes: usize,
    pub upper_bound: f64,
}
impl Default for QuadratureConfig {
    fn default() -> Self {
        Self {
            nodes: 256,
            upper_bound: 200.0,
        }
    }
}

/// Per underlying unit, in row order. Variance sensitivities are per variance
/// unit, not volatility points. Rate/yield/correlation Greeks are per 1.0 change;
/// theta is -dP/dT in currency/year. Gamma is d²P/dS² from Fourier differentiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HestonResult {
    pub prices: Vec<f64>,
    pub delta: Vec<f64>,
    pub gamma: Vec<f64>,
    pub initial_variance_sensitivity: Vec<f64>,
    pub kappa_sensitivity: Vec<f64>,
    pub long_run_variance_sensitivity: Vec<f64>,
    pub vol_of_variance_sensitivity: Vec<f64>,
    pub correlation_sensitivity: Vec<f64>,
    pub rho: Vec<f64>,
    pub theta: Vec<f64>,
    pub dividend_sensitivity: Vec<f64>,
    pub max_sampled_allocator_live_bytes: Option<u64>,
    pub max_sampled_allocator_reserved_bytes: Option<u64>,
}

/// Absolute tape shape bound, in addition to the session's expanded-cell cap.
/// This is an admission estimate, not an allocator quota or a measured peak.
pub const MAX_QUADRATURE_CELLS: usize = 262_144;

pub fn validate_inputs(
    options: &[HestonOption],
    params: &[HestonParams],
    quadrature: QuadratureConfig,
    precision: Precision,
) -> Result<usize, ComputeError> {
    let invalid = |s: &str| ComputeError::InvalidInput(s.into());
    if options.is_empty() || options.len() != params.len() {
        return Err(invalid(
            "Heston requires nonempty matching option and parameter slices",
        ));
    }
    if !(16..=2048).contains(&quadrature.nodes)
        || !quadrature.upper_bound.is_finite()
        || !(1.0..=10_000_000.0).contains(&quadrature.upper_bound)
    {
        return Err(invalid(
            "Heston quadrature requires 16..=2048 nodes and finite cutoff in 1..=10000000",
        ));
    }
    let cells = options
        .len()
        .checked_mul(quadrature.nodes)
        .ok_or_else(|| invalid("Heston quadrature shape overflow"))?;
    if cells > MAX_QUADRATURE_CELLS {
        return Err(invalid("Heston quadrature exceeds absolute cell cap"));
    }
    let mut identities = HashSet::with_capacity(options.len());
    for (o, p) in options.iter().zip(params) {
        if o.row_id.trim().is_empty() || !identities.insert(o.row_id.as_str()) {
            return Err(invalid("Heston row IDs must be nonempty and unique"));
        }
        let values = [
            o.spot,
            o.strike,
            o.maturity_years,
            o.rate,
            o.dividend_yield,
            p.initial_variance,
            p.kappa,
            p.long_run_variance,
            p.vol_of_variance,
            p.correlation,
        ];
        if values
            .iter()
            .any(|x| !x.is_finite() || (precision == Precision::F32 && !(*x as f32).is_finite()))
        {
            return Err(invalid(
                "Heston inputs must be finite in selected precision",
            ));
        }
        let positive = [
            o.spot,
            o.strike,
            o.maturity_years,
            p.kappa,
            p.long_run_variance,
            p.vol_of_variance,
        ];
        if positive
            .iter()
            .any(|x| *x <= 0.0 || (precision == Precision::F32 && (*x as f32) <= 0.0))
            || p.initial_variance < 0.0
            || p.correlation.abs() >= 1.0
            || (precision == Precision::F32 && (p.correlation as f32).abs() >= 1.0)
        {
            return Err(invalid(
                "Heston requires nonnegative current variance, positive spot/strike/time/long-run variance/kappa/xi and |correlation|<1; zero-xi boundary is not supported",
            ));
        }
    }
    Ok(cells)
}

/// Low-level batch: caller owns job serialization and memory admission. Session
/// row cap counts expanded option*quadrature cells, not contracts. A fresh graph
/// is created for every call. The cutoff itself is a fixed, untracked parameter.
pub fn differentiate_rows(
    session: &Session,
    options: &[HestonOption],
    params: &[HestonParams],
    quadrature: QuadratureConfig,
) -> Result<HestonResult, ComputeError> {
    execute(session, options, params, quadrature, true)
}

/// Untracked forward execution of exactly the same graph, useful for independent
/// node/cutoff refinement and bumped-price comparisons without allocating a tape.
pub fn price_rows(
    session: &Session,
    options: &[HestonOption],
    params: &[HestonParams],
    quadrature: QuadratureConfig,
) -> Result<Vec<f64>, ComputeError> {
    execute(session, options, params, quadrature, false).map(|x| x.prices)
}

#[derive(Clone)]
struct Complex {
    re: Tensor<2>,
    im: Tensor<2>,
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
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re.clone() * b.re.clone() - self.im.clone() * b.im.clone(),
            im: self.re * b.im + self.im * b.re,
        }
    }
    fn scale(self, b: Tensor<2>) -> Self {
        Self {
            re: self.re * b.clone(),
            im: self.im * b,
        }
    }
    fn div(self, b: Self) -> Self {
        let den = b.re.clone().square() + b.im.clone().square();
        Self {
            re: (self.re.clone() * b.re.clone() + self.im.clone() * b.im.clone()) / den.clone(),
            im: (self.im * b.re - self.re * b.im) / den,
        }
    }
    fn exp(self) -> Self {
        let mag = self.re.exp();
        Self {
            re: mag.clone() * self.im.clone().cos(),
            im: mag * self.im.sin(),
        }
    }
    fn sqrt(self) -> Self {
        let mag = (self.re.clone().square() + self.im.clone().square()).sqrt();
        let re = ((mag + self.re) * 0.5).sqrt();
        let im = self.im / (re.clone() * 2.0);
        Self { re, im }
    }
    /// log(1+z), preserving small complex increments.
    fn log1p(self) -> Self {
        Self {
            re: (self.re.clone() * 2.0 + self.re.clone().square() + self.im.clone().square())
                .log1p()
                * 0.5,
            im: self.im.atan2(self.re + 1.0),
        }
    }
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

fn execute(
    session: &Session,
    options: &[HestonOption],
    params: &[HestonParams],
    quadrature: QuadratureConfig,
    backward: bool,
) -> Result<HestonResult, ComputeError> {
    let cells = validate_inputs(options, params, quadrature, session.precision())?;
    if cells > session.max_batch_rows() {
        return Err(ComputeError::InvalidInput(format!(
            "Heston expanded batch {cells} exceeds session cap {}",
            session.max_batch_rows()
        )));
    }
    let device = if backward {
        session.device().clone().autodiff()
    } else {
        session.device().clone()
    };
    let dtype = match session.precision() {
        Precision::F32 => DType::F32,
        Precision::F64 => DType::F64,
    };
    let tensor = |v: Vec<f64>, shape: [usize; 2]| {
        Tensor::<2>::from_data(
            TensorData::new(v, shape),
            TensorCreationOptions::new(device.clone()).with_dtype(dtype),
        )
    };
    let leaf = |v: Vec<f64>| {
        let t = tensor(v, [options.len(), 1]);
        if backward { t.require_grad() } else { t }
    };
    let spot = leaf(options.iter().map(|o| o.spot).collect());
    let time = leaf(options.iter().map(|o| o.maturity_years).collect());
    let rate = leaf(options.iter().map(|o| o.rate).collect());
    let dividend = leaf(options.iter().map(|o| o.dividend_yield).collect());
    let v0 = leaf(params.iter().map(|p| p.initial_variance).collect());
    let kappa = leaf(params.iter().map(|p| p.kappa).collect());
    let variance = leaf(params.iter().map(|p| p.long_run_variance).collect());
    let xi = leaf(params.iter().map(|p| p.vol_of_variance).collect());
    let corr = leaf(params.iter().map(|p| p.correlation).collect());
    let strike = tensor(
        options.iter().map(|o| o.strike).collect(),
        [options.len(), 1],
    );
    let put = tensor(
        options
            .iter()
            .map(|o| {
                if matches!(o.side, OptionSide::Put) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
        [options.len(), 1],
    );
    let (nodes, weights) = gauss_legendre(quadrature.nodes, quadrature.upper_bound);
    let u = tensor(nodes, [1, quadrature.nodes]);
    let weights = tensor(weights, [1, quadrature.nodes]);
    // z=u-i/2 => z²+i*z = u²+1/4, purely real.
    let h = u.clone().square() + 0.25;
    let a = Complex {
        re: kappa.clone() - corr.clone() * xi.clone() * 0.5,
        im: -(corr.clone() * xi.clone() * u.clone()),
    };
    let aa = a.clone().mul(a.clone());
    let d = Complex {
        re: aa.re + xi.clone().square() * h.clone(),
        im: aa.im,
    }
    .sqrt();
    let apd = a.add(d.clone());
    // t=(a-d)/xi²=-(z²+i*z)/(a+d): avoids cancellation as xi -> 0.
    let zero = u.clone() * 0.0;
    let t = Complex {
        re: -h.clone(),
        im: zero.clone(),
    }
    .div(apd.clone());
    let g = t.clone().scale(xi.clone().square()).div(apd);
    let e = d.scale(-time.clone()).exp();
    let ge = g.clone().mul(e.clone());
    let log_ratio = ge.clone().neg().log1p().sub(g.clone().neg().log1p());
    let c = t
        .clone()
        .scale(time.clone())
        .sub(log_ratio.scale(xi.clone().square().recip() * 2.0))
        .scale(kappa.clone() * variance.clone());
    let one_minus_e = Complex {
        re: 1.0 - e.re,
        im: -e.im,
    };
    let one_minus_ge = Complex {
        re: 1.0 - ge.re,
        im: -ge.im,
    };
    let dd = t.mul(one_minus_e).div(one_minus_ge);
    let phi = c.add(dd.scale(v0.clone())).exp();
    let x =
        (spot.clone() / strike.clone()).log() + (rate.clone() - dividend.clone()) * time.clone();
    let phase = u * x;
    let reference_phi = (-variance.clone() * time.clone() * h.clone() * 0.5).exp();
    let real = (phi.re - reference_phi) * phase.clone().cos() - phi.im * phase.sin();
    let integral = (real.clone() / h * weights.clone()).sum_dim(1);
    let discounted_spot = spot.clone() * (-dividend.clone() * time.clone()).exp();
    let discounted_strike = strike.clone() * (-rate.clone() * time.clone()).exp();
    let prefactor = (spot.clone() * strike.clone()).sqrt()
        * (-(rate.clone() + dividend.clone()) * time.clone() * 0.5).exp()
        * std::f64::consts::FRAC_1_PI;
    let reference_call = black_scholes_prices(
        spot.clone().reshape([options.len()]),
        strike.clone().reshape([options.len()]),
        time.clone().reshape([options.len()]),
        variance.clone().sqrt().reshape([options.len()]),
        rate.clone().reshape([options.len()]),
        dividend.clone().reshape([options.len()]),
        (strike.clone() * 0.0 + 1.0).reshape([options.len()]),
    )
    .reshape([options.len(), 1]);
    let price =
        reference_call - prefactor.clone() * integral + put * (discounted_strike - discounted_spot);
    // Gamma is a separate Fourier derivative, so its reference arithmetic need
    // not allocate reverse tape nodes. The first-order price graph above keeps
    // every reference dependency tracked, including theta through sqrt(theta).
    let plain_spot = spot.clone().without_autodiff();
    let plain_time = time.clone().without_autodiff();
    let root_variance_time = (variance.clone().without_autodiff() * plain_time.clone()).sqrt();
    let reference_d1 = ((plain_spot.clone() / strike.without_autodiff()).log()
        + (rate.clone().without_autodiff() - dividend.clone().without_autodiff())
            * plain_time.clone()
        + root_variance_time.clone().square() * 0.5)
        / root_variance_time.clone();
    let reference_gamma = (-dividend.clone().without_autodiff() * plain_time).exp()
        * (-reference_d1.square() * 0.5).exp()
        / (plain_spot.clone() * root_variance_time)
        * (1.0 / (2.0 * std::f64::consts::PI).sqrt());
    let gamma = reference_gamma
        + prefactor.without_autodiff() / plain_spot.square()
            * (real.without_autodiff() * weights.without_autodiff()).sum_dim(1);
    session.synchronize()?;
    let mut telemetry = Telemetry::default();
    telemetry.sample(session);
    let prices = read(session, &price, options.len(), "price")?;
    let gamma = read(session, &gamma, options.len(), "gamma")?;
    let mut result = HestonResult {
        prices,
        gamma,
        delta: vec![],
        initial_variance_sensitivity: vec![],
        kappa_sensitivity: vec![],
        long_run_variance_sensitivity: vec![],
        vol_of_variance_sensitivity: vec![],
        correlation_sensitivity: vec![],
        rho: vec![],
        theta: vec![],
        dividend_sensitivity: vec![],
        max_sampled_allocator_live_bytes: None,
        max_sampled_allocator_reserved_bytes: None,
    };
    if backward {
        let mut grads = price.sum().backward();
        session.synchronize()?;
        telemetry.sample(session);
        result.delta = gradient(session, &spot, &mut grads, options.len(), "delta")?;
        result.initial_variance_sensitivity =
            gradient(session, &v0, &mut grads, options.len(), "v0")?;
        result.kappa_sensitivity = gradient(session, &kappa, &mut grads, options.len(), "kappa")?;
        result.long_run_variance_sensitivity = gradient(
            session,
            &variance,
            &mut grads,
            options.len(),
            "long run variance",
        )?;
        result.vol_of_variance_sensitivity =
            gradient(session, &xi, &mut grads, options.len(), "vol of variance")?;
        result.correlation_sensitivity =
            gradient(session, &corr, &mut grads, options.len(), "correlation")?;
        result.rho = gradient(session, &rate, &mut grads, options.len(), "rate")?;
        result.theta = gradient(session, &time, &mut grads, options.len(), "time")?
            .into_iter()
            .map(|x| -x)
            .collect();
        result.dividend_sensitivity =
            gradient(session, &dividend, &mut grads, options.len(), "dividend")?;
    }
    result.max_sampled_allocator_live_bytes = telemetry.live;
    result.max_sampled_allocator_reserved_bytes = telemetry.reserved;
    Ok(result)
}

fn read(
    session: &Session,
    tensor: &Tensor<2>,
    len: usize,
    name: &str,
) -> Result<Vec<f64>, ComputeError> {
    let v = session.read_vector(&tensor.clone().without_autodiff().reshape([len]))?;
    if v.len() != len || v.iter().any(|x| !x.is_finite()) {
        return Err(ComputeError::Execution(format!(
            "nonfinite Heston {name}; refine precision or reject input domain"
        )));
    }
    Ok(v)
}
fn gradient(
    session: &Session,
    leaf: &Tensor<2>,
    grads: &mut Gradients,
    len: usize,
    name: &str,
) -> Result<Vec<f64>, ComputeError> {
    let t = leaf
        .grad_remove(grads)
        .ok_or_else(|| ComputeError::Execution(format!("missing Heston {name} gradient")))?;
    read(session, &t, len, name)
}
#[derive(Default)]
struct Telemetry {
    live: Option<u64>,
    reserved: Option<u64>,
}
impl Telemetry {
    fn sample(&mut self, session: &Session) {
        if let Some(u) = session.device().memory_pool_usage() {
            self.live = Some(self.live.unwrap_or(0).max(u.bytes_in_use));
            self.reserved = Some(self.reserved.unwrap_or(0).max(u.bytes_reserved));
        }
    }
}

/// Deterministic host setup only; roots/weights are not pricing/reference data.
/// Integrate in t on [0,1], u=upper*t² and du=2*upper*t dt.
/// Keeping small-frequency resolution as the cutoff grows is essential near V0=0.
fn gauss_legendre(n: usize, upper: f64) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..32 {
            let (p, previous) = legendre(n, x);
            let derivative = n as f64 * (x * p - previous) / (x * x - 1.0);
            let step = p / derivative;
            x -= step;
            if step.abs() < 2e-15 {
                break;
            }
        }
        let (p, previous) = legendre(n, x);
        let derivative = n as f64 * (x * p - previous) / (x * x - 1.0);
        let weight = 1.0 / ((1.0 - x * x) * derivative * derivative);
        let lower_t = 0.5 * (1.0 - x);
        let upper_t = 0.5 * (1.0 + x);
        nodes[i] = upper * lower_t * lower_t;
        nodes[n - 1 - i] = upper * upper_t * upper_t;
        weights[i] = weight * 2.0 * upper * lower_t;
        weights[n - 1 - i] = weight * 2.0 * upper * upper_t;
    }
    (nodes, weights)
}
fn legendre(n: usize, x: f64) -> (f64, f64) {
    let (mut previous, mut current) = (1.0, x);
    for k in 2..=n {
        let next = ((2 * k - 1) as f64 * x * current - (k - 1) as f64 * previous) / k as f64;
        previous = current;
        current = next;
    }
    (current, previous)
}
