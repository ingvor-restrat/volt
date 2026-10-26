//! CPU-f64 finite-difference reference for dated European, Bermudan and American rights.
//!
//! Between cash dividends, `V_t + sigma² S² V_SS/2 + (r-q)S V_S - rV = 0`.
//! A sinh mesh clusters points at strike. Crank–Nicolson rollback uses two implicit
//! half-steps after nonsmooth events (Rannacher damping); American exercise solves
//! `min(A V-b, V-payoff)=0` by policy iteration, with a checked residual.
//! Bermudan dates are inserted as exact integer timestamps, never rounded to steps.
//! Cash dividends apply `V(t-,S)=V(t+,max(S-D,0))`; coincident exercise occurs **before**
//! the jump. This is a fixed-cash/limited-liability spot-jump model, distinct from
//! an escrowed-dividend model. Valuation is after any already-paid dividend.
//!
//! Prices and Greeks are per underlying unit; quantity/multiplier aggregation is
//! the caller's responsibility. Rates/yields are flat continuously compounded,
//! volatility is an annual fraction, and time uses Actual/365 Fixed. There is no
//! settlement lag, funding spread, stochastic rate, or exercise/assignment ledger.
//! Finite space/time domains introduce discretization error: compare refinements.
//!
//! Primary method references: Rannacher, Numer. Math. 43 (1984), 309–327,
//! <https://doi.org/10.1007/BF01390130>; policy iteration: Reisinger and Witte,
//! <https://arxiv.org/abs/1012.4976>. The dividend jump convention is also
//! documented by QuantLib's independently implemented spot-dividend handler:
//! <https://github.com/lballabio/QuantLib/blob/v1.40/ql/methods/finitedifferences/utilities/fdmdividendhandler.cpp>.

use crate::{ComputeError, OptionSide};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use volt::{ExerciseSchedule, TimestampNs};

const YEAR_NS: f64 = 365.0 * 86_400.0 * 1e9;
const MAX_ROWS: usize = 128;
const MAX_EVENTS: usize = 512;
const MAX_ADDED_EVENTS: usize = MAX_EVENTS + 66;
fn invalid(s: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn execution(s: impl Into<String>) -> ComputeError {
    ComputeError::Execution(s.into())
}
fn years(a: i64, b: i64) -> f64 {
    (b as i128 - a as i128) as f64 / YEAR_NS
}

fn deserialize_schedule<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<ExerciseSchedule, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    let map = value
        .as_object()
        .ok_or_else(|| serde::de::Error::custom("exercise must be an object"))?;
    let allowed: &[&str] = match map.get("style").and_then(|v| v.as_str()) {
        Some("european") => &["style", "expiry"],
        Some("american") => &["style", "first_exercise", "expiry"],
        Some("bermudan") => &["style", "opportunities"],
        _ => return Err(serde::de::Error::custom("unknown exercise style")),
    };
    if map.keys().any(|k| !allowed.contains(&k.as_str())) {
        return Err(serde::de::Error::custom("unknown exercise field"));
    }
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CashDividend {
    pub ex_date: TimestampNs,
    pub amount: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExerciseOption {
    pub row_id: String,
    pub contract_id: String,
    pub side: OptionSide,
    pub spot: f64,
    pub strike: f64,
    pub valuation_time: TimestampNs,
    #[serde(deserialize_with = "deserialize_schedule")]
    pub exercise: ExerciseSchedule,
    pub volatility: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    #[serde(default)]
    pub cash_dividends: Vec<CashDividend>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExerciseGrid {
    /// Number of spatial intervals, with strike-focused nonuniform spacing.
    pub space_steps: usize,
    /// Base time intervals; all exercise/dividend/start dates are added exactly.
    pub time_steps: usize,
    pub s_max_multiple: f64,
    pub rannacher_intervals: usize,
    pub max_policy_iterations: usize,
    /// LCP residual after positive diagonal row scaling and value-scale normalization.
    pub complementarity_tolerance: f64,
    /// Absolute volatility fraction bump (central), e.g. 0.001 = 0.1 vol point.
    pub volatility_bump: f64,
    /// Absolute continuously compounded rate bump (central).
    pub rate_bump: f64,
    /// Calendar-forward theta bump, shortened before the next dated event.
    pub theta_bump_ns: u64,
}
impl Default for ExerciseGrid {
    fn default() -> Self {
        Self {
            space_steps: 800,
            time_steps: 800,
            s_max_multiple: 8.,
            rannacher_intervals: 2,
            max_policy_iterations: 80,
            complementarity_tolerance: 1e-10,
            volatility_bump: 0.001,
            rate_bump: 0.0001,
            theta_bump_ns: 60_000_000_000,
        }
    }
}
impl ExerciseGrid {
    pub fn validate(&self) -> Result<(), ComputeError> {
        if !(32..=4096).contains(&self.space_steps)
            || !(8..=8192).contains(&self.time_steps)
            || self
                .space_steps
                .checked_mul(self.time_steps + MAX_ADDED_EVENTS)
                .is_none_or(|n| n > 12_000_000)
            || !self.s_max_multiple.is_finite()
            || !(2.0..=1000.).contains(&self.s_max_multiple)
            || self.rannacher_intervals > 8
            || !(2..=500).contains(&self.max_policy_iterations)
            || !self.complementarity_tolerance.is_finite()
            || !(1e-13..=1e-5).contains(&self.complementarity_tolerance)
            || !self.volatility_bump.is_finite()
            || !(1e-5..=0.1).contains(&self.volatility_bump)
            || !self.rate_bump.is_finite()
            || !(1e-6..=0.01).contains(&self.rate_bump)
            || !(1_000_000_000..=86_400_000_000_000).contains(&self.theta_bump_ns)
        {
            return Err(invalid(
                "invalid or excessive exercise grid/bump configuration",
            ));
        }
        Ok(())
    }
}
impl ExerciseOption {
    pub fn validate(&self) -> Result<(), ComputeError> {
        self.exercise
            .validate()
            .map_err(|e| invalid(e.to_string()))?;
        if self.row_id.trim().is_empty()
            || self.contract_id.trim().is_empty()
            || self.row_id.len() > 256
            || self.contract_id.len() > 256
            || self.valuation_time.0 < 0
            || self.exercise.expiry().0 <= self.valuation_time.0
            || years(self.valuation_time.0, self.exercise.expiry().0) > 30.
            || !self.spot.is_finite()
            || !(1e-6..=1e9).contains(&self.spot)
            || !self.strike.is_finite()
            || !(1e-6..=1e9).contains(&self.strike)
            || !(0.02..=50.).contains(&(self.spot / self.strike))
            || !self.volatility.is_finite()
            || !(0.002..=3.).contains(&self.volatility)
            || !self.rate.is_finite()
            || self.rate.abs() > 0.5
            || !self.dividend_yield.is_finite()
            || self.dividend_yield.abs() > 0.5
            || self.cash_dividends.len() > 64
        {
            return Err(invalid(
                "unsupported exercise option identity, dates, scale or financial inputs",
            ));
        }
        if let ExerciseSchedule::Bermudan { opportunities } = &self.exercise
            && opportunities.len() > MAX_EVENTS
        {
            return Err(invalid("too many Bermudan opportunities"));
        }
        let mut last = self.valuation_time.0;
        for d in &self.cash_dividends {
            if d.ex_date.0 <= last
                || d.ex_date.0 >= self.exercise.expiry().0
                || !d.amount.is_finite()
                || d.amount <= 0.
                || d.amount > self.spot.max(self.strike) * 10.
            {
                return Err(invalid(
                    "cash dividends must be positive, ordered and strictly between valuation and expiry",
                ));
            }
            last = d.ex_date.0;
        }
        Ok(())
    }
    fn payoff(&self, s: f64) -> f64 {
        match self.side {
            OptionSide::Call => (s - self.strike).max(0.),
            OptionSide::Put => (self.strike - s).max(0.),
        }
    }
    fn american_at(&self, t: i64) -> bool {
        matches!(&self.exercise,ExerciseSchedule::American{first_exercise,expiry} if t>=first_exercise.0&&t<=expiry.0)
    }
    fn exercise_at(&self, t: i64) -> bool {
        self.exercise.permits_exercise(TimestampNs(t))
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExerciseDiagnostics {
    pub space_nodes: usize,
    pub time_nodes: usize,
    pub rollback_substeps: usize,
    pub policy_iterations: usize,
    pub max_scaled_complementarity_residual: f64,
    pub upwind_nodes: usize,
    pub s_max: f64,
    pub spot_grid_spacing: f64,
    pub spot_diffusion_width: f64,
    pub event_dates: Vec<TimestampNs>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExercisePrice {
    pub row_id: String,
    pub contract_id: String,
    pub price: f64,
    pub volatility: f64,
    pub intrinsic_value: f64,
    pub exercise_permitted_now: bool,
    pub at_exercise_value: bool,
    pub diagnostics: ExerciseDiagnostics,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseRisk {
    pub valuation: ExercisePrice,
    /// Per one spot currency unit, spatial nonuniform FD stencil.
    pub delta: f64,
    /// Per squared spot currency unit, spatial nonuniform FD stencil.
    pub gamma: f64,
    pub vega_per_unit_volatility: f64,
    pub vega_per_vol_point: f64,
    pub rho_per_unit_rate: f64,
    pub rho_per_bp: f64,
    /// Calendar-forward dPrice/dCalendarYear with fixed future dates.
    pub theta_per_year: Option<f64>,
    pub theta_per_day: Option<f64>,
    pub theta_bump_ns: Option<u64>,
    pub theta_note: Option<String>,
    pub event_convention: String,
    pub greek_method: String,
}
struct Solution {
    times: Vec<i64>,
    nodes: Vec<f64>,
    values: Vec<f64>,
    diagnostics: ExerciseDiagnostics,
}
fn linear(xs: &[f64], ys: &[f64], s: f64) -> f64 {
    if s <= xs[0] {
        return ys[0];
    }
    if s >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let j = xs.partition_point(|x| *x < s).max(1);
    let z = (s - xs[j - 1]) / (xs[j] - xs[j - 1]);
    ys[j - 1] * (1. - z) + ys[j] * z
}
fn derivatives(xs: &[f64], ys: &[f64], s: f64) -> Result<(f64, f64), ComputeError> {
    if s < xs[1] || s > xs[xs.len() - 2] {
        return Err(invalid(
            "spot lies outside interior Greek stencil; increase spatial resolution/domain",
        ));
    }
    let j = xs
        .partition_point(|x| *x <= s)
        .saturating_sub(1)
        .clamp(1, xs.len() - 3);
    let at = |i: usize| {
        let hm = xs[i] - xs[i - 1];
        let hp = xs[i + 1] - xs[i];
        let d = -hp * ys[i - 1] / (hm * (hm + hp))
            + (hp - hm) * ys[i] / (hm * hp)
            + hm * ys[i + 1] / (hp * (hm + hp));
        let g = 2. * ys[i - 1] / (hm * (hm + hp)) - 2. * ys[i] / (hm * hp)
            + 2. * ys[i + 1] / (hp * (hm + hp));
        (d, g)
    };
    let (d0, g0) = at(j);
    let (d1, g1) = at(j + 1);
    let z = (s - xs[j]) / (xs[j + 1] - xs[j]);
    Ok((d0 * (1. - z) + d1 * z, g0 * (1. - z) + g1 * z))
}

fn time_grid(o: &ExerciseOption, c: &ExerciseGrid) -> Vec<i64> {
    let a = o.valuation_time.0;
    let b = o.exercise.expiry().0;
    let span = b as i128 - a as i128;
    let mut ts = (0..=c.time_steps)
        .map(|i| (a as i128 + span * i as i128 / c.time_steps as i128) as i64)
        .collect::<Vec<_>>();
    ts.extend(o.cash_dividends.iter().map(|d| d.ex_date.0));
    match &o.exercise {
        ExerciseSchedule::American { first_exercise, .. } => {
            if first_exercise.0 > a {
                ts.push(first_exercise.0);
            }
        }
        ExerciseSchedule::Bermudan { opportunities } => {
            ts.extend(opportunities.iter().filter(|t| t.0 >= a).map(|t| t.0))
        }
        _ => {}
    }
    ts.sort_unstable();
    ts.dedup();
    ts
}
fn boundary(o: &ExerciseOption, t: i64, smax: f64, ts: &[i64]) -> (f64, f64) {
    let mut lo = 0f64;
    let mut hi = 0f64;
    let candidates = std::iter::once(t).chain(ts.iter().copied().filter(|d| *d > t));
    for u in candidates {
        if !o.exercise_at(u) {
            continue;
        }
        let tau = years(t, u);
        let disc = (-o.rate * tau).exp();
        let stock = smax * (-o.dividend_yield * tau).exp()
            - o.cash_dividends
                .iter()
                .filter(|d| d.ex_date.0 > t && d.ex_date.0 < u)
                .map(|d| {
                    d.amount
                        * (-o.rate * years(t, d.ex_date.0)
                            - o.dividend_yield * years(d.ex_date.0, u))
                        .exp()
                })
                .sum::<f64>();
        lo = lo.max(o.strike * disc);
        hi = hi.max((stock - o.strike * disc).max(0.));
    }
    match o.side {
        OptionSide::Call => (0., hi),
        OptionSide::Put => (lo, 0.),
    }
}
fn tridiagonal(a: &[f64], b: &[f64], c: &[f64], rhs: &[f64]) -> Result<Vec<f64>, ComputeError> {
    let n = b.len();
    let mut d = rhs.to_vec();
    let mut z = c.to_vec();
    if !b[0].is_finite() || b[0] <= 0. {
        return Err(execution("invalid finite-difference pivot"));
    }
    z[0] /= b[0];
    d[0] /= b[0];
    for i in 1..n {
        let pivot = b[i] - a[i] * z[i - 1];
        if !pivot.is_finite() || pivot <= 0. {
            return Err(execution("nonpositive finite-difference pivot"));
        }
        z[i] /= pivot;
        d[i] = (d[i] - a[i] * d[i - 1]) / pivot;
    }
    for i in (0..n - 1).rev() {
        d[i] -= z[i] * d[i + 1];
    }
    if d.iter().any(|x| !x.is_finite()) {
        return Err(execution("nonfinite finite-difference solution"));
    }
    Ok(d)
}
struct Operator {
    lower: Vec<f64>,
    diag: Vec<f64>,
    upper: Vec<f64>,
}
#[allow(clippy::too_many_arguments)] // Internal rollback carries one mesh/operator and its dated interval.
fn step(
    o: &ExerciseOption,
    c: &ExerciseGrid,
    xs: &[f64],
    op: &Operator,
    old: &[f64],
    lo: i64,
    hi: i64,
    theta: f64,
    ts: &[i64],
    stats: &mut ExerciseDiagnostics,
) -> Result<Vec<f64>, ComputeError> {
    let n = xs.len();
    let dt = years(lo, hi);
    if dt <= 0. {
        return Ok(old.to_vec());
    }
    let mut a = vec![0.; n];
    let mut b = vec![1.; n];
    let mut up = vec![0.; n];
    let mut rhs = vec![0.; n];
    let (left, right) = boundary(o, lo, xs[n - 1], ts);
    rhs[0] = left;
    rhs[n - 1] = right;
    for i in 1..n - 1 {
        a[i] = -theta * dt * op.lower[i];
        b[i] -= theta * dt * op.diag[i];
        up[i] = -theta * dt * op.upper[i];
        rhs[i] = old[i]
            + (1. - theta)
                * dt
                * (op.lower[i] * old[i - 1] + op.diag[i] * old[i] + op.upper[i] * old[i + 1]);
    }
    stats.rollback_substeps += 1;
    if !o.american_at(lo) {
        return tridiagonal(&a, &b, &up, &rhs);
    }
    let floor = xs.iter().map(|s| o.payoff(*s)).collect::<Vec<_>>();
    // Seed the obstacle iteration with the full continuation solve so diffusion
    // reaches the OTM tail immediately, instead of propagating one node/pass.
    let mut v = tridiagonal(&a, &b, &up, &rhs)?;
    for i in 1..n - 1 {
        v[i] = v[i].max(floor[i]);
    }
    let scale = 1. + xs[n - 1] + o.strike;
    let mut mask = vec![false; n];
    let mut last_residual = f64::INFINITY;
    for _ in 0..c.max_policy_iterations {
        let mut aa = a.clone();
        let mut bb = b.clone();
        let mut cc = up.clone();
        let mut rr = rhs.clone();
        for i in 1..n - 1 {
            let residual = (a[i] * v[i - 1] + b[i] * v[i] + up[i] * v[i + 1] - rhs[i]) / b[i];
            mask[i] = v[i] - floor[i] <= residual;
            if mask[i] {
                aa[i] = 0.;
                bb[i] = 1.;
                cc[i] = 0.;
                rr[i] = floor[i];
            }
        }
        v = tridiagonal(&aa, &bb, &cc, &rr)?;
        stats.policy_iterations += 1;
        let mut violation = 0f64;
        for i in 1..n - 1 {
            let dual = (a[i] * v[i - 1] + b[i] * v[i] + up[i] * v[i + 1] - rhs[i]) / b[i];
            let primal = v[i] - floor[i];
            violation = violation
                .max((-dual).max(0.))
                .max((-primal).max(0.))
                .max(dual.min(primal).abs());
        }
        last_residual = violation / scale;
        if last_residual <= c.complementarity_tolerance {
            stats.max_scaled_complementarity_residual = stats
                .max_scaled_complementarity_residual
                .max(violation / scale);
            return Ok(v);
        }
    }
    Err(execution(format!(
        "American complementarity solver did not converge: residual {last_residual:e}, timestamp {lo}, volatility {}; refine grid or solver settings",
        o.volatility
    )))
}
fn solve(o: &ExerciseOption, c: &ExerciseGrid) -> Result<Solution, ComputeError> {
    solve_on_times(o, c, None)
}
fn solve_on_times(
    o: &ExerciseOption,
    c: &ExerciseGrid,
    retained: Option<&[i64]>,
) -> Result<Solution, ComputeError> {
    o.validate()?;
    c.validate()?;
    let n = c.space_steps;
    let scale = o.spot.max(o.strike);
    let horizon = years(o.valuation_time.0, o.exercise.expiry().0);
    // At the upper Dirichlet boundary the terminal put tail should be small.
    // Enlarge before meshing, while keeping a finite explicit supported domain.
    let tail_exponent = 4. * o.volatility * horizon.sqrt()
        - (o.rate - o.dividend_yield - 0.5 * o.volatility.powi(2)) * horizon;
    let cash_reserve = o.cash_dividends.iter().map(|d| d.amount).sum::<f64>()
        * ((o.rate - o.dividend_yield).abs() * horizon).exp();
    let required = o.strike * tail_exponent.exp() + cash_reserve;
    if !required.is_finite() || required > scale * 1000. {
        return Err(ComputeError::Unsupported("volatility/carry/dividend horizon requires an outer domain beyond the supported 1000x scale cap".into()));
    }
    let smax = (scale * c.s_max_multiple).max(required);
    let alpha = o.strike * 0.1;
    let left = (-o.strike / alpha).asinh();
    let right = ((smax - o.strike) / alpha).asinh();
    let mut xs = (0..=n)
        .map(|i| o.strike + alpha * (left + (right - left) * i as f64 / n as f64).sinh())
        .collect::<Vec<_>>();
    xs[0] = 0.;
    xs[n] = smax;
    // Exact strike node improves nonsmooth payoff placement without losing ordering.
    let k = xs.partition_point(|s| *s < o.strike).clamp(1, n - 1);
    let k = if (xs[k] - o.strike).abs() < (xs[k - 1] - o.strike).abs() {
        k
    } else {
        k - 1
    };
    xs[k] = o.strike;
    if xs.windows(2).any(|w| w[1] <= w[0]) {
        return Err(execution("degenerate spatial mesh"));
    }
    let ts = if let Some(retained) = retained {
        let mut ts = vec![o.valuation_time.0];
        ts.extend(retained.iter().copied().filter(|t| *t > o.valuation_time.0));
        ts
    } else {
        time_grid(o, c)
    };
    if n.checked_mul(ts.len()).is_none_or(|n| n > 12_000_000) {
        return Err(invalid("event-expanded grid exceeds solve cell limit"));
    }
    let mut stats = ExerciseDiagnostics {
        space_nodes: n + 1,
        time_nodes: ts.len(),
        s_max: smax,
        spot_grid_spacing: {
            let j = xs
                .partition_point(|s| *s <= o.spot)
                .saturating_sub(1)
                .min(n - 1);
            xs[j + 1] - xs[j]
        },
        spot_diffusion_width: o.spot
            * o.volatility
            * years(o.valuation_time.0, o.exercise.expiry().0).sqrt(),
        event_dates: {
            let mut events = o
                .cash_dividends
                .iter()
                .map(|d| d.ex_date)
                .collect::<Vec<_>>();
            match &o.exercise {
                ExerciseSchedule::European { expiry } => events.push(*expiry),
                ExerciseSchedule::American {
                    first_exercise,
                    expiry,
                } => {
                    if first_exercise.0 >= o.valuation_time.0 {
                        events.push(*first_exercise);
                    }
                    events.push(*expiry);
                }
                ExerciseSchedule::Bermudan { opportunities } => events.extend(
                    opportunities
                        .iter()
                        .filter(|d| d.0 >= o.valuation_time.0)
                        .copied(),
                ),
            }
            events.sort_unstable();
            events.dedup();
            events
        },
        ..Default::default()
    };
    let mut op = Operator {
        lower: vec![0.; n + 1],
        diag: vec![0.; n + 1],
        upper: vec![0.; n + 1],
    };
    for i in 1..n {
        let hm = xs[i] - xs[i - 1];
        let hp = xs[i + 1] - xs[i];
        let diff = o.volatility.powi(2) * xs[i].powi(2);
        let drift = (o.rate - o.dividend_yield) * xs[i];
        let mut a = (diff - drift * hp) / (hm * (hm + hp));
        let mut u = (diff + drift * hm) / (hp * (hm + hp));
        if a < 0. || u < 0. {
            a = diff / (hm * (hm + hp)) + (-drift).max(0.) / hm;
            u = diff / (hp * (hm + hp)) + drift.max(0.) / hp;
            stats.upwind_nodes += 1;
        }
        op.lower[i] = a;
        op.upper[i] = u;
        op.diag[i] = -a - u - o.rate;
    }
    let mut v = xs.iter().map(|s| o.payoff(*s)).collect::<Vec<_>>();
    let mut damping = c.rannacher_intervals;
    for j in (0..ts.len() - 1).rev() {
        let lo = ts[j];
        let hi = ts[j + 1];
        if damping > 0 && hi - lo >= 2 {
            let mid = lo + (hi - lo) / 2;
            v = step(o, c, &xs, &op, &v, mid, hi, 1., &ts, &mut stats)?;
            v = step(o, c, &xs, &op, &v, lo, mid, 1., &ts, &mut stats)?;
            damping -= 1;
        } else {
            v = step(o, c, &xs, &op, &v, lo, hi, 0.5, &ts, &mut stats)?;
        }
        let mut event = false;
        if let Some(d) = o.cash_dividends.iter().find(|d| d.ex_date.0 == lo) {
            v = xs
                .iter()
                .map(|s| linear(&xs, &v, (s - d.amount).max(0.)))
                .collect();
            event = true;
        }
        if o.exercise_at(lo) {
            for (s, value) in xs.iter().zip(&mut v) {
                *value = value.max(o.payoff(*s));
            }
            if !o.american_at(lo) {
                event = true;
            }
        }
        if event {
            damping = c.rannacher_intervals;
        }
        if v.iter()
            .any(|x| !x.is_finite() || *x < -1e-8 * (1. + o.strike))
        {
            return Err(execution(
                "nonfinite or negative grid value; numerical refinement required",
            ));
        }
    }
    Ok(Solution {
        times: ts,
        nodes: xs,
        values: v,
        diagnostics: stats,
    })
}
fn result(o: &ExerciseOption, s: &Solution) -> Result<ExercisePrice, ComputeError> {
    let p = linear(&s.nodes, &s.values, o.spot);
    if !p.is_finite() || p < 0. {
        return Err(execution("invalid interpolated option price"));
    }
    let intrinsic = o.payoff(o.spot);
    let now = o.exercise_at(o.valuation_time.0);
    Ok(ExercisePrice {
        row_id: o.row_id.clone(),
        contract_id: o.contract_id.clone(),
        price: p,
        volatility: o.volatility,
        intrinsic_value: intrinsic,
        exercise_permitted_now: now,
        at_exercise_value: now
            && intrinsic > 0.
            && (p - intrinsic).abs() <= 64. * f64::EPSILON * (1. + p.abs() + o.spot + o.strike),
        diagnostics: s.diagnostics.clone(),
    })
}
pub fn price(o: &ExerciseOption, c: &ExerciseGrid) -> Result<ExercisePrice, ComputeError> {
    result(o, &solve(o, c)?)
}
pub fn price_and_greeks(
    o: &ExerciseOption,
    c: &ExerciseGrid,
) -> Result<ExerciseRisk, ComputeError> {
    let s = solve(o, c)?;
    let base = result(o, &s)?;
    if !base.at_exercise_value
        && s.diagnostics.spot_diffusion_width < 2. * s.diagnostics.spot_grid_spacing
    {
        return Err(invalid(
            "Greek grid has fewer than two cells per spot diffusion width; increase space resolution",
        ));
    }
    if o.volatility <= c.volatility_bump {
        return Err(invalid(
            "central volatility bump reaches zero; choose smaller bump",
        ));
    }
    let (mut delta, mut gamma) = derivatives(&s.nodes, &s.values, o.spot)?;
    if base.at_exercise_value && base.intrinsic_value > 0. {
        delta = match o.side {
            OptionSide::Call => 1.,
            OptionSide::Put => -1.,
        };
        gamma = 0.;
    }
    let mut plus = o.clone();
    let mut minus = o.clone();
    plus.volatility += c.volatility_bump;
    minus.volatility -= c.volatility_bump;
    let vega = (price(&plus, c)?.price - price(&minus, c)?.price) / (2. * c.volatility_bump);
    plus = o.clone();
    minus = o.clone();
    plus.rate += c.rate_bump;
    minus.rate -= c.rate_bump;
    let rho = (price(&plus, c)?.price - price(&minus, c)?.price) / (2. * c.rate_bump);
    let now = o.valuation_time.0;
    let mut next = o.exercise.expiry().0;
    for d in &o.cash_dividends {
        next = next.min(d.ex_date.0);
    }
    match &o.exercise {
        ExerciseSchedule::American { first_exercise, .. } if first_exercise.0 > now => {
            next = next.min(first_exercise.0)
        }
        ExerciseSchedule::Bermudan { opportunities } => {
            if let Some(t) = opportunities.iter().find(|t| t.0 > now) {
                next = next.min(t.0);
            }
        }
        _ => {}
    }
    let gap = (next as i128 - now as i128) as u64;
    let h = c.theta_bump_ns.min(gap / 2);
    let theta_unavailable = h < 1_000_000_000
        || matches!(&o.exercise, ExerciseSchedule::Bermudan { .. }) && o.exercise_at(now);
    let theta = if theta_unavailable {
        None
    } else {
        plus = o.clone();
        plus.valuation_time = TimestampNs(
            now.checked_add(h as i64)
                .ok_or_else(|| invalid("theta timestamp overflow"))?,
        );
        let shifted = solve_on_times(&plus, c, Some(&s.times))?;
        Some((result(&plus, &shifted)?.price - base.price) / (h as f64 / YEAR_NS))
    };
    let numbers = [delta, gamma, vega, rho, theta.unwrap_or(0.)];
    if numbers.iter().any(|v| !v.is_finite()) {
        return Err(execution("nonfinite exercise Greeks"));
    }
    Ok(ExerciseRisk{valuation:base,delta,gamma,vega_per_unit_volatility:vega,vega_per_vol_point:vega*0.01,rho_per_unit_rate:rho,rho_per_bp:rho*0.0001,theta_per_year:theta,theta_per_day:theta.map(|x|x/365.),theta_bump_ns:theta.map(|_|h),theta_note:theta_unavailable.then(||"calendar theta unavailable on a Bermudan exercise instant or without a safe one-second bump before the next event".into()),event_convention:"Actual365Fixed; fixed dates; exercise before coincident cash-dividend jump; valuation after already-paid cash; calendar-forward theta excludes the next event".into(),greek_method:"delta/gamma: interpolated nodal nonuniform quadratic FD stencils; price: linear mesh interpolation; vega/rho: central complete repricing; theta: one-sided calendar-forward repricing on retained future time nodes".into()})
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExerciseIvConfig {
    pub min_volatility: f64,
    pub max_volatility: f64,
    pub price_tolerance: f64,
    pub volatility_tolerance: f64,
    pub min_vega: f64,
    pub max_iterations: usize,
}
impl Default for ExerciseIvConfig {
    fn default() -> Self {
        Self {
            min_volatility: 0.005,
            max_volatility: 1.,
            price_tolerance: 1e-5,
            volatility_tolerance: 1e-6,
            min_vega: 1e-4,
            max_iterations: 40,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExerciseIvState {
    Converged,
    WeaklyIdentified,
    OutsideBracket,
    IterationLimit,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseIvResult {
    pub state: ExerciseIvState,
    pub volatility: Option<f64>,
    pub model_price: Option<f64>,
    pub price_residual: Option<f64>,
    pub volatility_bracket: [f64; 2],
    pub iterations: usize,
    pub reason: String,
}
pub fn implied_volatility(
    o: &ExerciseOption,
    target: f64,
    c: &ExerciseGrid,
    iv: &ExerciseIvConfig,
) -> Result<ExerciseIvResult, ComputeError> {
    o.validate()?;
    c.validate()?;
    if !target.is_finite()
        || target < 0.
        || !iv.min_volatility.is_finite()
        || !iv.max_volatility.is_finite()
        || !(0.002..3.).contains(&iv.min_volatility)
        || iv.max_volatility > 3.
        || iv.max_volatility <= iv.min_volatility
        || !iv.price_tolerance.is_finite()
        || !(1e-10..=1.).contains(&iv.price_tolerance)
        || !iv.volatility_tolerance.is_finite()
        || !(1e-9..=0.1).contains(&iv.volatility_tolerance)
        || !iv.min_vega.is_finite()
        || iv.min_vega <= 0.
        || !(2..=80).contains(&iv.max_iterations)
    {
        return Err(invalid("invalid exercise IV target/configuration"));
    }
    let intrinsic = o.payoff(o.spot);
    let make =
        |state, volatility, model_price, bracket, iterations, reason: &str| ExerciseIvResult {
            state,
            volatility,
            model_price,
            price_residual: model_price.map(|p| p - target),
            volatility_bracket: bracket,
            iterations,
            reason: reason.into(),
        };
    if o.exercise_at(o.valuation_time.0) && target < intrinsic - iv.price_tolerance {
        return Ok(make(
            ExerciseIvState::OutsideBracket,
            None,
            None,
            [iv.min_volatility, iv.max_volatility],
            0,
            "price is below immediately exercisable payoff",
        ));
    }
    let mut low = iv.min_volatility;
    let mut high = iv.max_volatility;
    let mut trial = o.clone();
    trial.volatility = low;
    let pl = price(&trial, c)?.price;
    trial.volatility = high;
    let ph = price(&trial, c)?.price;
    if ph < pl - iv.price_tolerance {
        return Err(execution(
            "IV model/grid response is nonmonotone in the configured bracket; use a different bracket or model",
        ));
    }
    if target < pl - iv.price_tolerance || target > ph + iv.price_tolerance {
        return Ok(make(
            ExerciseIvState::OutsideBracket,
            None,
            None,
            [low, high],
            0,
            "quote lies outside configured model volatility bracket",
        ));
    }
    if o.exercise_at(o.valuation_time.0) && (target - intrinsic).abs() <= iv.price_tolerance {
        return Ok(make(
            ExerciseIvState::WeaklyIdentified,
            None,
            Some(intrinsic),
            [iv.min_volatility, iv.max_volatility],
            0,
            "exercise-value quote does not identify a unique positive volatility",
        ));
    }
    if ph - pl <= iv.price_tolerance {
        return Ok(make(
            ExerciseIvState::WeaklyIdentified,
            None,
            None,
            [low, high],
            0,
            "model price is flat across configured volatility bracket",
        ));
    }
    for iteration in 1..=iv.max_iterations {
        let mid = (low + high) * 0.5;
        trial.volatility = mid;
        let p = price(&trial, c)?.price;
        if (p - target).abs() <= iv.price_tolerance && high - low <= iv.volatility_tolerance * 4. {
            let h = (mid * 0.01).clamp(1e-5, 0.001).min((mid - 0.002) * 0.5);
            if h <= 0. || !h.is_finite() {
                return Ok(make(
                    ExerciseIvState::WeaklyIdentified,
                    Some(mid),
                    Some(p),
                    [low, high],
                    iteration,
                    "volatility too close to model limit for a stable conditioning bump",
                ));
            }
            let mut a = trial.clone();
            let mut b = trial.clone();
            a.volatility += h;
            b.volatility -= h;
            let vega = (price(&a, c)?.price - price(&b, c)?.price) / (2. * h);
            if !vega.is_finite() {
                return Err(execution("nonfinite IV conditioning sensitivity"));
            }
            let state = if vega < iv.min_vega {
                ExerciseIvState::WeaklyIdentified
            } else {
                ExerciseIvState::Converged
            };
            return Ok(make(
                state,
                Some(mid),
                Some(p),
                [low, high],
                iteration,
                if vega < iv.min_vega {
                    "small model vega"
                } else {
                    "price and volatility bracket tolerances satisfied"
                },
            ));
        }
        if p < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(make(
        ExerciseIvState::IterationLimit,
        None,
        None,
        [low, high],
        iv.max_iterations,
        "configured IV iteration limit",
    ))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExerciseBatchRow {
    pub option: ExerciseOption,
    #[serde(default)]
    pub market_price: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExerciseBatchRequest {
    pub schema: String,
    #[serde(default)]
    pub grid: ExerciseGrid,
    #[serde(default)]
    pub iv: ExerciseIvConfig,
    pub rows: Vec<ExerciseBatchRow>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseBatchRowResult {
    pub row_id: String,
    pub contract_id: String,
    pub risk: Option<ExerciseRisk>,
    pub implied_volatility: Option<ExerciseIvResult>,
    pub error: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseBatchResult {
    pub schema: String,
    pub execution_policy: String,
    pub rows: Vec<ExerciseBatchRowResult>,
}
/// Batch keeps source order and per-row failures. If a market price is supplied,
/// Greeks are returned only for a positive-vega converged IV root in the bracket.
pub fn price_batch(request: &ExerciseBatchRequest) -> Result<ExerciseBatchResult, ComputeError> {
    request.grid.validate()?;
    if request.iv.max_iterations > 80
        || request.schema != "volt.compute.exercise_pricing.v1"
        || request.rows.is_empty()
        || request.rows.len() > MAX_ROWS
    {
        return Err(invalid("unsupported exercise batch schema/row count"));
    }
    let mut counts = BTreeMap::new();
    for row in &request.rows {
        *counts.entry(&row.option.row_id).or_insert(0usize) += 1;
    }
    let mut identities = BTreeMap::new();
    let mut conflicts = HashSet::new();
    for row in &request.rows {
        let key =
            serde_json::to_string(&(row.option.side, row.option.strike, &row.option.exercise))
                .map_err(|e| invalid(e.to_string()))?;
        if identities
            .insert(&row.option.contract_id, key.clone())
            .is_some_and(|v| v != key)
        {
            conflicts.insert(&row.option.contract_id);
        }
    }
    let solves = request
        .rows
        .iter()
        .map(|r| {
            if r.market_price.is_some() {
                request.iv.max_iterations.saturating_add(10)
            } else {
                6
            }
        })
        .try_fold(0usize, |a, b| a.checked_add(b))
        .ok_or_else(|| invalid("batch work overflow"))?;
    if request
        .grid
        .space_steps
        .checked_mul(request.grid.time_steps + MAX_ADDED_EVENTS)
        .and_then(|n| n.checked_mul(solves))
        .is_none_or(|n| n > 500_000_000)
    {
        return Err(ComputeError::BudgetExceeded(
            "exercise batch solve-cell admission exceeds 500 million; split request or lower grid"
                .into(),
        ));
    }
    let rows = request
        .rows
        .iter()
        .map(|r| {
            let o = &r.option;
            let mut out = ExerciseBatchRowResult {
                row_id: o.row_id.clone(),
                contract_id: o.contract_id.clone(),
                risk: None,
                implied_volatility: None,
                error: None,
            };
            let work =
                || -> Result<(Option<ExerciseRisk>, Option<ExerciseIvResult>), ComputeError> {
                    if counts[&o.row_id] > 1 || conflicts.contains(&o.contract_id) {
                        return Err(invalid(
                            "duplicate row or conflicting exact contract identity",
                        ));
                    }
                    if let Some(target) = r.market_price {
                        let v = implied_volatility(o, target, &request.grid, &request.iv)?;
                        let risk = if v.state == ExerciseIvState::Converged {
                            let mut priced = o.clone();
                            priced.volatility = v
                                .volatility
                                .ok_or_else(|| execution("missing converged IV"))?;
                            Some(price_and_greeks(&priced, &request.grid)?)
                        } else {
                            None
                        };
                        Ok((risk, Some(v)))
                    } else {
                        Ok((Some(price_and_greeks(o, &request.grid)?), None))
                    }
                };
            match work() {
                Ok((risk, iv)) => {
                    out.risk = risk;
                    out.implied_volatility = iv;
                }
                Err(e) => out.error = Some(e.to_string()),
            }
            out
        })
        .collect();
    Ok(ExerciseBatchResult {
        schema: "volt.compute.exercise_pricing_result.v1".into(),
        execution_policy:
            "cpu_f64_finite_difference; flat r/q/vol; Actual365Fixed; per underlying unit".into(),
        rows,
    })
}
