use std::f64::consts::PI;

use serde::Serialize;
use volt::{
    CanonicalPortfolio, ExerciseSchedule, OptionContract, OptionRight, Position, PriceMicros,
    Settlement, TerminalPayoffProjection, TimestampNs,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Serialize)]
pub struct RiskSnapshot {
    pub step: u64,
    pub spot: f64,
    pub last_return: f64,
    pub volatility: f64,
    pub days_to_expiry: f64,
    pub portfolio_hash: String,
    pub payoff_knots: usize,
    pub metrics: PortfolioMetrics,
    pub bounds: PayoffBounds,
    pub legs: Vec<LegSnapshot>,
    pub expiry_curve: Vec<ExpiryPoint>,
    pub shock_grid: Vec<ShockCell>,
    pub spot_history: Vec<f64>,
    pub pnl_history: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PortfolioMetrics {
    pub mark: f64,
    pub pnl: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PayoffBounds {
    pub max_profit: f64,
    pub max_loss: f64,
    pub lower_break_even: f64,
    pub upper_break_even: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LegSnapshot {
    pub contract_id: String,
    pub right: &'static str,
    pub quantity: i64,
    pub strike: f64,
    pub mark: f64,
    pub volatility: f64,
    pub delta: f64,
    pub pnl: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ExpiryPoint {
    pub shock: f64,
    pub spot: f64,
    pub pnl: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ShockCell {
    pub spot_shock: f64,
    pub volatility_shock: f64,
    pub pnl: f64,
}

#[derive(Debug)]
pub struct RiskSimulation {
    market: MarketModel,
    legs: Vec<StrategyLeg>,
    projection: TerminalPayoffProjection,
    portfolio_hash: String,
    entry_cash: f64,
    spot_history: Vec<f64>,
    pnl_history: Vec<f64>,
}

impl RiskSimulation {
    pub fn new(seed: u64) -> Result<Self> {
        let market = MarketModel::new(seed);
        let mut legs = build_iron_condor(market.spot, market.volatility)?;
        let positions = legs
            .iter()
            .map(|leg| Position::new(leg.contract.clone(), leg.quantity))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let portfolio = CanonicalPortfolio::from_positions(positions)?;
        let projection = portfolio.terminal_payoff_projection()?;
        let portfolio_hash = portfolio.semantic_hash()?;
        let entry_view = strategy_value(&legs, market.spot, market.volatility, 45.0 / 365.0);
        let entry_cash = -entry_view.mark;
        for (leg, mark) in legs.iter_mut().zip(entry_view.marks) {
            leg.entry_mark = mark.price;
        }
        let initial_spot = market.spot;

        Ok(Self {
            market,
            legs,
            projection,
            portfolio_hash,
            entry_cash,
            spot_history: vec![initial_spot],
            pnl_history: vec![0.0],
        })
    }

    pub fn next_snapshot(&mut self) -> Result<RiskSnapshot> {
        self.market.step();
        let days_to_expiry = (45.0 - self.market.steps as f64 / 13.0).max(0.02);
        let time_to_expiry = days_to_expiry / 365.0;
        let mut view = strategy_value(
            &self.legs,
            self.market.spot,
            self.market.volatility,
            time_to_expiry,
        );
        view.pnl += self.entry_cash;

        self.spot_history.push(self.market.spot);
        self.pnl_history.push(view.pnl);
        trim_history(&mut self.spot_history, 90);
        trim_history(&mut self.pnl_history, 90);

        let bounds = PayoffBounds {
            max_profit: self.entry_cash,
            max_loss: self.entry_cash - 500.0,
            lower_break_even: 95.0 - self.entry_cash / 100.0,
            upper_break_even: 105.0 + self.entry_cash / 100.0,
        };
        let legs = self
            .legs
            .iter()
            .zip(&view.marks)
            .map(|(leg, mark)| {
                let strike = leg.contract.strike.0 as f64 / 1_000_000.0;
                let units = leg.quantity as f64 * leg.contract.multiplier as f64;
                LegSnapshot {
                    contract_id: leg.contract.contract_id.0.clone(),
                    right: match leg.contract.right {
                        OptionRight::Call => "CALL",
                        OptionRight::Put => "PUT",
                    },
                    quantity: leg.quantity,
                    strike,
                    mark: mark.price,
                    volatility: leg_volatility(self.market.volatility, strike),
                    delta: mark.delta * units,
                    pnl: (mark.price - leg.entry_mark) * units,
                }
            })
            .collect();
        let expiry_curve = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
            .into_iter()
            .map(|shock| {
                let spot = self.market.spot * (1.0 + shock);
                let pnl = self
                    .projection
                    .evaluate(PriceMicros((spot * 1_000_000.0).round() as i64))?
                    as f64
                    / 1_000_000.0
                    + self.entry_cash;
                Ok(ExpiryPoint { shock, spot, pnl })
            })
            .collect::<Result<Vec<_>>>()?;
        let shock_grid = [-0.05, 0.0, 0.05]
            .into_iter()
            .flat_map(|spot_shock| {
                [-0.05, 0.0, 0.05]
                    .into_iter()
                    .map(move |volatility_shock| (spot_shock, volatility_shock))
            })
            .map(|(spot_shock, volatility_shock)| {
                let pnl = strategy_value(
                    &self.legs,
                    self.market.spot * (1.0 + spot_shock),
                    (self.market.volatility + volatility_shock).max(0.05),
                    time_to_expiry,
                )
                .mark
                    + self.entry_cash;
                ShockCell {
                    spot_shock,
                    volatility_shock,
                    pnl,
                }
            })
            .collect();

        Ok(RiskSnapshot {
            step: self.market.steps,
            spot: self.market.spot,
            last_return: self.market.last_return,
            volatility: self.market.volatility,
            days_to_expiry,
            portfolio_hash: self.portfolio_hash.clone(),
            payoff_knots: self.projection.call_knots.len(),
            metrics: PortfolioMetrics {
                mark: view.mark,
                pnl: view.pnl,
                delta: view.delta,
                gamma: view.gamma,
                vega: view.vega,
                theta: view.theta,
            },
            bounds,
            legs,
            expiry_curve,
            shock_grid,
            spot_history: self.spot_history.clone(),
            pnl_history: self.pnl_history.clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct StrategyLeg {
    contract: OptionContract,
    quantity: i64,
    entry_mark: f64,
}

#[derive(Debug, Clone, Copy)]
struct Greeks {
    price: f64,
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
}

#[derive(Debug)]
struct StrategyView {
    marks: Vec<Greeks>,
    mark: f64,
    pnl: f64,
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
}

#[derive(Debug)]
struct MarketModel {
    rng: XorShift64,
    steps: u64,
    spot: f64,
    volatility: f64,
    mean_log_volatility: f64,
    last_return: f64,
}

fn build_iron_condor(spot: f64, volatility: f64) -> Result<Vec<StrategyLeg>> {
    let expiry = TimestampNs(1_900_000_000_000_000_000);
    let specifications = [
        (90.0, OptionRight::Put, 1_i64),
        (95.0, OptionRight::Put, -1_i64),
        (105.0, OptionRight::Call, -1_i64),
        (110.0, OptionRight::Call, 1_i64),
    ];
    specifications
        .into_iter()
        .map(|(strike, right, quantity)| {
            let side = match right {
                OptionRight::Call => "C",
                OptionRight::Put => "P",
            };
            let contract = OptionContract::new(
                format!("SIMX-{side}-{strike:.0}"),
                "SIMX",
                "USD",
                right,
                PriceMicros((strike * 1_000_000.0) as i64),
                100,
                ExerciseSchedule::European { expiry },
                Settlement::Cash,
            )?;
            let entry_mark = black_scholes(
                spot,
                strike,
                45.0 / 365.0,
                leg_volatility(volatility, strike),
                right,
            )
            .price;
            Ok(StrategyLeg {
                contract,
                quantity,
                entry_mark,
            })
        })
        .collect()
}

fn strategy_value(
    legs: &[StrategyLeg],
    spot: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> StrategyView {
    let marks = legs
        .iter()
        .map(|leg| {
            black_scholes(
                spot,
                leg.contract.strike.0 as f64 / 1_000_000.0,
                time_to_expiry,
                leg_volatility(volatility, leg.contract.strike.0 as f64 / 1_000_000.0),
                leg.contract.right,
            )
        })
        .collect::<Vec<_>>();
    let mut view = StrategyView {
        marks,
        mark: 0.0,
        pnl: 0.0,
        delta: 0.0,
        gamma: 0.0,
        vega: 0.0,
        theta: 0.0,
    };
    for (leg, mark) in legs.iter().zip(&view.marks) {
        let units = leg.quantity as f64 * leg.contract.multiplier as f64;
        view.mark += units * mark.price;
        view.delta += units * mark.delta;
        view.gamma += units * mark.gamma;
        view.vega += units * mark.vega;
        view.theta += units * mark.theta;
    }
    view.pnl = view.mark;
    view
}

fn black_scholes(spot: f64, strike: f64, time: f64, volatility: f64, right: OptionRight) -> Greeks {
    let rate = 0.04;
    let time = time.max(1.0 / 365_000.0);
    let volatility = volatility.max(0.001);
    let sqrt_time = time.sqrt();
    let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility.powi(2)) * time)
        / (volatility * sqrt_time);
    let d2 = d1 - volatility * sqrt_time;
    let discount = (-rate * time).exp();
    let density = normal_pdf(d1);
    let (price, delta, theta_year) = match right {
        OptionRight::Call => (
            spot * normal_cdf(d1) - strike * discount * normal_cdf(d2),
            normal_cdf(d1),
            -(spot * density * volatility) / (2.0 * sqrt_time)
                - rate * strike * discount * normal_cdf(d2),
        ),
        OptionRight::Put => (
            strike * discount * normal_cdf(-d2) - spot * normal_cdf(-d1),
            normal_cdf(d1) - 1.0,
            -(spot * density * volatility) / (2.0 * sqrt_time)
                + rate * strike * discount * normal_cdf(-d2),
        ),
    };
    Greeks {
        price: price.max(0.0),
        delta,
        gamma: density / (spot * volatility * sqrt_time),
        vega: spot * density * sqrt_time / 100.0,
        theta: theta_year / 365.0,
    }
}

fn leg_volatility(atm_volatility: f64, strike: f64) -> f64 {
    (atm_volatility + (100.0 - strike) * 0.0012).clamp(0.05, 1.50)
}

fn normal_pdf(value: f64) -> f64 {
    (-0.5 * value * value).exp() / (2.0 * PI).sqrt()
}

fn normal_cdf(value: f64) -> f64 {
    let absolute = value.abs();
    let k = 1.0 / (1.0 + 0.231_641_9 * absolute);
    let polynomial = k
        * (0.319_381_530
            + k * (-0.356_563_782
                + k * (1.781_477_937 + k * (-1.821_255_978 + k * 1.330_274_429))));
    let upper = 1.0 - normal_pdf(absolute) * polynomial;
    if value >= 0.0 {
        upper
    } else {
        1.0 - upper
    }
}

impl MarketModel {
    fn new(seed: u64) -> Self {
        let volatility: f64 = 0.24;
        Self {
            rng: XorShift64::new(seed),
            steps: 0,
            spot: 100.0,
            volatility,
            mean_log_volatility: volatility.ln(),
            last_return: 0.0,
        }
    }

    fn step(&mut self) {
        self.steps += 1;
        let dt: f64 = 1.0 / (252.0 * 13.0);
        let spot_shock = self.rng.normal();
        let independent_vol_shock = self.rng.normal();
        let correlation = -0.55;
        let vol_shock = correlation * spot_shock
            + (1.0_f64 - correlation.powi(2)).sqrt() * independent_vol_shock;
        let mut log_volatility = self.volatility.ln();
        log_volatility +=
            7.0 * (self.mean_log_volatility - log_volatility) * dt + 0.85 * dt.sqrt() * vol_shock;
        self.volatility = log_volatility.exp().clamp(0.08, 0.80);
        let log_return =
            (0.04 - 0.5 * self.volatility.powi(2)) * dt + self.volatility * dt.sqrt() * spot_shock;
        self.last_return = log_return.exp() - 1.0;
        self.spot *= log_return.exp();
    }
}

#[derive(Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn normal(&mut self) -> f64 {
        let first = ((self.next() as f64 + 1.0) / (u64::MAX as f64 + 2.0))
            .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let second = (self.next() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        (-2.0 * first.ln()).sqrt() * (2.0 * PI * second).cos()
    }
}

fn trim_history(values: &mut Vec<f64>, length: usize) {
    if values.len() > length {
        values.remove(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_put_parity_holds_for_teaching_pricer() {
        let call = black_scholes(100.0, 100.0, 0.25, 0.20, OptionRight::Call);
        let put = black_scholes(100.0, 100.0, 0.25, 0.20, OptionRight::Put);
        let parity = 100.0 - 100.0 * (-0.04_f64 * 0.25).exp();
        assert!((call.price - put.price - parity).abs() < 1e-6);
    }

    #[test]
    fn condor_projection_is_bounded() {
        let legs = build_iron_condor(100.0, 0.24).unwrap();
        let positions = legs
            .iter()
            .map(|leg| Position::new(leg.contract.clone(), leg.quantity).unwrap())
            .collect::<Vec<_>>();
        let projection = CanonicalPortfolio::from_positions(positions)
            .unwrap()
            .terminal_payoff_projection()
            .unwrap();
        let far_down = projection.evaluate(PriceMicros(70_000_000)).unwrap();
        let center = projection.evaluate(PriceMicros(100_000_000)).unwrap();
        let far_up = projection.evaluate(PriceMicros(130_000_000)).unwrap();
        assert_eq!(far_down, far_up);
        assert!(center > far_down);
    }

    #[test]
    fn snapshots_are_repeatable_for_a_seed() {
        let mut left = RiskSimulation::new(28_695).unwrap();
        let mut right = RiskSimulation::new(28_695).unwrap();
        let left = serde_json::to_string(&left.next_snapshot().unwrap()).unwrap();
        let right = serde_json::to_string(&right.next_snapshot().unwrap()).unwrap();
        assert_eq!(left, right);
    }
}
