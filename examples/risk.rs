//! `risk` is a live, simulation-driven options strategy monitor.
//!
//! Volt supplies exact contracts, canonical portfolio identity, and terminal
//! payoff algebra. This example adds an explicitly non-authoritative
//! Black--Scholes teaching model and a correlated stochastic-volatility path.

use std::f64::consts::PI;
use std::io::{self, IsTerminal, Write};
use std::thread;
use std::time::Duration;

use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
};
use volt::{
    CanonicalPortfolio, ExerciseSchedule, OptionContract, OptionRight, Position, PriceMicros,
    Settlement, TerminalPayoffProjection, TimestampNs,
};

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[38;5;45m";
const BLUE: &str = "\x1b[38;5;75m";
const GREEN: &str = "\x1b[38;5;84m";
const RED: &str = "\x1b[38;5;203m";
const AMBER: &str = "\x1b[38;5;221m";
const MUTED: &str = "\x1b[38;5;244m";
const WHITE: &str = "\x1b[38;5;255m";
const SPARKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
struct Config {
    interval: Duration,
    ticks: Option<usize>,
    seed: u64,
    record: bool,
    no_color: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(180),
            ticks: None,
            seed: 0x7017,
            record: false,
            no_color: false,
        }
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

fn main() -> Result<()> {
    let config = parse_args()?;
    let mut market = MarketModel::new(config.seed);
    let mut legs = build_iron_condor(market.spot, market.volatility)?;
    let positions = legs
        .iter()
        .map(|leg| Position::new(leg.contract.clone(), leg.quantity))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let portfolio = CanonicalPortfolio::from_positions(positions)?;
    let projection = portfolio.terminal_payoff_projection()?;
    let portfolio_hash = portfolio.semantic_hash()?;
    let entry_view = strategy_value(&legs, market.spot, market.volatility, 45.0 / 365.0);
    let entry_value = entry_view.mark;
    let entry_cash = -entry_value;
    for (leg, mark) in legs.iter_mut().zip(entry_view.marks) {
        leg.entry_mark = mark.price;
    }

    let stdout_is_terminal = io::stdout().is_terminal();
    let interactive = stdout_is_terminal && !config.record;
    let color = !config.no_color && (interactive || config.record);
    let mut terminal = TerminalSession::enter(interactive)?;
    let limit = config
        .ticks
        .or(config.record.then_some(32))
        .or((!interactive).then_some(1));
    let mut spot_history = vec![market.spot];
    let mut pnl_history = vec![0.0];
    let mut tick = 0_usize;

    loop {
        tick += 1;
        market.step();
        let days_to_expiry = (45.0 - tick as f64 / 13.0).max(0.02);
        let time_to_expiry = days_to_expiry / 365.0;
        let mut view = strategy_value(&legs, market.spot, market.volatility, time_to_expiry);
        view.pnl += entry_cash;
        spot_history.push(market.spot);
        pnl_history.push(view.pnl);
        trim_history(&mut spot_history, 48);
        trim_history(&mut pnl_history, 48);

        let frame = render(
            &legs,
            &projection,
            &portfolio_hash,
            &market,
            days_to_expiry,
            entry_cash,
            &view,
            &spot_history,
            &pnl_history,
            color,
        )?;
        if interactive {
            terminal.draw(&frame)?;
        } else {
            let mut stdout = io::stdout().lock();
            stdout.write_all(frame.as_bytes())?;
            if config.record {
                stdout.write_all(b"\n\x0c\n")?;
            } else if limit.is_some_and(|maximum| maximum > 1) {
                stdout.write_all(b"\n\n")?;
            }
            stdout.flush()?;
        }

        if limit.is_some_and(|maximum| tick >= maximum) {
            break;
        }
        if interactive {
            if wait_for_exit(config.interval)? {
                break;
            }
        } else if !config.interval.is_zero() {
            thread::sleep(config.interval);
        }
    }

    terminal.leave()?;
    Ok(())
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

#[allow(clippy::too_many_arguments)]
fn render(
    legs: &[StrategyLeg],
    projection: &TerminalPayoffProjection,
    portfolio_hash: &str,
    market: &MarketModel,
    days_to_expiry: f64,
    entry_cash: f64,
    view: &StrategyView,
    spot_history: &[f64],
    pnl_history: &[f64],
    color: bool,
) -> Result<String> {
    let mut out = String::new();
    let max_profit = entry_cash;
    let max_loss = entry_cash - 500.0;
    let lower_break_even = 95.0 - entry_cash / 100.0;
    let upper_break_even = 105.0 + entry_cash / 100.0;

    out.push_str(&styled(" VOLT / RISK ", &[BOLD, CYAN], color));
    out.push_str(&styled(
        "  SIMX EUROPEAN IRON CONDOR MONITOR\n",
        &[BOLD, WHITE],
        color,
    ));
    out.push_str(&styled(
        &format!(
            " path step {:>4}  spot {:>8.2}  tick {}  IV {:>6.2}%  DTE {:>5.2}\n",
            market.steps,
            market.spot,
            plain_signed_percent(market.last_return),
            market.volatility * 100.0,
            days_to_expiry
        ),
        &[MUTED],
        color,
    ));
    out.push_str(&styled(
        " model: correlated GBM + mean-reverting log volatility; Black–Scholes teaching marks\n",
        &[DIM, BLUE],
        color,
    ));
    out.push_str(&styled(
        &format!(
            " canonical {}…  payoff knots {}  cash-settled European contracts\n",
            &portfolio_hash[..12],
            projection.call_knots.len()
        ),
        &[DIM, BLUE],
        color,
    ));
    out.push_str(&styled(
        "────────────────────────────────────────────────────────────────────────────────────────────\n",
        &[MUTED],
        color,
    ));
    out.push_str(&styled(
        " LIVE RISK                  VALUE         Δ          Γ      VEGA/pt     THETA/day\n",
        &[BOLD, AMBER],
        color,
    ));
    out.push_str(&format!(
        " portfolio mark       {}   {:>+9.2}  {:>9.3}  {:>10.2}  {:>12.2}\n",
        money(view.mark, color),
        view.delta,
        view.gamma,
        view.vega,
        view.theta
    ));
    out.push_str(&format!(
        " mark-to-model P&L     {}   spot {}   pnl {}\n",
        money(view.pnl, color),
        styled(&sparkline(spot_history, 28), &[CYAN], color),
        styled(
            &sparkline(pnl_history, 28),
            &[if view.pnl >= 0.0 { GREEN } else { RED }],
            color
        )
    ));
    out.push_str(&format!(
        " expiry bounds         max {}   min {}   break-even {:.2} / {:.2}\n",
        money(max_profit, color),
        money(max_loss, color),
        lower_break_even,
        upper_break_even
    ));

    out.push_str(&styled(
        "\n LEGS                 QTY   STRIKE      MARK       IV      Δ/contract      LEG P&L\n",
        &[BOLD, AMBER],
        color,
    ));
    for (leg, mark) in legs.iter().zip(&view.marks) {
        let strike = leg.contract.strike.0 as f64 / 1_000_000.0;
        let leg_pnl =
            (mark.price - leg.entry_mark) * leg.quantity as f64 * leg.contract.multiplier as f64;
        let delta_contract = mark.delta * leg.quantity as f64 * leg.contract.multiplier as f64;
        out.push_str(&format!(
            " {:<18} {:>+4}  {:>7.2}   {:>7.3}   {:>6.2}%   {:>+12.2}   {}\n",
            leg.contract.contract_id.0,
            leg.quantity,
            strike,
            mark.price,
            leg_volatility(market.volatility, strike) * 100.0,
            delta_contract,
            money(leg_pnl, color)
        ));
    }

    out.push_str(&styled(
        "\n EXPIRY P&L              −15%       −10%        −5%       SPOT        +5%       +10%       +15%\n",
        &[BOLD, AMBER],
        color,
    ));
    out.push(' ');
    for shock in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15] {
        let terminal_spot = market.spot * (1.0 + shock);
        let payoff =
            projection.evaluate(PriceMicros((terminal_spot * 1_000_000.0).round() as i64))? as f64
                / 1_000_000.0
                + entry_cash;
        out.push_str(&money_cell(payoff, 11, color));
    }
    out.push('\n');

    out.push_str(&styled(
        "\n MARK-TO-MODEL SHOCKS       IV −5pt           IV NOW           IV +5pt\n",
        &[BOLD, AMBER],
        color,
    ));
    for spot_shock in [-0.05, 0.0, 0.05] {
        out.push_str(&format!(" spot {:+.0}%             ", spot_shock * 100.0));
        for vol_shock in [-0.05, 0.0, 0.05] {
            let shocked = strategy_value(
                legs,
                market.spot * (1.0 + spot_shock),
                (market.volatility + vol_shock).max(0.05),
                days_to_expiry / 365.0,
            )
            .mark
                + entry_cash;
            out.push_str(&money_cell(shocked, 17, color));
        }
        out.push('\n');
    }
    out.push_str(&styled(
        "\n q quit  •  deterministic simulation  •  educational marks  •  no quote or trading authority\n",
        &[DIM, MUTED],
        color,
    ));
    Ok(out)
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

fn money(value: f64, color: bool) -> String {
    styled(
        &format!("${value:>+10.2}"),
        &[BOLD, if value >= 0.0 { GREEN } else { RED }],
        color,
    )
}

fn money_cell(value: f64, width: usize, color: bool) -> String {
    styled(
        &format!("{value:>+width$.0}"),
        &[if value >= 0.0 { GREEN } else { RED }],
        color,
    )
}

fn plain_signed_percent(value: f64) -> String {
    format!("{:>+7.3}%", value * 100.0)
}

fn sparkline(values: &[f64], width: usize) -> String {
    let start = values.len().saturating_sub(width);
    let values = &values[start..];
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (maximum - minimum).max(f64::EPSILON);
    values
        .iter()
        .map(|value| {
            let bucket = (((value - minimum) / range) * (SPARKS.len() - 1) as f64).round() as usize;
            SPARKS[bucket.min(SPARKS.len() - 1)]
        })
        .collect()
}

fn trim_history(values: &mut Vec<f64>, length: usize) {
    if values.len() > length {
        values.remove(0);
    }
}

fn styled(text: &str, codes: &[&str], enabled: bool) -> String {
    if !enabled {
        return text.to_owned();
    }
    format!("{}{}{}", codes.concat(), text, RESET)
}

fn wait_for_exit(timeout: Duration) -> Result<bool> {
    if !event::poll(timeout)? {
        return Ok(false);
    }
    loop {
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press
                && matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
            {
                return Ok(true);
            }
        }
        if !event::poll(Duration::ZERO)? {
            return Ok(false);
        }
    }
}

struct TerminalSession {
    active: bool,
}

impl TerminalSession {
    fn enter(active: bool) -> Result<Self> {
        if active {
            enable_raw_mode()?;
            execute!(io::stdout(), EnterAlternateScreen, Hide)?;
        }
        Ok(Self { active })
    }

    fn draw(&mut self, frame: &str) -> Result<()> {
        execute!(io::stdout(), MoveTo(0, 0), Clear(ClearType::All))?;
        let mut stdout = io::stdout().lock();
        stdout.write_all(frame.as_bytes())?;
        stdout.flush()?;
        Ok(())
    }

    fn leave(&mut self) -> Result<()> {
        if self.active {
            execute!(io::stdout(), Show, LeaveAlternateScreen)?;
            disable_raw_mode()?;
            self.active = false;
        }
        Ok(())
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        if self.active {
            let _ = execute!(io::stdout(), Show, LeaveAlternateScreen);
            let _ = disable_raw_mode();
        }
    }
}

fn parse_args() -> Result<Config> {
    let mut config = Config::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--interval-ms" => {
                config.interval = Duration::from_millis(value(&mut arguments, "--interval-ms")?)
            }
            "--ticks" => config.ticks = Some(value(&mut arguments, "--ticks")?),
            "--seed" => config.seed = value(&mut arguments, "--seed")?,
            "--record" => config.record = true,
            "--no-color" => config.no_color = true,
            "-h" | "--help" => {
                println!(
                    "risk — live options strategy monitor\n\n\
                     cargo run --example risk -- [options]\n\n\
                     --interval-ms N    refresh delay (default 180)\n\
                     --ticks N          stop after N path steps\n\
                     --seed N           deterministic random seed\n\
                     --record           emit ANSI frames separated by form-feed\n\
                     --no-color         suppress ANSI color"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument {unknown}; use --help").into()),
        }
    }
    if config.ticks == Some(0) {
        return Err("--ticks must be positive".into());
    }
    Ok(config)
}

fn value<T>(arguments: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    arguments
        .next()
        .ok_or_else(|| format!("{name} requires a value").into())
        .and_then(|raw| {
            raw.parse()
                .map_err(|error| format!("invalid value for {name}: {error}").into())
        })
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
}
