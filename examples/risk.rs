//! `risk` is a live, simulation-driven options strategy monitor.
//!
//! The terminal and browser examples share one deterministic Rust simulation.
//! This view keeps the compact, keyboard-friendly feel of `top`.

mod support;

use std::io::{self, IsTerminal, Write};
use std::thread;
use std::time::Duration;

use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
};
use support::risk_model::{RiskSimulation, RiskSnapshot};

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
const RULE: &str =
    "────────────────────────────────────────────────────────────────────────────────────────────────";

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

fn main() -> Result<()> {
    let config = parse_args()?;
    let mut simulation = RiskSimulation::new(config.seed)?;
    let stdout_is_terminal = io::stdout().is_terminal();
    let interactive = stdout_is_terminal && !config.record;
    let color = !config.no_color && (interactive || config.record);
    let mut terminal = TerminalSession::enter(interactive)?;
    let limit = config
        .ticks
        .or(config.record.then_some(32))
        .or((!interactive).then_some(1));
    let mut tick = 0_usize;

    loop {
        tick += 1;
        let snapshot = simulation.next_snapshot()?;
        let frame = render(&snapshot, color);
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

fn render(snapshot: &RiskSnapshot, color: bool) -> String {
    let mut out = String::new();

    out.push_str(&styled(" VOLT / RISK ", &[BOLD, CYAN], color));
    out.push_str(&styled(
        "  SIMX EUROPEAN IRON CONDOR MONITOR\n",
        &[BOLD, WHITE],
        color,
    ));
    out.push_str(&styled(
        &format!(
            " step {:>5}   spot {:>8.2}   tick {:>+8.3}%   IV {:>6.2}%   DTE {:>5.2}\n",
            snapshot.step,
            snapshot.spot,
            snapshot.last_return * 100.0,
            snapshot.volatility * 100.0,
            snapshot.days_to_expiry
        ),
        &[MUTED],
        color,
    ));
    out.push_str(&styled(
        " model  correlated GBM + mean-reverting log volatility; Black-Scholes teaching marks\n",
        &[DIM, BLUE],
        color,
    ));
    out.push_str(&styled(
        &format!(
            " book   {}…   {} payoff knots   cash-settled European contracts\n",
            &snapshot.portfolio_hash[..12],
            snapshot.payoff_knots
        ),
        &[DIM, BLUE],
        color,
    ));
    out.push_str(&styled(&format!("{RULE}\n"), &[MUTED], color));

    out.push_str(&styled(
        " LIVE RISK               VALUE        DELTA        GAMMA      VEGA/PT    THETA/DAY\n",
        &[BOLD, AMBER],
        color,
    ));
    out.push_str(&format!(
        " Portfolio mark   {}  {:>+11.2}  {:>+11.3}  {:>+11.2}  {:>+11.2}\n",
        money(snapshot.metrics.mark, color),
        snapshot.metrics.delta,
        snapshot.metrics.gamma,
        snapshot.metrics.vega,
        snapshot.metrics.theta
    ));
    out.push_str(&format!(
        " Model P&L        {}\n",
        money(snapshot.metrics.pnl, color)
    ));
    out.push_str(&format!(
        " Expiry bounds     max {}   min {}   break-even {:>6.2} / {:>6.2}\n",
        money(snapshot.bounds.max_profit, color),
        money(snapshot.bounds.max_loss, color),
        snapshot.bounds.lower_break_even,
        snapshot.bounds.upper_break_even
    ));
    out.push_str(&format!(
        " Spot history      {}\n",
        styled(&sparkline(&snapshot.spot_history, 28), &[CYAN], color)
    ));
    out.push_str(&format!(
        " P&L history       {}\n",
        styled(
            &sparkline(&snapshot.pnl_history, 28),
            &[if snapshot.metrics.pnl >= 0.0 {
                GREEN
            } else {
                RED
            }],
            color
        )
    ));

    out.push_str(&styled(
        "\n LEGS                 QTY    STRIKE       MARK         IV        DELTA       LEG P&L\n",
        &[BOLD, AMBER],
        color,
    ));
    for leg in &snapshot.legs {
        out.push_str(&format!(
            " {:<18} {:>+4}  {:>8.2}  {:>9.3}  {:>8.2}%  {:>+11.2}  {}\n",
            leg.contract_id,
            leg.quantity,
            leg.strike,
            leg.mark,
            leg.volatility * 100.0,
            leg.delta,
            money(leg.pnl, color)
        ));
    }

    out.push_str(&styled("\n EXPIRY P&L\n", &[BOLD, AMBER], color));
    out.push_str(" Scenario      ");
    for point in &snapshot.expiry_curve {
        let label = if point.shock == 0.0 {
            "SPOT".to_owned()
        } else {
            format!("{:+.0}%", point.shock * 100.0)
        };
        out.push_str(&format!("{label:>11}"));
    }
    out.push('\n');
    out.push_str(" Terminal spot ");
    for point in &snapshot.expiry_curve {
        out.push_str(&format!("{:>11.2}", point.spot));
    }
    out.push('\n');
    out.push_str(" P&L           ");
    for point in &snapshot.expiry_curve {
        out.push_str(&pnl_cell(point.pnl, color));
    }
    out.push('\n');

    out.push_str(&styled(
        "\n MARK-TO-MODEL SHOCKS            IV -5pt         IV NOW         IV +5pt\n",
        &[BOLD, AMBER],
        color,
    ));
    for row in snapshot.shock_grid.chunks(3) {
        out.push_str(&format!(
            " Spot {:+.0}%                 ",
            row[0].spot_shock * 100.0
        ));
        for cell in row {
            out.push_str(&wide_pnl_cell(cell.pnl, color));
        }
        out.push('\n');
    }
    out.push_str(&styled(
        "\n q quit  •  deterministic simulation  •  educational marks  •  no trading authority\n",
        &[DIM, MUTED],
        color,
    ));
    out
}

fn money(value: f64, color: bool) -> String {
    styled(
        &format!("${value:>+11.2}"),
        &[BOLD, if value >= 0.0 { GREEN } else { RED }],
        color,
    )
}

fn pnl_cell(value: f64, color: bool) -> String {
    styled(
        &format!("{value:>+11.0}"),
        &[if value >= 0.0 { GREEN } else { RED }],
        color,
    )
}

fn wide_pnl_cell(value: f64, color: bool) -> String {
    styled(
        &format!("{value:>+17.0}"),
        &[if value >= 0.0 { GREEN } else { RED }],
        color,
    )
}

fn sparkline(values: &[f64], width: usize) -> String {
    let start = values.len().saturating_sub(width);
    let values = &values[start..];
    let first = values.first().copied().unwrap_or_default();
    let padded = std::iter::repeat_n(first, width.saturating_sub(values.len()))
        .chain(values.iter().copied())
        .collect::<Vec<_>>();
    let minimum = padded.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = padded.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (maximum - minimum).max(f64::EPSILON);
    padded
        .iter()
        .map(|value| {
            let bucket = (((value - minimum) / range) * (SPARKS.len() - 1) as f64).round() as usize;
            SPARKS[bucket.min(SPARKS.len() - 1)]
        })
        .collect()
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
