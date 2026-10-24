use clap::Parser;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};
use volt_compute::{
    market_path_diagnostics::{
        PAPER_HESTON_CALL_REFERENCE, PAPER_REFERENCE_SOURCE, PathDiagnosticConfig, diagnose_paths,
    },
    market_paths::{IntegratedVarianceRule, MarketModel, MarketPathConfig},
};

#[derive(Parser)]
#[command(
    about = "Streaming host f64 BS/Heston path controls with exact CIR endpoints and explicit scheme bias"
)]
struct Args {
    #[arg(long, default_value_t = 1_000_000)]
    paths: usize,
    #[arg(long, default_value_t = 30)]
    steps: usize,
    #[arg(long, default_value_t = 30.0)]
    horizon_days: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    first_path_id: u64,
    #[arg(long, default_value_t = 32768)]
    chunk_paths: usize,
    #[arg(long, default_value_t = 64)]
    host_working_mib: u64,
    #[arg(long,default_value="heston",value_parser=["heston","bs"])]
    model: String,
    #[arg(long,default_value="left",value_parser=["left","trapezoid"])]
    integral: String,
    #[arg(long, default_value_t = 100.0)]
    spot: f64,
    #[arg(long, default_value_t = 100.0)]
    strike: f64,
    #[arg(long, default_value_t = 0.0)]
    rate: f64,
    #[arg(long, default_value_t = 0.0)]
    dividend_yield: f64,
    #[arg(long, default_value_t = 0.2)]
    bs_volatility: f64,
    #[arg(long, default_value_t = 0.04)]
    initial_variance: f64,
    #[arg(long, default_value_t = 1.0)]
    kappa: f64,
    #[arg(long, default_value_t = 0.04)]
    long_run_variance: f64,
    #[arg(long, default_value_t = 2.0)]
    vol_of_variance: f64,
    #[arg(long,default_value_t=-0.7)]
    correlation: f64,
    #[arg(long)]
    reference_call: Option<f64>,
    #[arg(long, requires = "reference_call")]
    reference_source: Option<String>,
    #[arg(long)]
    output: Option<PathBuf>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut market = MarketPathConfig::paper_heston(args.paths, args.seed);
    market.first_path_id = args.first_path_id;
    market.steps = args.steps;
    market.dt = args.horizon_days / 365.0 / args.steps as f64;
    market.initial_spot = args.spot;
    market.rate = args.rate;
    market.dividend_yield = args.dividend_yield;
    market.integrated_variance_rule = if args.integral == "left" {
        IntegratedVarianceRule::Left
    } else {
        IntegratedVarianceRule::Trapezoid
    };
    if args.model == "bs" {
        market.model = MarketModel::BlackScholes {
            volatility: args.bs_volatility,
        };
    } else if let MarketModel::Heston { ref mut params } = market.model {
        params.initial_variance = args.initial_variance;
        params.kappa = args.kappa;
        params.long_run_variance = args.long_run_variance;
        params.vol_of_variance = args.vol_of_variance;
        params.correlation = args.correlation;
    }
    market.validate()?;
    let paper = args.model == "heston"
        && args.spot == 100.0
        && args.strike == 100.0
        && args.horizon_days == 30.0
        && args.initial_variance == 0.04
        && args.kappa == 1.0
        && args.long_run_variance == 0.04
        && args.vol_of_variance == 2.0
        && args.correlation == -0.7
        && args.rate == 0.0
        && args.dividend_yield == 0.0;
    let (reference, source) = if let Some(reference) = args.reference_call {
        (
            Some(reference),
            Some(
                args.reference_source
                    .unwrap_or_else(|| "explicit caller supplied external reference".into()),
            ),
        )
    } else if paper {
        (
            Some(PAPER_HESTON_CALL_REFERENCE),
            Some(PAPER_REFERENCE_SOURCE.into()),
        )
    } else if args.model == "bs" {
        let price = if args.bs_volatility == 0.0 {
            (args.spot * (-args.dividend_yield * market.horizon()).exp()
                - args.strike * (-args.rate * market.horizon()).exp())
            .max(0.0)
        } else {
            volt_compute::reference_price(&volt_compute::OptionRow {
                row_id: "path-control".into(),
                contract_id: "numeric-call".into(),
                side: volt_compute::OptionSide::Call,
                spot: args.spot,
                strike: args.strike,
                maturity_years: market.horizon(),
                volatility: args.bs_volatility,
                rate: args.rate,
                dividend_yield: args.dividend_yield,
                quantity: 1,
                multiplier: 1,
            })?
        };
        (
            Some(price),
            Some("independent scalar Black-Scholes libm CDF".into()),
        )
    } else {
        (None, None)
    };
    let config = PathDiagnosticConfig {
        market,
        call_strike: args.strike,
        requested_chunk_paths: args.chunk_paths,
        host_working_bytes: args
            .host_working_mib
            .checked_mul(1024 * 1024)
            .ok_or("host byte count overflow")?,
        call_reference: reference,
        reference_source: source,
    };
    let report = diagnose_paths(&config)?;
    if let Some(path) = args.output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    Ok(())
}
