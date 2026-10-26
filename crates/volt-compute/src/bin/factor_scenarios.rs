//! Stream bounded factor/signal scenarios into native-learner episode batches.
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::PathBuf,
};
use volt_compute::{
    OptionSide,
    factor_scenarios::*,
    joint_option_market::{ScenarioRange, validate_ranges},
    native_learner::Split,
};
#[derive(Parser)]
#[command(
    about = "Explicit factor-risk, predictive-signal and European-liability episode generator"
)]
struct Args {
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    example: bool,
    #[arg(long, default_value_t = 128)]
    batch_paths: usize,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    model: FactorModel,
    config: FactorConfig,
    partitions: Vec<ScenarioRange>,
}
fn illustration() -> Request {
    let model=FactorModel{asset_ids:vec!["SYNTH-A".into(),"SYNTH-B".into()],factor_ids:vec!["synthetic-equity-factor".into()],initial_prices:vec![100.,80.],loadings:vec![vec![1.],vec![0.8]],annual_factor_covariance:vec![vec![0.0256]],annual_idiosyncratic_volatility:vec![0.12,0.1],dividend_yields:vec![0.,0.],annual_base_excess_return:vec![0.,0.],annual_signal_coefficients:vec![0.2,-0.12],coefficient_uncertainty_std:vec![0.03,0.02],signal_initial:0.,signal_persistence:0.95,signal_innovation_std:(1_f64-0.95_f64.powi(2)).sqrt(),provenance:FactorProvenance{model_id:"synthetic-factor-and-signal-illustration".into(),source_sha256:"0".repeat(64),synthetic:true,decision_origin_ns:None,known_at_ns:None,assumptions:"Invented parameters. Annual decimal stock/factor returns; persistent observed dimensionless signal. Per-path drift-response coefficients are uncertain and latent. No fitted forecast or historical-data claim. Actual typed model bytes enter dataset identity.".into()}};
    let config = FactorConfig {
        seed: 731807,
        steps: 20,
        dt: 1. / 252.,
        rate: 0.,
        liabilities: vec![FactorOption {
            asset: 0,
            side: OptionSide::Call,
            strike: 100.,
            maturity_steps: 20,
            quantity: 1,
            multiplier: 1,
        }],
        observation: FactorObservation::MarketAndSignal,
        trading_costs: vec![0.0005; 2],
        liquidation_costs: vec![0.; 2],
        max_working_bytes: 64 * 1024 * 1024,
    };
    Request {
        model,
        config,
        partitions: vec![
            ScenarioRange {
                split: Split::Train,
                first_path: 0,
                paths: 16,
            },
            ScenarioRange {
                split: Split::Validation,
                first_path: 1_000_000,
                paths: 16,
            },
            ScenarioRange {
                split: Split::Test,
                first_path: 2_000_000,
                paths: 16,
            },
        ],
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let a = Args::parse();
    if a.example {
        if a.input.is_some() || a.output.is_some() {
            return Err("--example takes no input/output".into());
        }
        println!("{}", serde_json::to_string_pretty(&illustration())?);
        return Ok(());
    }
    if a.batch_paths == 0 || a.batch_paths > 4096 {
        return Err("batch paths must be1..4096".into());
    }
    let input = a.input.ok_or("--input required")?;
    let output = a.output.ok_or("--output required")?;
    if input == output
        || fs::canonicalize(&input)
            .ok()
            .zip(fs::canonicalize(&output).ok())
            .is_some_and(|(a, b)| a == b)
    {
        return Err("output must differ from input".into());
    }
    let mut b = Vec::new();
    File::open(&input)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut b)?;
    if b.len() > 16 * 1024 * 1024 {
        return Err("request exceeds16MiB".into());
    }
    let r: Request = serde_json::from_slice(&b)?;
    r.config.validate(&r.model)?;
    validate_ranges(&r.partitions)?;
    let count = r
        .partitions
        .iter()
        .try_fold(0usize, |n, p| n.checked_add(p.paths))
        .ok_or("path count overflow")?;
    if count > 1_000_000 {
        return Err("request exceeds one million paths; use separate explicit requests".into());
    }
    for part in &r.partitions {
        plan(
            &r.model,
            &r.config,
            &ScenarioRange {
                split: part.split,
                first_path: part.first_path,
                paths: part.paths.min(a.batch_paths),
            },
        )?;
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temp = output.with_extension(format!("factor-tmp-{}", std::process::id()));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)?;
    let mut writer = BufWriter::new(file);
    let result = (|| -> Result<(), Box<dyn Error>> {
        for part in &r.partitions {
            let mut used = 0;
            while used < part.paths {
                let n = (part.paths - used).min(a.batch_paths);
                let range = ScenarioRange {
                    split: part.split,
                    first_path: part.first_path + used as u64,
                    paths: n,
                };
                let x = generate(&r.model, &r.config, range)?;
                serde_json::to_writer(&mut writer, &x)?;
                writer.write_all(b"\n")?;
                used += n;
            }
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(&temp, &output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result?;
    eprintln!("{count} paths streamed with batch cap{}", a.batch_paths);
    Ok(())
}
