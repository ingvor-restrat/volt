//! Bounded dated GBM/holder episodes as a single JSON response.
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    io::{self, Read, Write},
    path::PathBuf,
};
use volt_compute::{
    bermudan_market::{
        BermudanMarketConfig, SuppliedStopping, generate, generate_from_paths,
        generate_from_paths_with_stopping,
    },
    exercise_pricing::HolderGridCache,
    joint_option_market::ScenarioRange,
};
#[derive(Parser)]
#[command(about = "Numerical Bermudan holder, GBM stopped episodes and financed delta control")]
struct Args {
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    config: BermudanMarketConfig,
    range: ScenarioRange,
    #[serde(default)]
    pre_dividend_spots: Option<Vec<f64>>,
    #[serde(default)]
    stopping: Option<SuppliedStopping>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut data = Vec::new();
    let reader: Box<dyn Read> = match &args.input {
        Some(p) => Box::new(std::fs::File::open(p)?),
        None => Box::new(io::stdin()),
    };
    reader.take(2 * 1024 * 1024 + 1).read_to_end(&mut data)?;
    if data.len() > 2 * 1024 * 1024 {
        return Err("request exceeds2MiB".into());
    }
    let request: Request = serde_json::from_slice(&data)?;
    let mut cache = HolderGridCache::new(request.config.max_holder_scalar_values)?;
    let result = match (&request.pre_dividend_spots, &request.stopping) {
        (None, None) => generate(&request.config, request.range, &mut cache)?,
        (Some(spots), None) => {
            generate_from_paths(&request.config, request.range, spots, &mut cache)?
        }
        (Some(spots), Some(stops)) => generate_from_paths_with_stopping(
            &request.config,
            request.range,
            spots,
            stops,
            &mut cache,
        )?,
        (None, Some(_)) => {
            return Err("supplied stopping requires an explicit shared market cohort".into());
        }
    };
    if let Some(path) = args.output {
        // Do not overwrite an existing input, result or concurrent producer.
        let temporary = path.with_extension(format!("partial-{}", std::process::id()));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let write = serde_json::to_writer(&mut file, &result)
            .map_err(io::Error::other)
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.sync_all());
        drop(file);
        if let Err(e) = write {
            let _ = std::fs::remove_file(&temporary);
            return Err(e.into());
        }
        let publish = std::fs::hard_link(&temporary, &path);
        let _ = std::fs::remove_file(&temporary);
        publish?;
    } else {
        let mut out = io::BufWriter::new(io::stdout().lock());
        serde_json::to_writer(&mut out, &result)?;
        out.write_all(b"\n")?;
    }
    Ok(())
}
