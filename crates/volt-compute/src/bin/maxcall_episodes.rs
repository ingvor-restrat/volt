//! Turn a selected multi-asset holder into complete, financed writer episodes.
use clap::Parser;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::PathBuf,
};
use volt_compute::{
    ExecutionTarget, Precision,
    maxcall_learning::{MaxCallCheckpoint, MaxCallPredictor},
    multi_asset_market::{
        MaxCallEpisodeConfig, MaxCallStoppingPolicy, assemble_episodes, generate_paths,
    },
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    request: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    paths: usize,
    seed: u64,
    namespace: u64,
    first_path: u64,
    target: ExecutionTarget,
    precision: Precision,
    inference_batch_rows: usize,
    inference_working_bytes: u64,
    economics: MaxCallEpisodeConfig,
}
fn read<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<T, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("input exceeds 16 MiB".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut checkpoint_bytes = Vec::new();
    std::fs::File::open(&args.checkpoint)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut checkpoint_bytes)?;
    if checkpoint_bytes.len() > 16 * 1024 * 1024 {
        return Err("checkpoint exceeds 16 MiB".into());
    }
    let checkpoint: MaxCallCheckpoint = serde_json::from_slice(&checkpoint_bytes)?;
    let request: Request = read(&args.request)?;
    let actual_hash = format!("{:x}", Sha256::digest(&checkpoint_bytes));
    if !actual_hash.eq_ignore_ascii_case(&request.economics.holder_sha256) {
        return Err("holder SHA-256 differs from the supplied checkpoint".into());
    }
    checkpoint.validate()?;
    if request.paths == 0 || request.paths > 65536 {
        return Err("use 1..65536 paths per episode file".into());
    }
    let market = checkpoint.market();
    let paths = generate_paths(
        market,
        request.paths,
        request.seed,
        request.first_path,
        request.namespace,
        request.economics.max_working_bytes,
    )?;
    let mut policy = MaxCallPredictor::new(
        &checkpoint,
        request.target,
        request.precision,
        request.inference_batch_rows,
        request.inference_working_bytes,
    )?;
    let dates = market.times.len();
    let mut stops = vec![dates - 1; request.paths];
    for t in 0..dates - 1 {
        if !market.exercise[t] {
            continue;
        }
        let active: Vec<usize> = stops
            .iter()
            .enumerate()
            .filter_map(|(i, tau)| (*tau == dates - 1).then_some(i))
            .collect();
        if active.is_empty() {
            break;
        }
        let mut states = Vec::with_capacity(active.len() * market.assets());
        for i in &active {
            states.extend_from_slice(paths.state(*i, t)?);
        }
        let decisions =
            MaxCallStoppingPolicy::should_stop_batch(&mut policy, t, &states, market.assets())?;
        if decisions.len() != active.len() {
            return Err("holder returned a different decision count".into());
        }
        for (i, stop) in active.into_iter().zip(decisions) {
            if stop {
                stops[i] = t;
            }
        }
    }
    let episodes = assemble_episodes(market, &paths, &stops, &request.economics)?;
    let temporary = args
        .output
        .with_extension(format!("partial-{}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let write = serde_json::to_writer(&mut file, &episodes)
        .map_err(std::io::Error::other)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all());
    drop(file);
    if let Err(e) = write {
        let _ = std::fs::remove_file(&temporary);
        return Err(e.into());
    }
    let published = std::fs::hard_link(&temporary, &args.output);
    let _ = std::fs::remove_file(&temporary);
    published?;
    Ok(())
}
