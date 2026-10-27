//! Bounded JSON interface to correlated GBM max-call scenarios and episodes.
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    io::{self, Read, Write},
    path::PathBuf,
};
use volt_compute::multi_asset_market::*;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: Option<PathBuf>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    market: MaxCallMarket,
    paths: usize,
    seed: u64,
    first_path: u64,
    namespace: u64,
    max_working_bytes: u64,
    #[serde(default)]
    stopping_steps: Option<Vec<usize>>,
    #[serde(default)]
    episode: Option<MaxCallEpisodeConfig>,
}
#[derive(Serialize)]
struct Response {
    paths: MaxCallPaths,
    episodes: Option<MaxCallEpisodes>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut input = Vec::new();
    let reader: Box<dyn Read> = if let Some(p) = args.input {
        Box::new(std::fs::File::open(p)?)
    } else {
        Box::new(io::stdin())
    };
    reader.take(2 * 1024 * 1024 + 1).read_to_end(&mut input)?;
    if input.len() > 2 * 1024 * 1024 {
        return Err("request exceeds 2 MiB".into());
    }
    let request: Request = serde_json::from_slice(&input)?;
    if request.stopping_steps.is_some() != request.episode.is_some() {
        return Err("stopping_steps and episode must be supplied together".into());
    }
    let paths = generate_paths(
        &request.market,
        request.paths,
        request.seed,
        request.first_path,
        request.namespace,
        request.max_working_bytes,
    )?;
    let episodes = if let (Some(stops), Some(config)) = (&request.stopping_steps, &request.episode)
    {
        Some(assemble_episodes(&request.market, &paths, stops, config)?)
    } else {
        None
    };
    let mut out = io::BufWriter::new(io::stdout().lock());
    serde_json::to_writer(&mut out, &Response { paths, episodes })?;
    out.write_all(b"\n")?;
    Ok(())
}
