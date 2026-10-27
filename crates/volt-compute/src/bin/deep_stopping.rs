//! Native backward stopping training/evaluation and a bounded JSON-line worker.
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::{
    io::{self, BufRead, Read, Write},
    path::PathBuf,
};
use volt_compute::deep_stopping::{
    IntervalHedgeCheckpoint, StoppingCheckpoint, StoppingEvaluationConfig, StoppingTrainingConfig,
    evaluate_bounds, evaluate_interval_hedge, stop_path, train, train_interval_hedge,
};
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    Example {
        #[arg(long)]
        output: PathBuf,
    },
    Train {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
    },
    Evaluate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    HedgeTrain {
        #[arg(long)]
        holder: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        initial_price: f64,
        #[arg(long)]
        checkpoint: PathBuf,
    },
    HedgeEvaluate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        paths: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value_t = 0)]
        first_path: u64,
        #[arg(long)]
        output: PathBuf,
    },
    Serve,
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Train {
        config: StoppingTrainingConfig,
    },
    Evaluate {
        checkpoint: StoppingCheckpoint,
        config: StoppingEvaluationConfig,
    },
    Stop {
        checkpoint: StoppingCheckpoint,
        spots: Vec<f64>,
    },
    Ping,
}
const LIMIT: usize = 16 * 1024 * 1024;
fn read<T: serde::de::DeserializeOwned>(p: PathBuf) -> Result<T, Box<dyn std::error::Error>> {
    let mut data = Vec::new();
    std::fs::File::open(p)?
        .take((LIMIT + 1) as u64)
        .read_to_end(&mut data)?;
    if data.len() > LIMIT {
        return Err("input exceeds16MiB".into());
    }
    Ok(serde_json::from_slice(&data)?)
}
fn save(p: PathBuf, x: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    let tmp = p.with_extension(format!("partial-{}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&tmp)?;
    let result = serde_json::to_writer_pretty(&mut file, x)
        .map_err(io::Error::other)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all());
    drop(file);
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    let result = std::fs::hard_link(&tmp, &p);
    let _ = std::fs::remove_file(&tmp);
    result?;
    Ok(())
}
fn bounded_line(input: &mut impl BufRead) -> io::Result<Vec<u8>> {
    let mut line = Vec::new();
    loop {
        let bytes = input.fill_buf()?;
        if bytes.is_empty() {
            return Ok(line);
        }
        let count = bytes
            .iter()
            .position(|x| *x == b'\n')
            .map(|n| n + 1)
            .unwrap_or(bytes.len());
        if line.len() + count > LIMIT {
            return Err(io::Error::other(
                "request exceeds16MiB; terminating without reinterpreting remainder",
            ));
        }
        let done = bytes[count - 1] == b'\n';
        line.extend_from_slice(&bytes[..count]);
        input.consume(count);
        if done {
            return Ok(line);
        }
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Action::Example { output } => save(
            output,
            &serde_json::json!({"training":StoppingTrainingConfig::default(),"evaluation":StoppingEvaluationConfig::default()}),
        )?,
        Action::Train { config, checkpoint } => save(checkpoint, &train(&read(config)?)?)?,
        Action::Evaluate {
            checkpoint,
            config,
            output,
        } => save(
            output,
            &evaluate_bounds(&read(checkpoint)?, &read(config)?)?,
        )?,
        Action::HedgeTrain {
            holder,
            config,
            initial_price,
            checkpoint,
        } => save(
            checkpoint,
            &train_interval_hedge(&read(holder)?, &read(config)?, initial_price)?,
        )?,
        Action::HedgeEvaluate {
            checkpoint,
            paths,
            seed,
            first_path,
            output,
        } => save(
            output,
            &evaluate_interval_hedge(
                &read::<IntervalHedgeCheckpoint>(checkpoint)?,
                paths,
                seed,
                first_path,
            )?,
        )?,
        Action::Serve => {
            let stdin = io::stdin();
            let mut out = io::BufWriter::new(io::stdout().lock());
            let mut input = stdin.lock();
            loop {
                let line = bounded_line(&mut input)?;
                if line.is_empty() {
                    break;
                }
                let result = (|| -> Result<serde_json::Value, Box<dyn std::error::Error>> {
                    if line.len() > LIMIT {
                        return Err("request exceeds16MiB".into());
                    }
                    Ok(match serde_json::from_slice::<Request>(&line)? {
                        Request::Train { config } => serde_json::to_value(train(&config)?)?,
                        Request::Evaluate { checkpoint, config } => {
                            serde_json::to_value(evaluate_bounds(&checkpoint, &config)?)?
                        }
                        Request::Stop { checkpoint, spots } => {
                            serde_json::to_value(stop_path(&checkpoint, &spots)?)?
                        }
                        Request::Ping => {
                            serde_json::json!({"schema":volt_compute::deep_stopping::SCHEMA})
                        }
                    })
                })();
                let reply = match result {
                    Ok(result) => serde_json::json!({"ok":true,"result":result}),
                    Err(e) => serde_json::json!({"ok":false,"error":e.to_string()}),
                };
                let encoded = serde_json::to_vec(&reply)?;
                if encoded.len() > LIMIT {
                    writeln!(
                        &mut out,
                        "{{\"ok\":false,\"error\":\"response exceeds16MiB; use file output or smaller cohorts\"}}"
                    )?;
                } else {
                    out.write_all(&encoded)?;
                }
                writeln!(&mut out)?;
                out.flush()?;
            }
        }
    }
    Ok(())
}
