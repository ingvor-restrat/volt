//! Bounded JSON interface for European/Bermudan/American CPU-f64 pricing and IV.
use clap::Parser;
use std::{
    fs,
    io::{self, Read, Write},
    path::PathBuf,
};
use volt_compute::exercise_pricing::{ExerciseBatchRequest, price_batch};
#[derive(Parser)]
#[command(
    about = "Dated exercise-aware CPU-f64 option pricing/Greeks; JSON request on stdin or --input"
)]
struct Args {
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args
        .input
        .as_ref()
        .zip(args.output.as_ref())
        .is_some_and(|(a, b)| {
            a == b
                || fs::canonicalize(a)
                    .ok()
                    .zip(fs::canonicalize(b).ok())
                    .is_some_and(|(a, b)| a == b)
        })
    {
        return Err("output must differ from input".into());
    }
    let mut bytes = Vec::new();
    let reader: Box<dyn Read> = if let Some(p) = args.input {
        Box::new(fs::File::open(p)?)
    } else {
        Box::new(io::stdin())
    };
    reader.take(4 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 4 * 1024 * 1024 {
        return Err("request exceeds 4MiB".into());
    }
    let request: ExerciseBatchRequest = serde_json::from_slice(&bytes)?;
    let result = price_batch(&request)?;
    let json = serde_json::to_vec_pretty(&result)?;
    if let Some(p) = args.output {
        let temp = p.with_extension(format!("partial-{}", std::process::id()));
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        if let Err(e) = file.write_all(&json).and_then(|_| file.sync_all()) {
            let _ = fs::remove_file(&temp);
            return Err(e.into());
        }
        drop(file);
        if let Err(e) = fs::rename(&temp, p) {
            let _ = fs::remove_file(temp);
            return Err(e.into());
        }
    } else {
        println!("{}", String::from_utf8(json)?);
    }
    Ok(())
}
