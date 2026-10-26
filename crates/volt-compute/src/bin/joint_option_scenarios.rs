//! Bounded, parameter-explicit joint option scenario producer.
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};
use volt_compute::{joint_option_market::*, native_learner::Split};
#[derive(Parser)]
#[command(about = "Joint index-return/five-coefficient IV scenarios and native episode batches")]
struct Args {
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    /// Print a synthetic request with illustrative linear innovation quantiles.
    #[arg(long)]
    example: bool,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    parameters: JointParameters,
    config: JointOptionConfig,
    initial: JointState,
    innovation_model: InnovationModel,
    partitions: Vec<ScenarioRange>,
    range: ScenarioRange,
    #[serde(default)]
    fixed_innovations: Option<Vec<[f64; 6]>>,
}
fn example() -> Request {
    let mut rows = [[0.; 6]; 14];
    for i in 0..5 {
        rows[i + 1][i] = 1.;
    }
    rows[0][5] = 2.;
    rows[8][0] = 0.1;
    rows[8][5] = 1.;
    rows[9] = [0.9; 6];
    for sigma in rows[7].iter_mut().take(5).skip(1) {
        *sigma = 0.05;
    }
    rows[13] = [3.; 6];
    let provenance=JointProvenance{decision_origin_ns:None,calibration_known_ns:None,initial_state_known_ns:None,calibration_id:"synthetic-illustration".into(),calibration_sha256:"0".repeat(64),initial_state_id:"synthetic-initial-state".into(),initial_state_sha256:"0".repeat(64),innovation_table_id:"synthetic-linear-quantiles-NOT-calibrated-NIG".into(),innovation_table_sha256:"0".repeat(64),synthetic:true,availability_note:"All inputs chosen for demonstration; no historical observations or calibrated distribution. Placeholder source digests; actual typed input bytes are bound in dataset identity.".into()};
    let mut correlation = [[0.; 6]; 6];
    for (i, r) in correlation.iter_mut().enumerate() {
        r[i] = 1.;
    }
    Request {
        parameters: JointParameters { rows },
        config: JointOptionConfig::paper_case(731801, provenance),
        initial: JointState {
            stock: 100.,
            betas: [0.2, 0., -0.05, 0., 0.],
            lag_betas: [0.2, 0., -0.05, 0., 0.],
            conditional_variances: [0.0004, 0.00001, 0.00001, 0.00001, 0.00001, 0.04],
            previous_innovations: [0.; 6],
        },
        innovation_model: InnovationModel {
            original_correlation: correlation,
            probabilities: vec![0.0001, 0.9995],
            quantiles: vec![[-1.; 6], [1.; 6]],
        },
        partitions: vec![
            ScenarioRange {
                split: Split::Train,
                first_path: 0,
                paths: 8,
            },
            ScenarioRange {
                split: Split::Validation,
                first_path: 1_000_000,
                paths: 8,
            },
            ScenarioRange {
                split: Split::Test,
                first_path: 2_000_000,
                paths: 8,
            },
        ],
        range: ScenarioRange {
            split: Split::Train,
            first_path: 0,
            paths: 8,
        },
        fixed_innovations: None,
    }
}
fn write_atomic(path: &Path, data: &impl Serialize) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("joint-tmp-{}", std::process::id()));
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| -> Result<(), Box<dyn Error>> {
        serde_json::to_writer(&mut f, data)?;
        f.write_all(b"\n")?;
        f.sync_all()?;
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}
fn main() -> Result<(), Box<dyn Error>> {
    let a = Args::parse();
    if a.example {
        if a.input.is_some() || a.output.is_some() {
            return Err("--example takes no input/output".into());
        }
        println!("{}", serde_json::to_string_pretty(&example())?);
        return Ok(());
    }
    let input = a
        .input
        .ok_or("--input required (use --example for a synthetic request)")?;
    let output = a.output.ok_or("--output required")?;
    if input == output
        || fs::canonicalize(&input)
            .ok()
            .zip(fs::canonicalize(&output).ok())
            .is_some_and(|(a, b)| a == b)
    {
        return Err("output must differ from input".into());
    }
    let mut bytes = Vec::new();
    File::open(&input)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("request exceeds16MiB".into());
    }
    let r: Request = serde_json::from_slice(&bytes)?;
    validate_ranges(&r.partitions)?;
    let end = r
        .range
        .first_path
        .checked_add(r.range.paths as u64)
        .ok_or("range overflow")?;
    if !r.partitions.iter().any(|p| {
        p.split == r.range.split
            && r.range.first_path >= p.first_path
            && end <= p.first_path + p.paths as u64
    }) {
        return Err("requested range outside declared split".into());
    }
    let out = if let Some(e) = r.fixed_innovations {
        generate_with_innovations(&r.parameters, &r.config, &r.initial, r.range, e)?
    } else {
        generate(
            &r.parameters,
            &r.config,
            &r.initial,
            &r.innovation_model,
            r.range,
        )?
    };
    write_atomic(&output, &out)?;
    eprintln!(
        "{} paths; {} decisions; modeled host bytes {}; dataset {}",
        out.batch.paths, out.config.steps, out.estimated_working_bytes, out.dataset_id
    );
    Ok(())
}
