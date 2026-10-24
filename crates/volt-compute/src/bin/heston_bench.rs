use clap::Parser;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use volt_compute::{
    ExecutionTarget, OptionSide, Precision,
    budget::MemoryBudget,
    heston::{HestonOption, HestonParams, QuadratureConfig},
    heston_job::{HestonJobConfig, HestonJobRequest, HestonPosition, metrics, plan_heston},
    heston_validation::{quadrature_stages, run_checked_heston},
};

#[derive(Parser)]
#[command(
    about = "Heston Fourier risk with independent node/cutoff refinement and explicit position accounting"
)]
struct Args {
    #[arg(long, default_value = "cpu")]
    device: ExecutionTarget,
    #[arg(long,default_value="f64",value_parser=["f32","f64"])]
    precision: String,
    #[arg(long,default_value="paper",value_parser=["paper","lewis"])]
    profile: String,
    #[arg(long, default_value_t = 4)]
    rows: usize,
    /// Base nodes; the checked run also uses twice this number.
    #[arg(long)]
    nodes: Option<usize>,
    /// Base Fourier cutoff; the checked run also uses twice this value.
    #[arg(long)]
    cutoff: Option<f64>,
    #[arg(long, default_value_t = 65536)]
    max_batch_rows: usize,
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    compute_budget_mib: u64,
    #[arg(long, default_value_t = 256)]
    host_working_mib: u64,
    #[arg(long, default_value_t = 32)]
    compute_reserve_mib: u64,
    #[arg(long)]
    plan_only: bool,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    greeks_output: Option<PathBuf>,
}
fn fixture(profile: &str, n: usize) -> HestonJobRequest {
    let paper = profile == "paper";
    let options: Vec<_> = (0..n)
        .map(|i| HestonOption {
            row_id: format!("heston-{i}"),
            side: if i % 2 == 0 {
                OptionSide::Call
            } else {
                OptionSide::Put
            },
            spot: 100.0,
            strike: if paper {
                [90.0, 100.0, 110.0, 100.0][i % 4]
            } else {
                [80.0, 90.0, 100.0, 110.0, 120.0][i % 5]
            },
            maturity_years: if paper { 30.0 / 365.0 } else { 1.0 },
            rate: if paper { 0.0 } else { 0.01 },
            dividend_yield: if paper { 0.0 } else { 0.02 },
        })
        .collect();
    let params = vec![
        if paper {
            HestonParams {
                initial_variance: 0.04,
                kappa: 1.0,
                long_run_variance: 0.04,
                vol_of_variance: 2.0,
                correlation: -0.7,
            }
        } else {
            HestonParams {
                initial_variance: 0.04,
                kappa: 4.0,
                long_run_variance: 0.25,
                vol_of_variance: 1.0,
                correlation: -0.5,
            }
        };
        n
    ];
    let positions = options
        .iter()
        .enumerate()
        .map(|(i, o)| HestonPosition {
            row_id: o.row_id.clone(),
            contract_id: format!("synthetic-heston-{i}"),
            quantity: if i % 3 == 0 { -2 } else { 1 },
            multiplier: 100,
        })
        .collect();
    HestonJobRequest {
        schema: "volt.compute.heston_request.v1".into(),
        currency: "USD".into(),
        options,
        params,
        positions,
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.input.is_none() && (args.rows == 0 || args.rows > 16384) {
        return Err("fixture rows must be1..=16384".into());
    }
    let outputs: Vec<_> = [&args.output, &args.greeks_output]
        .into_iter()
        .flatten()
        .collect();
    for output in &outputs {
        if let Some(input) = &args.input
            && resolved(output)? == resolved(input)?
        {
            return Err("output cannot overwrite input".into());
        }
    }
    if outputs.len() == 2 && resolved(outputs[0])? == resolved(outputs[1])? {
        return Err("report and Greeks outputs must differ".into());
    }
    let supplied = args.input.is_some();
    let request: HestonJobRequest = if let Some(path) = args.input {
        serde_json::from_reader(File::open(path)?)?
    } else {
        fixture(&args.profile, args.rows)
    };
    let mib = |n: u64| {
        n.checked_mul(1024 * 1024)
            .ok_or("budget byte count overflow")
    };
    let config = HestonJobConfig {
        target: args.device,
        precision: if args.precision == "f64" {
            Precision::F64
        } else {
            Precision::F32
        },
        quadrature: QuadratureConfig {
            nodes: args
                .nodes
                .unwrap_or(if args.profile == "paper" { 1024 } else { 256 }),
            upper_bound: args.cutoff.unwrap_or(if args.profile == "paper" {
                2000.0
            } else {
                160.0
            }),
        },
        budget: MemoryBudget {
            compute_bytes: mib(args.compute_budget_mib)?,
            host_working_bytes: mib(args.host_working_mib)?,
            compute_reserve_bytes: mib(args.compute_reserve_mib)?,
            max_batch_rows: args.max_batch_rows,
            ..MemoryBudget::default()
        },
    };
    let plans = quadrature_stages(config.quadrature)?
        .map(|quadrature| {
            plan_heston(
                &request,
                &HestonJobConfig {
                    quadrature,
                    ..config.clone()
                },
            )
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let input_hash = format!("{:x}", Sha256::digest(serde_json::to_vec(&request)?));
    if args.plan_only {
        let report = serde_json::json!({"schema":"volt.compute.heston_admission.v1","config":config,"plans":plans,"input_sha256":input_hash});
        if let Some(path) = args.output {
            write_json(&path, &report)?;
        }
        serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
        return Ok(());
    }
    let checked = run_checked_heston(&request, &config)?;
    let final_stage = checked.stages.last().ok_or("missing final Heston stage")?;
    let mut reference_checks = Vec::new();
    if !supplied {
        for (i, o) in request.options.iter().enumerate() {
            let mut references: Vec<(&str, f64)> = Vec::new();
            if args.profile == "paper" && (o.strike - 100.0).abs() < f64::EPSILON {
                let call = matches!(o.side, OptionSide::Call);
                references = vec![
                    ("price", 1.691833925110386),
                    ("delta", 0.6960070233948596 - if call { 0.0 } else { 1.0 }),
                    ("gamma", 0.11217017223716684),
                    ("initial_variance", 28.449951907626616),
                    ("kappa", 0.0303899056727107),
                    ("long_run_variance", 1.548386508970144),
                    ("vol_of_variance", -0.3654360637103979),
                    ("correlation", 0.2687017483712485),
                ];
            } else if args.profile == "lewis" {
                let k = ((o.strike - 80.0) / 10.0) as usize;
                let prices = if matches!(o.side, OptionSide::Call) {
                    [
                        26.774758743998854,
                        20.933_349_000_596_71,
                        16.070154917028834,
                        12.132211516709845,
                        9.024913483457836,
                    ]
                } else {
                    [
                        7.958878113256768,
                        12.017966707346305,
                        17.055_270_961_270_11,
                        23.017_825_898_442_8,
                        29.811026202682472,
                    ]
                };
                references.push(("price", prices[k]));
            }
            for (name, reference) in references {
                let observed = metrics(&final_stage.values)
                    .into_iter()
                    .find(|(n, _)| *n == name)
                    .ok_or("missing metric")?
                    .1[i];
                let tolerance = if config.precision == Precision::F64 {
                    3e-8 + 1e-9 * reference.abs()
                } else {
                    3e-4 + 5e-5 * reference.abs()
                };
                reference_checks.push(serde_json::json!({"row_id":o.row_id,"metric":name,"reference":reference,"observed":observed,
                    "absolute_error":(observed-reference).abs(),"tolerance":tolerance,"passed":(observed-reference).abs()<=tolerance}));
            }
        }
    }
    let mut bound_checks = Vec::with_capacity(request.options.len());
    for (i, o) in request.options.iter().enumerate() {
        let sd = o.spot * (-o.dividend_yield * o.maturity_years).exp();
        let kd = o.strike * (-o.rate * o.maturity_years).exp();
        let (lower, upper) = if matches!(o.side, OptionSide::Call) {
            ((sd - kd).max(0.0), sd)
        } else {
            ((kd - sd).max(0.0), kd)
        };
        let tolerance = if config.precision == Precision::F64 {
            1e-8 + 1e-9 * upper
        } else {
            3e-4 + 5e-5 * upper
        };
        let price = final_stage.values.prices[i];
        let gamma = final_stage.values.gamma[i];
        bound_checks.push(serde_json::json!({"row_id":o.row_id,"lower_price":lower,"upper_price":upper,"price":price,
            "gamma":gamma,"passed":price>=lower-tolerance&&price<=upper+tolerance&&gamma>=-checked.absolute_tolerance}));
    }
    let passed = checked.passed
        && reference_checks.iter().all(|v| v["passed"] == true)
        && bound_checks.iter().all(|v| v["passed"] == true);
    if let Some(path) = args.greeks_output {
        create_parent(&path)?;
        let mut writer = BufWriter::new(File::create(path)?);
        for (i, (option, position)) in request.options.iter().zip(&request.positions).enumerate() {
            let mut values = serde_json::Map::new();
            for (name, array) in metrics(&final_stage.values) {
                values.insert(name.into(), serde_json::json!(array[i]));
            }
            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({"schema":"volt.compute.heston_greeks.v1","input_sha256":input_hash,
                "row_id":option.row_id,"contract_id":position.contract_id,"quantity":position.quantity,"multiplier":position.multiplier,
                "currency":request.currency,"greeks_per_unit":values}),
            )?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
    }
    let executable_sha256 = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .map(|b| format!("{:x}", Sha256::digest(b)));
    let report = serde_json::json!({"schema":"volt.compute.heston_run.v1","config":config,"input_sha256":input_hash,
        "profile":if supplied{"numeric_request"}else{&args.profile},"currency":request.currency,
        "executable_sha256":executable_sha256,"lockfile_sha256":format!("{:x}",Sha256::digest(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock")))),
        "os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"fusion_feature_enabled":cfg!(feature="fusion"),
        "build_profile":if cfg!(debug_assertions){"debug"}else{"release"},"checked":checked,"external_reference_checks":reference_checks,
        "reference_sources":["https://github.com/lballabio/QuantLib/blob/e96579fd471b2e9f6bc974c96674d1ccfff248fa/test-suite/hestonmodel.cpp",
            "QuantLib1.40 adaptive Gatheral/Lobatto, cross-checked exp-sinh, finite-difference Richardson controls"],
        "no_arbitrage_checks":bound_checks,"passed":passed,
        "units":"per underlying unit; v0/longrun sensitivities per unit variance; theta calendar decay/year; other annual fractions/correlation per1.0",
        "scope":"numerical European Heston risk, not canonical contract validation, path simulation, or paper replication",
        "timing":"each stage includes planning/setup/upload/reverse/Fourier gamma/readback/stable weighted aggregation; refinement comparison excluded"});
    if let Some(path) = args.output {
        write_json(&path, &report)?;
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    if !passed {
        return Err("Heston refinement/reference validation failed".into());
    }
    Ok(())
}
fn resolved(path: &Path) -> std::io::Result<PathBuf> {
    path.canonicalize().or_else(|_| std::path::absolute(path))
}
fn create_parent(path: &Path) -> std::io::Result<()> {
    if let Some(p) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p)?;
    }
    Ok(())
}
fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    create_parent(path)?;
    let mut w = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut w, value)?;
    w.write_all(b"\n")?;
    w.flush()?;
    Ok(())
}
