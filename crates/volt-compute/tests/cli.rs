use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
static NEXT_SCRATCH: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "volt-compute-cli-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn output_cannot_replace_the_input_file() {
    let scratch = Scratch::new();
    let input = scratch.0.join("input.json");
    let original = serde_json::to_vec(&volt_compute::benchmark::fixture(1)).unwrap();
    std::fs::write(&input, &original).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_volt-compute"))
        .args(["--chunked", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&input)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(std::fs::read(input).unwrap(), original);
}

#[test]
fn admission_does_not_need_the_requested_gpu() {
    let output = Command::new(env!("CARGO_BIN_EXE_volt-compute"))
        .args([
            "--chunked",
            "--plan-only",
            "--device",
            "cuda:0",
            "--rows",
            "257",
            "--max-batch-rows",
            "31",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema"], "volt.compute.admission_report.v1");
    assert_eq!(report["plan"]["chunks"], 9);
}

#[test]
fn chunked_cli_exports_ordered_identity_bound_marks() {
    let scratch = Scratch::new();
    let marks = scratch.0.join("marks.jsonl");
    let summary = scratch.0.join("summary.json");
    let output = Command::new(env!("CARGO_BIN_EXE_volt-compute"))
        .args([
            "--chunked",
            "--rows",
            "257",
            "--max-batch-rows",
            "31",
            "--marks-output",
        ])
        .arg(&marks)
        .arg("--output")
        .arg(&summary)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&std::fs::read(summary).unwrap()).unwrap();
    assert_eq!(report["stats"]["completed_chunks"], 9);
    assert_eq!(report["validation"]["passed"], true);
    let text = std::fs::read_to_string(marks).unwrap();
    let lines = text
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 257);
    assert_eq!(lines[0]["row_id"], "scenario-row-0");
    assert_eq!(lines[256]["row_id"], "scenario-row-256");
    assert_eq!(
        lines[256]["numeric_input_hash"],
        report["numeric_input_hash"]
    );
}

#[test]
fn volt_cli_preserves_portfolio_and_market_provenance() {
    use volt::{
        CanonicalPortfolio, ExerciseSchedule, OptionContract, OptionRight, Position, PriceMicros,
        Settlement, TimestampNs,
    };
    use volt_compute::volt_adapter::{
        ContractPricingInput, MARKET_INPUTS_SCHEMA, PortfolioMarketInputs, UnderlyingMarketInput,
        VALUATION_REQUEST_SCHEMA, VoltValuationRequest, YearBasis,
    };
    let as_of = TimestampNs(1_800_000_000_000_000_000);
    let contract = OptionContract::new(
        "C100",
        "XYZ",
        "USD",
        OptionRight::Call,
        PriceMicros(100_000_000),
        100,
        ExerciseSchedule::European {
            expiry: TimestampNs(as_of.0 + 365 * 86400 * 1_000_000_000),
        },
        Settlement::Physical,
    )
    .unwrap();
    let portfolio =
        CanonicalPortfolio::from_positions([Position::new(contract.clone(), 3).unwrap()]).unwrap();
    let expected_hash = portfolio.semantic_hash().unwrap();
    let request = VoltValuationRequest {
        schema_version: VALUATION_REQUEST_SCHEMA.into(),
        portfolio,
        market: PortfolioMarketInputs {
            schema_version: MARKET_INPUTS_SCHEMA.into(),
            snapshot_id: "cli-market".into(),
            as_of,
            max_observation_age_ns: 0,
            year_basis: YearBasis::Actual365Fixed,
            underlyings: vec![UnderlyingMarketInput {
                underlying: contract.underlying.clone(),
                observed_at: as_of,
                spot: PriceMicros(100_000_000),
            }],
            contracts: vec![ContractPricingInput {
                contract_id: contract.contract_id.clone(),
                observed_at: as_of,
                volatility: 0.2,
                rate: 0.05,
                dividend_yield: 0.0,
            }],
        },
    };
    let scratch = Scratch::new();
    let input = scratch.0.join("portfolio.json");
    serde_json::to_writer(std::fs::File::create(&input).unwrap(), &request).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_volt-compute"))
        .args([
            "--chunked",
            "--precision",
            "f64",
            "--max-batch-rows",
            "1",
            "--volt-input",
        ])
        .arg(input)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["volt_provenance"]["portfolio_hash"], expected_hash);
    assert_eq!(report["volt_provenance"]["snapshot_id"], "cli-market");
    assert_eq!(report["volt_provenance"]["currency"], "USD");
    assert!(
        (report["option_position_value"].as_f64().unwrap() - 300.0 * 10.450583572185565).abs()
            < 1e-9
    );
}

#[test]
fn aad_plan_uses_reverse_graph_model_without_opening_gpu() {
    let output = Command::new(env!("CARGO_BIN_EXE_aad_bench"))
        .args([
            "--plan-only",
            "--device",
            "cuda:999",
            "--rows",
            "257",
            "--max-batch-rows",
            "31",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema"], "volt.compute.aad_plan.v1");
    assert_eq!(report["plan"]["chunks"], 9);
    assert_eq!(report["plan"]["estimated_compute_bytes_per_row"], 1024);
}

#[test]
fn aad_cli_exports_ordered_validated_greeks_and_preserves_inputs() {
    let scratch = Scratch::new();
    let rows = volt_compute::benchmark::fixture(7);
    let input = scratch.0.join("input.json");
    let original = serde_json::to_vec(&rows).unwrap();
    std::fs::write(&input, &original).unwrap();
    let marks = scratch.0.join("greeks.jsonl");
    let report = scratch.0.join("report.json");
    let output = Command::new(env!("CARGO_BIN_EXE_aad_bench"))
        .args(["--input"])
        .arg(&input)
        .args([
            "--precision",
            "f64",
            "--repeats",
            "1",
            "--gamma",
            "--max-batch-rows",
            "3",
            "--output",
        ])
        .arg(&report)
        .arg("--greeks-output")
        .arg(&marks)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["dispatches"][0]["completed_chunks"], 3);
    let marks: Vec<Value> = std::fs::read_to_string(marks)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for (row, mark) in rows.iter().zip(marks) {
        assert_eq!(mark["row_id"], row.row_id);
        assert!(mark["delta"].is_number());
        assert!(mark["gamma"].is_number());
        assert!(mark["numeric_input_hash"].is_string());
    }
    let invalid = Command::new(env!("CARGO_BIN_EXE_aad_bench"))
        .arg("--input")
        .arg(&input)
        .arg("--greeks-output")
        .arg(&input)
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert_eq!(std::fs::read(input).unwrap(), original);
}

#[test]
fn monte_carlo_plan_accounts_for_expanded_pairs_without_device() {
    let output = Command::new(env!("CARGO_BIN_EXE_monte_carlo_bench"))
        .args([
            "--plan-only",
            "--device",
            "cuda:999",
            "--rows",
            "3",
            "--pairs",
            "1000",
            "--max-batch-rows",
            "31",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["plan"]["pairs_per_batch"], 5);
    assert_eq!(report["plan"]["expanded_outcomes_per_batch"], 30);
    assert_eq!(report["plan"]["total_option_outcomes"], 6000);
    assert_eq!(report["plan"]["batches"], 200);
}

#[test]
fn monte_carlo_cli_exports_identity_bound_estimates_and_preserves_inputs() {
    let scratch = Scratch::new();
    let mut rows = volt_compute::benchmark::fixture(2);
    for row in &mut rows {
        row.spot = 100.0;
        row.strike = 100.0;
        row.maturity_years = 1.0;
        row.volatility = 0.2;
    }
    let input = scratch.0.join("mc-input.json");
    let original = serde_json::to_vec(&rows).unwrap();
    std::fs::write(&input, &original).unwrap();
    let estimates = scratch.0.join("estimates.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_monte_carlo_bench"))
        .arg("--input")
        .arg(&input)
        .args([
            "--pairs",
            "2048",
            "--precision",
            "f64",
            "--max-batch-rows",
            "512",
            "--estimates-output",
        ])
        .arg(&estimates)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["dispatches"][0]["completed_batches"], 16);
    assert!(report["normal_f64_sha256"].is_string());
    let estimates: Vec<Value> = std::fs::read_to_string(estimates)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for (row, estimate) in rows.iter().zip(estimates) {
        assert_eq!(estimate["row_id"], row.row_id);
        assert_eq!(estimate["gamma"]["samples"], 2048);
        assert!(estimate["gamma"]["standard_error"].is_number());
    }
    let invalid = Command::new(env!("CARGO_BIN_EXE_monte_carlo_bench"))
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&input)
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert_eq!(std::fs::read(input).unwrap(), original);
}

#[test]
fn heston_cli_plans_all_refinements_offline_and_exports_validated_risk() {
    let plan = Command::new(env!("CARGO_BIN_EXE_heston_bench"))
        .args([
            "--plan-only",
            "--profile",
            "lewis",
            "--device",
            "cuda:999",
            "--rows",
            "5",
            "--max-batch-rows",
            "1024",
        ])
        .output()
        .unwrap();
    assert!(
        plan.status.success(),
        "{}",
        String::from_utf8_lossy(&plan.stderr)
    );
    let plan: Value = serde_json::from_slice(&plan.stdout).unwrap();
    assert_eq!(plan["plans"].as_array().unwrap().len(), 3);
    assert_eq!(plan["plans"][0]["chunk_rows"], 4);
    assert_eq!(plan["plans"][2]["chunk_rows"], 2);
    let scratch = Scratch::new();
    let greeks = scratch.0.join("heston-greeks.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_heston_bench"))
        .args(["--profile", "lewis", "--rows", "2", "--greeks-output"])
        .arg(&greeks)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["checked"]["checks"].as_array().unwrap().len(), 22);
    assert!(
        report["external_reference_checks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["passed"] == true)
    );
    let lines: Vec<Value> = std::fs::read_to_string(greeks)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1]["row_id"], "heston-1");
    assert!(lines[1]["greeks_per_unit"]["initial_variance"].is_number());
    assert!(lines[1]["input_sha256"].is_string());
}
