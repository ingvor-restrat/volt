use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
use volt_compute::{hedge_application::HedgeCostConfig, volatility_strategy::*};
fn small() -> StraddleStudyConfig {
    StraddleStudyConfig {
        paths: 16,
        batch_paths: 4,
        finest_steps: 12,
        hedge_steps: vec![3, 6, 12],
        audit_paths: 8,
        ..Default::default()
    }
}
fn close(a: f64, b: f64, atol: f64) {
    assert!(
        (a - b).abs() <= atol,
        "{a:e} != {b:e}; gap={:e}",
        (a - b).abs()
    );
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "volt-straddle-control-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn path(&self, n: &str) -> PathBuf {
        self.0.join(n)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
fn quotes_payoff_units_and_greek_attribution_match_independent_controls() {
    // Independent ATM zero-carry closed form: 2S(2Phi(sigma sqrt(T)/2)-1).
    let s = 100.0;
    let sigma = 0.2;
    let t = 0.25;
    let d = 0.5 * sigma * f64::sqrt(t);
    let normal = |x: f64| 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
    let density = (-0.5 * d * d).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let q = quote_straddle(s, s, t, sigma).unwrap();
    close(q.price, 2.0 * s * (2.0 * normal(d) - 1.0), 5e-14);
    close(q.delta, 2.0 * normal(d) - 1.0, 1e-15);
    close(q.gamma, 2.0 * density / (s * sigma * t.sqrt()), 1e-15);
    close(q.theta, -s * density * sigma / t.sqrt(), 1e-13);
    close(q.vega, 2.0 * s * density * t.sqrt(), 1e-13);
    let c = small();
    let fine: Vec<_> = (0..=c.finest_steps)
        .map(|i| 100.0 + i as f64 * 0.25)
        .collect();
    let result = evaluate_straddle_path(&c, 0, &fine, 6, StraddleHedge::Delta).unwrap();
    close(
        result.option_mark_gain,
        (103.0_f64 - 100.0).abs() - q.price,
        1e-13,
    );
    close(
        result.pnl,
        result.theta_approximation + result.gamma_approximation + result.higher_order_residual
            - result.transaction_cost,
        1e-12,
    );
    close(result.inventory_tracking_gain, 0.0, 1e-15);
    assert!(result.higher_order_residual.abs() > 1e-8);
    assert!(quote_straddle(s, s, 0.0, sigma).is_err());
}
#[test]
fn signed_positions_and_multiplier_apply_once_and_real_fees_break_antisymmetry_correctly() {
    let c = small();
    let fine = fine_gbm_path(&c, 0).unwrap();
    let unit = evaluate_straddle_path(&c, 0, &fine, 12, StraddleHedge::Delta).unwrap();
    let large = StraddleStudyConfig {
        quantity: 10.0,
        multiplier: 100,
        ..c.clone()
    };
    let big = evaluate_straddle_path(&large, 0, &fine, 12, StraddleHedge::Delta).unwrap();
    close(big.pnl, 1000.0 * unit.pnl, 1e-10);
    close(big.turnover, 1000.0 * unit.turnover, 1e-9);
    let short = StraddleStudyConfig {
        quantity: -1.0,
        ..c.clone()
    };
    let opposite = evaluate_straddle_path(&short, 0, &fine, 12, StraddleHedge::Delta).unwrap();
    close(unit.pnl, -opposite.pnl, 0.0);
    let costs = HedgeCostConfig {
        proportional: 0.00013,
        terminal_proportional: 0.00047,
    };
    let charged = evaluate_straddle_path(
        &StraddleStudyConfig {
            cost: costs,
            ..c.clone()
        },
        0,
        &fine,
        12,
        StraddleHedge::Delta,
    )
    .unwrap();
    let negative = evaluate_straddle_path(
        &StraddleStudyConfig {
            cost: costs,
            ..short
        },
        0,
        &fine,
        12,
        StraddleHedge::Delta,
    )
    .unwrap();
    close(
        charged.pnl + negative.pnl,
        -charged.transaction_cost - negative.transaction_cost,
        1e-13,
    );
    close(
        charged.transaction_cost,
        costs.proportional * (unit.turnover - unit.terminal_turnover)
            + costs.terminal_proportional * unit.terminal_turnover,
        1e-13,
    );
    assert!(charged.terminal_turnover > 0.0);
    close(charged.terminal_cash, charged.pnl, 1e-13);
    let nohedge = evaluate_straddle_path(&c, 0, &fine, 12, StraddleHedge::NoHedge).unwrap();
    close(nohedge.transaction_cost, 0.0, 0.0);
    close(nohedge.turnover, 0.0, 0.0);
    close(nohedge.pnl, nohedge.option_mark_gain, 0.0);
}
#[test]
fn shared_fine_market_identity_is_partition_independent_and_ignores_quote_and_inventory() {
    let c = small();
    let first = run_straddle_study(&c, None).unwrap();
    let mut other = c.clone();
    other.batch_paths = 7;
    other.quantity = -3.5;
    other.multiplier = 2;
    other.pricing_volatility = 0.31;
    other.cost.proportional = 0.00017;
    other.hedge_steps = vec![12, 3];
    let second = run_straddle_study(&other, None).unwrap();
    assert_eq!(
        first.physical_market_recipe_sha256,
        second.physical_market_recipe_sha256
    );
    assert_eq!(
        first.physical_market_stream_sha256,
        second.physical_market_stream_sha256
    );
    assert_ne!(first.recipe_sha256, second.recipe_sha256);
    for a in &first.cases {
        close(
            a.mean_realized_variance,
            first.cases[0].mean_realized_variance,
            0.0,
        );
    }
    let fine = fine_gbm_path(&c, 0).unwrap();
    for steps in [3, 6, 12] {
        let r = evaluate_straddle_path(&c, 0, &fine, steps, StraddleHedge::Delta).unwrap();
        assert_eq!(r.terminal_spot.to_bits(), fine[12].to_bits());
    }
    other.physical_volatility = 0.21;
    let changed = run_straddle_study(&other, None).unwrap();
    assert_ne!(
        first.physical_market_stream_sha256,
        changed.physical_market_stream_sha256
    );
}
#[test]
fn ledger_audits_reconcile_signed_quantity_non_ppm_fees_and_band_positions() {
    for quantity in [1.25, -1.25] {
        let c = StraddleStudyConfig {
            quantity,
            multiplier: 100,
            cost: HedgeCostConfig {
                proportional: 1.0 / 512.0,
                terminal_proportional: 1.0 / 1024.0,
            },
            band_width: Some(0.25),
            ..small()
        };
        let report = run_straddle_study(&c, None).unwrap();
        assert_eq!(report.cases.len(), 9);
        for case in report.cases {
            assert_eq!(case.audits.len(), 8);
            assert!(
                case.max_cash_identity_error < 1e-8 && case.max_attribution_identity_error < 1e-8
            );
            for audit in case.audits {
                assert!(
                    audit.exact_reconciliation
                        && audit.float_identity_within_bound
                        && audit.quantization_within_bound
                );
                assert!(audit.quantization_gap.abs() <= audit.quantization_error_bound_currency);
                close(audit.ledger_fee, 0.001953, 0.0);
                close(audit.requested_fee, 1.0 / 512.0, 0.0);
            }
        }
    }
}
#[test]
fn matched_and_distinct_sigmas_have_analytic_mean_controls_with_declared_sampling_error() {
    // Fixed seed and fixed six-standard-error diagnostic, not a profitability
    // acceptance condition. Adapted stock gains have zero expectation at r=q=0.
    for physical in [0.0, 0.2, 0.35] {
        let c = StraddleStudyConfig {
            paths: 8192,
            batch_paths: 128,
            finest_steps: 12,
            hedge_steps: vec![3, 12],
            physical_volatility: physical,
            audit_paths: 0,
            ..Default::default()
        };
        let report = run_straddle_study(&c, None).unwrap();
        for case in report.cases {
            let error = (case.mean_profit - case.analytic_expected_profit_before_costs).abs();
            assert!(
                error <= 6.0 * case.standard_error + 1e-11,
                "sigmaP={physical},hedge={:?}:error={error},SE={}",
                case.strategy,
                case.standard_error
            );
            if physical == 0.2 {
                close(case.analytic_expected_profit_before_costs, 0.0, 0.0);
            }
            if physical == 0.35 {
                assert!(case.analytic_expected_profit_before_costs > 0.0);
            }
            if physical == 0.0 {
                assert!(case.analytic_expected_profit_before_costs < 0.0);
            }
        }
    }
}
#[test]
fn admission_and_atomic_samples_reject_invalid_inputs_without_published_partial_files() {
    let c = small();
    let temp = Temp::new();
    let output = temp.path("samples.jsonl");
    let report = run_straddle_study(&c, Some(&output)).unwrap();
    let artifact = report.samples.unwrap();
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(artifact.rows, (c.paths * 6) as u64);
    assert_eq!(artifact.bytes, bytes.len() as u64);
    assert_eq!(artifact.sha256, format!("{:x}", Sha256::digest(&bytes)));
    assert!(run_straddle_study(&c, Some(&output)).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), bytes);
    let mut bad = c.clone();
    bad.max_working_bytes = 1;
    assert!(run_straddle_study(&bad, Some(&temp.path("never.jsonl"))).is_err());
    bad = c.clone();
    bad.first_path_id = u64::MAX;
    assert!(bad.validate().is_err());
    bad = c.clone();
    bad.hedge_steps = vec![5];
    assert!(bad.validate().is_err());
    bad = c.clone();
    bad.hedge_steps = vec![3, 3];
    assert!(bad.validate().is_err());
    // Deterministic physical overflow happens only during path generation: temp
    // exists internally but Drop removes it and the destination stays absent.
    bad = c.clone();
    bad.initial_spot = 1e-300;
    bad.strike = 1e-300;
    bad.physical_volatility = 1e200;
    let failed = temp.path("failed.jsonl");
    assert!(run_straddle_study(&bad, Some(&failed)).is_err());
    assert!(!failed.exists());
    assert!(
        std::fs::read_dir(&temp.0).unwrap().all(|p| !p
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("partial"))
    );
}
#[test]
fn public_cli_json_request_plan_study_and_replay_output_roundtrip() {
    let temp = Temp::new();
    let input = temp.path("request.json");
    let report = temp.path("report.json");
    let samples = temp.path("samples.jsonl");
    std::fs::write(&input, serde_json::to_vec(&small()).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_volatility_strategy"))
        .args(["--config"])
        .arg(&input)
        .args(["--output"])
        .arg(&report)
        .args(["--samples-output"])
        .arg(&samples)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(report).unwrap()).unwrap();
    assert_eq!(json["schema"], "volt.compute.straddle_report.v1");
    assert_eq!(json["cases"].as_array().unwrap().len(), 6);
    assert_eq!(
        std::fs::read(samples)
            .unwrap()
            .split(|b| *b == b'\n')
            .filter(|x| !x.is_empty())
            .count(),
        96
    );
    let example = Command::new(env!("CARGO_BIN_EXE_volatility_strategy"))
        .arg("--example-config")
        .output()
        .unwrap();
    assert!(example.status.success());
    let example: StraddleStudyConfig = serde_json::from_slice(&example.stdout).unwrap();
    example.validate().unwrap();
}
