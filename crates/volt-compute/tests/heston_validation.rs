use volt_compute::{
    ExecutionTarget, OptionSide, Precision,
    budget::MemoryBudget,
    heston::{HestonOption, HestonParams, QuadratureConfig},
    heston_job::{HestonJobConfig, HestonJobRequest, HestonPosition},
    heston_validation::{quadrature_stages, run_checked_heston},
};
fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}
fn request() -> HestonJobRequest {
    HestonJobRequest {
        schema: "volt.compute.heston_request.v1".into(),
        currency: "USD".into(),
        options: vec![HestonOption {
            row_id: "call".into(),
            side: OptionSide::Call,
            spot: 100.0,
            strike: 100.0,
            maturity_years: 1.0,
            rate: 0.01,
            dividend_yield: 0.02,
        }],
        params: vec![HestonParams {
            initial_variance: 0.04,
            kappa: 4.0,
            long_run_variance: 0.25,
            vol_of_variance: 1.0,
            correlation: -0.5,
        }],
        positions: vec![HestonPosition {
            row_id: "call".into(),
            contract_id: "synthetic-call".into(),
            quantity: 1,
            multiplier: 100,
        }],
    }
}
#[test]
fn refinement_checks_both_directions_and_all_risk_outputs() {
    let config = HestonJobConfig {
        target: target(),
        precision: Precision::F32,
        quadrature: QuadratureConfig {
            nodes: 256,
            upper_bound: 160.0,
        },
        budget: MemoryBudget::default(),
    };
    let result = run_checked_heston(&request(), &config).unwrap();
    assert!(result.passed, "{:?}", result.checks);
    assert_eq!(result.stages.len(), 3);
    assert_eq!(result.checks.len(), 22);
    assert_eq!(
        result
            .checks
            .iter()
            .filter(|c| c.direction == "nodes")
            .count(),
        11
    );
    assert_eq!(
        result
            .checks
            .iter()
            .filter(|c| c.direction == "cutoff")
            .count(),
        11
    );
    assert!((result.stages[2].values.prices[0] - 16.070154917028834).abs() < 0.0002);
}
#[test]
fn inadequate_cutoff_is_reported_without_claiming_accuracy() {
    let config = HestonJobConfig {
        target: target(),
        precision: Precision::F32,
        quadrature: QuadratureConfig {
            nodes: 16,
            upper_bound: 1.0,
        },
        budget: MemoryBudget::default(),
    };
    let result = run_checked_heston(&request(), &config).unwrap();
    assert!(!result.passed);
    assert!(
        result
            .checks
            .iter()
            .any(|c| c.direction == "cutoff" && c.mismatched_rows > 0)
    );
}
#[test]
fn all_refinement_admissions_are_checked_before_opening_a_device() {
    let config = HestonJobConfig {
        target: ExecutionTarget::Cuda { index: 999 },
        precision: Precision::F32,
        quadrature: QuadratureConfig {
            nodes: 256,
            upper_bound: 160.0,
        },
        budget: MemoryBudget {
            max_batch_rows: 256,
            ..MemoryBudget::default()
        },
    };
    let error = run_checked_heston(&request(), &config)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cannot admit"), "{error}");
    assert!(
        quadrature_stages(QuadratureConfig {
            nodes: 2048,
            upper_bound: 160.0
        })
        .is_err()
    );
    assert!(
        quadrature_stages(QuadratureConfig {
            nodes: 256,
            upper_bound: 6_000_000.0
        })
        .is_err()
    );
}
