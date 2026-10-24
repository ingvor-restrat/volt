use volt_compute::{
    market_path_diagnostics::{
        PathDiagnosticConfig, conditional_drift, diagnose_paths, plan_path_diagnostics,
    },
    market_paths::{
        FUNDED_VARIANCE_CLAIM, IntegratedVarianceRule, MarketModel, MarketPathConfig, STOCK,
        generate_paths,
    },
    monte_carlo::OnlineMoments,
};

fn config(paths: usize) -> PathDiagnosticConfig {
    let mut market = MarketPathConfig::paper_heston(paths, 91);
    market.steps = 7;
    PathDiagnosticConfig {
        market,
        call_strike: 100.0,
        requested_chunk_paths: 13,
        host_working_bytes: 64 * 1024 * 1024,
        call_reference: None,
        reference_source: None,
    }
}

#[test]
fn million_path_plan_is_streamed_under64mib_and_bad_budget_fails_offline() {
    let mut config = config(1_000_000);
    config.market.steps = 30;
    config.requested_chunk_paths = 32768;
    let plan = plan_path_diagnostics(&config).unwrap();
    assert_eq!(plan.chunk_paths, 32768);
    assert_eq!(plan.chunks, 31);
    assert!(plan.estimated_working_bytes < 64 * 1024 * 1024);
    assert!(plan.estimated_working_bytes <= plan.host_admission_limit_bytes);
    config.host_working_bytes = 1;
    assert!(plan_path_diagnostics(&config).is_err());
    config.host_working_bytes = 64 * 1024 * 1024;
    config.call_reference = Some(1.0);
    assert!(plan_path_diagnostics(&config).is_err());
}

#[test]
fn chunk_sizes_preserve_independent_path_statistics_and_count() {
    let mut config = config(53);
    let chunked = diagnose_paths(&config).unwrap();
    config.requested_chunk_paths = 53;
    let full = diagnose_paths(&config).unwrap();
    assert_eq!(
        serde_json::to_value(&chunked.times).unwrap(),
        serde_json::to_value(&full.times).unwrap()
    );
    assert_eq!(
        chunked.terminal_discounted_call.mean,
        full.terminal_discounted_call.mean
    );
    assert_eq!(
        chunked.terminal_discounted_call.standard_error,
        full.terminal_discounted_call.standard_error
    );
    assert_eq!(chunked.terminal_discounted_call.samples, 53);
    assert_ne!(chunked.chunk_manifest_hash, full.chunk_manifest_hash);
}

#[test]
fn exact_conditional_mgf_matches_independent_path_means_and_exposes_claim_bias() {
    for rule in [
        IntegratedVarianceRule::Left,
        IntegratedVarianceRule::Trapezoid,
    ] {
        for current in [0.0, 0.04, 0.3] {
            let mut market = MarketPathConfig::paper_heston(20_000, 732);
            market.steps = 1;
            market.integrated_variance_rule = rule;
            if let MarketModel::Heston { ref mut params } = market.model {
                params.initial_variance = current;
            }
            let expected = conditional_drift(&market, current).unwrap();
            let paths = generate_paths(&market).unwrap();
            let mut stock = OnlineMoments::default();
            let mut claim = OnlineMoments::default();
            for path in 0..market.paths {
                stock.add(paths.value(path, 1, STOCK).unwrap() / 100.0 - 1.0);
                claim.add(
                    paths.value(path, 1, FUNDED_VARIANCE_CLAIM).unwrap()
                        - paths.value(path, 0, FUNDED_VARIANCE_CLAIM).unwrap(),
                );
            }
            for (moment, mean) in [
                (stock, expected.adjusted_stock_fractional_drift.unwrap()),
                (claim, expected.discounted_claim_absolute_drift),
            ] {
                assert!(
                    (moment.mean - mean).abs()
                        < 6.0 * moment.estimate().standard_error.unwrap() + 1e-12,
                    "{rule:?} V{current}: mean{} expected{mean}, SE{:?}",
                    moment.mean,
                    moment.estimate().standard_error
                );
            }
            if rule == IntegratedVarianceRule::Left && current == 0.0 {
                assert!(expected.discounted_claim_absolute_drift < 0.0);
            }
        }
    }
}

#[test]
fn variance_claim_reports_discrete_expectation_and_continuous_bias_separately() {
    let mut config = config(1000);
    config.market.steps = 3;
    config.market.dt = 0.25;
    if let MarketModel::Heston { ref mut params } = config.market.model {
        params.initial_variance = 0.3;
    }
    let report = diagnose_paths(&config).unwrap();
    let last = report.times.last().unwrap();
    assert_ne!(
        last.integrated_variance.reference,
        last.continuous_integrated_variance_reference
    );
    assert_ne!(
        last.discounted_variance_claim_discrete_expectation
            .reference,
        last.discounted_variance_claim_continuous_martingale_target
            .reference
    );
    assert!(report.scheme_disclosure.contains("genuine scheme bias"));
    if let MarketModel::Heston { ref mut params } = config.market.model {
        params.correlation = 0.0;
    }
    assert!(
        conditional_drift(&config.market, 0.3)
            .unwrap()
            .adjusted_stock_fractional_drift
            .unwrap()
            .abs()
            < 1e-15
    );
}
