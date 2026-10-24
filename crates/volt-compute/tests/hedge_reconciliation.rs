use volt_compute::{
    hedge_application::{HedgeCostConfig, HedgePathBatch},
    hedge_ledger::HedgeInstrumentKind,
    hedge_reconciliation::{AuditModelFamily, reconcile_path},
    model_hedge::scalar_hedge_pnl,
};

fn batch(paths: usize, steps: usize, hedges: usize) -> HedgePathBatch {
    let mut marks = Vec::new();
    for p in 0..paths {
        for t in 0..=steps {
            for h in 0..hedges {
                marks.push(if h % 2 == 0 {
                    100.0 + p as f64 * 10.0 + t as f64 * (h as f64 + 1.0)
                } else {
                    0.003_287_671_233 + (p + t + h) as f64 * 0.000_012_345_678
                });
            }
        }
    }
    HedgePathBatch {
        paths,
        steps,
        hedge_count: hedges,
        feature_count: hedges,
        marks,
        features: vec![0.0; paths * steps * hedges],
        payoffs: (0..paths).map(|p| 5.123_456_789 + p as f64).collect(),
        source_hash: "raw-zero-rate-test-batch".into(),
    }
}

#[test]
fn mixed_fractional_long_short_stock_and_variance_reconcile_and_report_quantization() {
    let batch = batch(2, 3, 4);
    let path0 = vec![
        -0.25,
        1_000.123_456_789,
        1.25,
        -700.000_000_5,
        -0.500_000_5,
        -200.222_222_2,
        0.0,
        450.0,
        0.100_000_5,
        0.0,
        -0.75,
        -1.5,
    ];
    let path1: Vec<_> = path0.iter().map(|h| -0.7 * h).collect();
    let mut all = path0.clone();
    all.extend(&path1);
    let cost = HedgeCostConfig {
        proportional: 1.0 / 512.0,
        terminal_proportional: 0.0,
    };
    let reference = scalar_hedge_pnl(&batch, &all, cost, 2.345_678_9).unwrap();
    for (p, holdings) in [(0, &path0), (1, &path1)] {
        let audit = reconcile_path(
            &batch,
            p,
            holdings,
            AuditModelFamily::Heston,
            1.0 / 365.0,
            2.345_678_9,
            cost,
        )
        .unwrap();
        assert!((audit.unquantized_float_pnl - reference.pnl[p]).abs() < 1e-12);
        assert!(
            audit.exact_reconciliation
                && audit.float_identity_within_bound
                && audit.quantization_within_bound
        );
        assert_eq!(
            audit.exact_ledger_pnl_picos,
            audit.exact_independent_identity_picos
        );
        assert!(audit.quantization_gap.abs() > 0.0);
        assert!(audit.quantization_gap.abs() <= audit.quantization_error_bound_currency);
        assert_eq!(audit.trading_fee.parts_per_million, 1953);
        assert!((audit.trading_fee.rate_residual - 0.000_000_125).abs() < 1e-18);
        assert_eq!(audit.terminal_fee.parts_per_million, 0);
        assert_eq!(audit.journal.len(), 4);
        assert_eq!(audit.journal[0].market.as_of.0, 0);
        assert_eq!(audit.journal[3].market.as_of.0, 3 * 86_400_000_000_000_i64);
        assert!(audit.final_state.settled);
        assert!(audit.final_state.holdings.values().all(|q| q.0 == 0));
        assert_eq!(audit.final_state.cumulative_interest_picos, 0);
        assert_eq!(
            audit.journal.last().unwrap().cost_model.parts_per_million,
            0
        );
        assert!(matches!(
            audit.ledger_config.instruments[1].kind,
            HedgeInstrumentKind::FundedVarianceClaim { .. }
        ));
        let json = serde_json::to_value(&audit).unwrap();
        assert!(json["exact_ledger_pnl_picos"].is_string());
        assert!(json["final_state"]["cash_picos"].is_string());
    }
}

#[test]
fn nearest_half_micro_ties_round_away_from_zero_in_all_declared_units() {
    let mut batch = batch(1, 2, 1);
    batch.marks = vec![0.000_000_5, 0.000_001_5, 0.000_002_5];
    batch.payoffs[0] = -0.000_000_5;
    let holdings = [-0.000_000_5, 0.000_001_5];
    let audit = reconcile_path(
        &batch,
        0,
        &holdings,
        AuditModelFamily::BlackScholes,
        1.0 / 365.0,
        0.000_000_5,
        HedgeCostConfig {
            proportional: 0.000_000_5,
            terminal_proportional: 0.0,
        },
    )
    .unwrap();
    assert_eq!(audit.initial_premium_micros, "1");
    assert_eq!(audit.terminal_liability_micros, "-1");
    assert_eq!(audit.rounded_holdings[0].0, -1);
    assert_eq!(audit.rounded_holdings[1].0, 2);
    assert_eq!(audit.journal[0].trades[0].price.0, 1);
    assert_eq!(audit.journal[1].trades[0].price.0, 2);
    assert_eq!(audit.journal[2].trades[0].price.0, 3);
    assert_eq!(audit.trading_fee.parts_per_million, 1);
    assert!(
        audit.exact_reconciliation
            && audit.float_identity_within_bound
            && audit.quantization_within_bound
    );
}

#[test]
fn terminal_cost_and_exact_ppm_are_explicit_and_picounit_fee_ties_are_bounded() {
    let mut batch = batch(1, 1, 1);
    batch.marks = vec![1.0, 1.0];
    batch.payoffs[0] = 0.0;
    let holdings = [0.000_001];
    let audit = reconcile_path(
        &batch,
        0,
        &holdings,
        AuditModelFamily::BlackScholes,
        1.0 / 365.0,
        0.0,
        HedgeCostConfig {
            proportional: 0.015625,
            terminal_proportional: 0.015625,
        },
    )
    .unwrap();
    assert_eq!(audit.trading_fee.parts_per_million, 15625);
    assert_eq!(audit.trading_fee.rate_residual, 0.0);
    assert_eq!(audit.journal[1].cost_model.parts_per_million, 15625);
    assert!(audit.final_state.cumulative_cost_picos > 0);
    assert!(audit.exact_reconciliation && audit.float_identity_within_bound);
    // One quantity micro times one price micro is one picounit. At50% fee,
    // exactly0.5picounit rounds up. Two independently rounded trades differ
    // from the unrounded rational identity by exactly one picounit.
    batch.marks = vec![0.000_001, 0.000_001];
    let tie = reconcile_path(
        &batch,
        0,
        &holdings,
        AuditModelFamily::BlackScholes,
        1.0 / 365.0,
        0.0,
        HedgeCostConfig {
            proportional: 0.5,
            terminal_proportional: 0.5,
        },
    )
    .unwrap();
    assert_eq!(tie.final_state.cumulative_cost_picos, 2);
    assert_eq!(tie.max_rounding_residual.max_fee_picounits, 0.5);
    assert_eq!(tie.fee_rounding_bound_picos, "1");
    assert!((tie.float_identity_difference_currency - 1e-12).abs() < 1e-27);
    assert!(
        tie.exact_reconciliation
            && tie.float_identity_within_bound
            && tie.quantization_within_bound
    );
}

#[test]
fn selected_path_identity_and_hash_change_with_actual_inputs() {
    let batch = batch(2, 2, 2);
    let holdings = [0.4, -1000.0, 0.6, 1200.0];
    let cost = HedgeCostConfig::default();
    let a = reconcile_path(
        &batch,
        0,
        &holdings,
        AuditModelFamily::Heston,
        1.0 / 365.0,
        2.0,
        cost,
    )
    .unwrap();
    let b = reconcile_path(
        &batch,
        1,
        &holdings,
        AuditModelFamily::Heston,
        1.0 / 365.0,
        2.0,
        cost,
    )
    .unwrap();
    assert_ne!(a.selected_input_hash, b.selected_input_hash);
    let mut changed = batch.clone();
    changed.marks[0] += 0.000_001;
    let c = reconcile_path(
        &changed,
        0,
        &holdings,
        AuditModelFamily::Heston,
        1.0 / 365.0,
        2.0,
        cost,
    )
    .unwrap();
    assert_ne!(a.selected_input_hash, c.selected_input_hash);
    assert_eq!(a.source_hash, c.source_hash); // Source declaration is not proof.
}

#[test]
fn malformed_shapes_quantization_overflows_and_subnanosecond_grids_are_rejected() {
    let batch = batch(1, 2, 2);
    let valid = [0.4, -1000.0, 0.6, 1200.0];
    let zero = HedgeCostConfig::default();
    let call = |batch: &HedgePathBatch, holdings: &[f64], dt, premium, cost| {
        reconcile_path(
            batch,
            0,
            holdings,
            AuditModelFamily::Heston,
            dt,
            premium,
            cost,
        )
    };
    assert!(
        reconcile_path(
            &batch,
            1,
            &valid,
            AuditModelFamily::Heston,
            1.0 / 365.0,
            0.0,
            zero
        )
        .is_err()
    );
    assert!(call(&batch, &valid[..3], 1.0 / 365.0, 0.0, zero).is_err());
    let mut bad = valid;
    bad[0] = f64::NAN;
    assert!(call(&batch, &bad, 1.0 / 365.0, 0.0, zero).is_err());
    bad = valid;
    bad[0] = 1e20;
    assert!(call(&batch, &bad, 1.0 / 365.0, 0.0, zero).is_err());
    assert!(call(&batch, &valid, 1e-20, 0.0, zero).is_err());
    assert!(call(&batch, &valid, 1e100, 0.0, zero).is_err());
    assert!(call(&batch, &valid, 1.0 / 365.0, 1e100, zero).is_err());
    assert!(
        call(
            &batch,
            &valid,
            1.0 / 365.0,
            0.0,
            HedgeCostConfig {
                proportional: 5000.0,
                terminal_proportional: 0.0
            }
        )
        .is_err()
    );
    let mut changed = batch.clone();
    changed.marks[0] = 0.000_000_1;
    assert!(call(&changed, &valid, 1.0 / 365.0, 0.0, zero).is_err());
    changed = batch.clone();
    changed.marks[1] = 1e30;
    assert!(call(&changed, &valid, 1.0 / 365.0, 0.0, zero).is_err());
    let odd = self::batch(1, 2, 3);
    assert!(
        reconcile_path(
            &odd,
            0,
            &[0.0; 6],
            AuditModelFamily::Heston,
            1.0 / 365.0,
            0.0,
            zero
        )
        .is_err()
    );
}

#[test]
fn generated_market_traces_reconcile_across_fractional_paper_fee_levels() {
    use volt_compute::{
        hedge_application::HedgeLiability,
        market_paths::{MarketPathConfig, generate_paths},
        monte_carlo::normal_at,
    };
    let mut config = MarketPathConfig::paper_heston(12, 142);
    config.steps = 5;
    let market = generate_paths(&config).unwrap();
    let batch =
        HedgePathBatch::from_markets(&[market], HedgeLiability::Call { strike: 100.0 }).unwrap();
    let per = batch.steps * batch.hedge_count;
    let holdings: Vec<_> = (0..batch.paths * per)
        .map(|i| normal_at(31, i as u64) * if i % 2 == 0 { 2.0 } else { 10_000.0 })
        .collect();
    for epsilon in [0.0, 1.0 / 8192.0, 1.0 / 128.0, 1.0 / 64.0] {
        let cost = HedgeCostConfig {
            proportional: epsilon,
            terminal_proportional: 0.0,
        };
        let scalar = scalar_hedge_pnl(&batch, &holdings, cost, 1.7).unwrap();
        for p in 0..batch.paths {
            let audit = reconcile_path(
                &batch,
                p,
                &holdings[p * per..(p + 1) * per],
                AuditModelFamily::Heston,
                config.dt,
                1.7,
                cost,
            )
            .unwrap();
            assert!(
                audit.exact_reconciliation
                    && audit.float_identity_within_bound
                    && audit.quantization_within_bound,
                "path{p} eps{epsilon} gap{} bound{}",
                audit.quantization_gap,
                audit.quantization_error_bound_currency
            );
            assert!((audit.unquantized_float_pnl - scalar.pnl[p]).abs() < 1e-10);
        }
    }
}
