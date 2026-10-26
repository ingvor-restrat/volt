use volt_compute::hedge_lot_qubo::*;
fn fixture() -> HedgeLotProblem {
    let context = RiskContext {
        underlying_id: "S".into(),
        currency: "USD".into(),
        generation: 3,
        as_of_ns: 100,
        max_quote_age_ns: 10,
        greek_convention: "fixed_volatility".into(),
        provenance: "Synthetic quoted-risk tests".into(),
    };
    let stock = QuotedInstrument {
        instrument_id: "stock".into(),
        quote_id: "q-stock".into(),
        underlying_id: "S".into(),
        currency: "USD".into(),
        generation: 3,
        observed_at_ns: 99,
        greek_convention: "fixed_volatility".into(),
        eligible: true,
        mark: 100.,
        per_unit_risk: RiskVector {
            delta: 1.,
            gamma: 0.,
            vega_per_vol_point: 0.,
        },
        multiplier: 1,
        current_contracts: 0,
        min_contracts: -20,
        max_contracts: 20,
        proportional_fee: 0.001,
        fixed_fee_per_selected_lot: 0.5,
    };
    let option = QuotedInstrument {
        instrument_id: "call".into(),
        quote_id: "q-call".into(),
        mark: 5.,
        multiplier: 100,
        per_unit_risk: RiskVector {
            delta: 0.5,
            gamma: 0.05,
            vega_per_vol_point: 0.2,
        },
        min_contracts: 0,
        max_contracts: 2,
        proportional_fee: 0.02,
        fixed_fee_per_selected_lot: 1.,
        ..stock.clone()
    };
    HedgeLotProblem {
        context,
        instruments: vec![stock, option],
        lots: vec![
            SignedLot {
                lot_id: "stock-buy10".into(),
                instrument: 0,
                contracts: 10,
            },
            SignedLot {
                lot_id: "stock-sell10".into(),
                instrument: 0,
                contracts: -10,
            },
            SignedLot {
                lot_id: "call-buy1".into(),
                instrument: 1,
                contracts: 1,
            },
            SignedLot {
                lot_id: "call-buy2".into(),
                instrument: 1,
                contracts: 2,
            },
        ],
        current_risk: RiskVector {
            delta: -100.,
            gamma: -10.,
            vega_per_vol_point: -50.,
        },
        target_risk: RiskVector {
            delta: 0.,
            gamma: 0.,
            vega_per_vol_point: 0.,
        },
        risk_scales: [100., 10., 50.],
        risk_weights: [1.; 3],
        fee_weight: 0.001,
        cash_before: 2000.,
        cash_borrowing_limit: 0.,
    }
}
fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} vs {b}");
}
#[test]
fn qubo_matches_explicit_risk_and_actual_fees_for_every_mask() {
    let p = fixture();
    let q = build(&p).unwrap();
    assert!(!q.feasibility_constraints_encoded);
    for mask in 0..16 {
        let e = evaluate(&p, mask).unwrap();
        near(e.objective, q.value(mask).unwrap());
    }
    let e = evaluate(&p, 8).unwrap();
    near(e.fees, 21.);
    near(e.cash_after, 979.);
    near(e.residual_risk.delta, 0.);
    near(e.residual_risk.gamma, 0.);
    near(e.residual_risk.vega_per_vol_point, -10.);
    near(e.objective, 0.061);
}
#[test]
fn exact_solution_respects_cash_inventory_and_single_choice() {
    let p = fixture();
    let x = solve(&p).unwrap();
    assert_eq!(x.enumerated_masks, 16);
    assert_eq!(x.selected.mask, 8);
    assert!(x.selected.feasible);
    assert_eq!(x.selected.final_contracts, vec![0, 2]);
    assert!(!evaluate(&p, 12).unwrap().feasible);
    assert!(!evaluate(&p, 3).unwrap().feasible);
    let mut limited = p.clone();
    limited.cash_before = 500.;
    let y = solve(&limited).unwrap();
    assert!(y.selected.cash_after >= 0.);
    assert_ne!(y.selected.mask, 8);
}
#[test]
fn signed_lot_multiplier_and_cash_execution_are_financial() {
    let mut p = fixture();
    p.cash_before = 10.;
    p.lots[1].contracts = -20;
    p.instruments[0].min_contracts = -20;
    let e = evaluate(&p, 10).unwrap();
    assert!(e.feasible);
    assert_eq!(e.execution[0].instrument_id, "stock");
    assert!(e.execution.iter().all(|x| x.cash_after >= 0.));
    assert_eq!(e.execution[0].contracts, -20);
    assert_eq!(e.execution[1].underlying_units, 200.);
    near(e.execution[0].fee, 2.5);
    near(e.execution[1].fee, 21.);
    near(e.cash_after, 986.5);
}
#[test]
fn quote_generation_time_and_units_are_checked() {
    let p = fixture();
    for mode in 0..6 {
        let mut bad = p.clone();
        match mode {
            0 => bad.instruments[0].generation += 1,
            1 => bad.instruments[0].currency = "EUR".into(),
            2 => bad.instruments[0].observed_at_ns = 101,
            3 => bad.instruments[0].observed_at_ns = 0,
            4 => bad.instruments[0].greek_convention = "sticky".into(),
            _ => bad.instruments[0].eligible = false,
        };
        assert!(build(&bad).is_err());
    }
}
#[test]
fn overflow_invalid_dimensions_and_masks_fail_closed() {
    let mut p = fixture();
    p.lots.resize(17, p.lots[0].clone());
    assert!(solve(&p).is_err());
    let p = fixture();
    assert!(evaluate(&p, 16).is_err());
    let mut p = fixture();
    p.risk_scales[0] = 0.;
    assert!(build(&p).is_err());
    let mut p = fixture();
    p.instruments[0].mark = f64::MAX;
    assert!(build(&p).is_err());
    let mut p = fixture();
    p.instruments[1].multiplier = 0;
    assert!(build(&p).is_err());
}
#[test]
fn cli_native_greek_example_is_a_finite_executable_proposal() {
    use std::{io::Write, process::Command};
    let exe = env!("CARGO_BIN_EXE_hedge_lot_qubo");
    let out = Command::new(exe).arg("--example").output().unwrap();
    assert!(out.status.success());
    let p: HedgeLotProblem = serde_json::from_slice(&out.stdout).unwrap();
    let result = solve(&p).unwrap();
    assert_eq!(result.selected.mask, 129);
    assert!(result.selected.feasible);
    near(result.selected.post_trade_risk.delta, 0.);
    near(result.selected.post_trade_risk.gamma, 0.);
    near(result.selected.post_trade_risk.vega_per_vol_point, 0.);
    assert_eq!(result.selected.execution[0].instrument_id, "stock");
    assert_eq!(result.selected.final_contracts, vec![-200, 2]);
    assert_eq!(result.enumerated_masks, 256);
    let file = std::env::temp_dir().join(format!("volt-lot-qubo-{}.json", std::process::id()));
    let mut f = std::fs::File::create(&file).unwrap();
    f.write_all(&out.stdout).unwrap();
    drop(f);
    let cli = Command::new(exe)
        .args(["--input", file.to_str().unwrap()])
        .output()
        .unwrap();
    std::fs::remove_file(file).unwrap();
    assert!(cli.status.success());
    let decoded: LotSolution = serde_json::from_slice(&cli.stdout).unwrap();
    assert_eq!(decoded.selected.mask, 129);
}
#[test]
fn arbitrary_targets_scales_and_weights_match_the_expansion() {
    let mut p = fixture();
    p.target_risk = RiskVector {
        delta: 5.,
        gamma: -1.,
        vega_per_vol_point: 0.3,
    };
    p.risk_weights = [2., 0.5, 3.];
    p.risk_scales = [20., 4., 2.];
    p.fee_weight = 0.02;
    let mut q = build(&p).unwrap();
    for mask in 0..16 {
        near(
            q.value(mask).unwrap(),
            evaluate(&p, mask).unwrap().objective,
        );
    }
    q.upper[1][0] = 1.;
    assert!(q.value(0).is_err());
}
#[test]
fn sixteen_bit_limit_is_actually_enumerated() {
    let mut p = fixture();
    p.instruments.truncate(1);
    p.instruments[0].mark = 1.;
    p.instruments[0].proportional_fee = 0.;
    p.instruments[0].fixed_fee_per_selected_lot = 0.;
    p.lots = (1..=16)
        .map(|q| SignedLot {
            lot_id: format!("buy:{q}"),
            instrument: 0,
            contracts: q,
        })
        .collect();
    p.current_risk = RiskVector {
        delta: -16.,
        gamma: 0.,
        vega_per_vol_point: 0.,
    };
    p.risk_scales = [1.; 3];
    p.risk_weights = [1., 0., 0.];
    p.fee_weight = 0.;
    let s = solve(&p).unwrap();
    assert_eq!(s.enumerated_masks, 65536);
    assert_eq!(s.feasible_masks, 17);
    assert_eq!(s.selected.mask, 1 << 15);
    near(s.selected.objective, 0.);
}
