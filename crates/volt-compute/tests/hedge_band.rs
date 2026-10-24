use volt_compute::{
    hedge_application::{HedgeCostConfig, HedgePathBatch},
    hedge_band::{banded_holdings, select_no_trade_band},
    risk_objective::RiskObjective,
};

#[test]
fn projects_to_band_boundary_preserves_inventory_inside_and_resets_each_path() {
    let targets = vec![
        0.5, 1000.0, 0.55, 1100.0, -0.2, -200.0, -0.5, -1000.0, -0.55, -1100.0, 0.2, 200.0,
    ];
    let actual = banded_holdings(&targets, 2, 3, &[0.1, 200.0], 4096).unwrap();
    assert_eq!(
        actual,
        vec![
            0.4,
            800.0,
            0.45000000000000007,
            900.0,
            -0.1,
            0.0,
            -0.4,
            -800.0,
            -0.45000000000000007,
            -900.0,
            0.1,
            0.0
        ]
    );
    assert_eq!(
        banded_holdings(&targets, 2, 3, &[0.0, 0.0], 4096).unwrap(),
        targets
    );
    assert!(banded_holdings(&targets, 2, 3, &[0.1, 200.0], 8).is_err());
}

#[test]
fn validation_selection_can_prefer_no_trading_after_costs_and_is_tie_stable() {
    let batch = HedgePathBatch {
        paths: 2,
        steps: 2,
        hedge_count: 1,
        feature_count: 1,
        marks: vec![100.0; 6],
        features: vec![0.0; 4],
        payoffs: vec![0.0; 2],
        source_hash: "separate-validation".into(),
    };
    let targets = vec![0.5, -0.5, 0.5, -0.5];
    let grid = vec![vec![0.0], vec![0.5], vec![1.0]];
    let selected = select_no_trade_band(
        &batch,
        &targets,
        HedgeCostConfig {
            proportional: 0.01,
            terminal_proportional: 0.0,
        },
        0.0,
        RiskObjective::SquaredLoss,
        &grid,
        4096,
    )
    .unwrap();
    assert_eq!(selected.selected_candidate, 1);
    assert_eq!(selected.candidate_risks[1], 0.0);
    assert_eq!(selected.candidate_risks[2], 0.0);
    assert_eq!(selected.validation_source_hash, "separate-validation");
    let no_cost = select_no_trade_band(
        &batch,
        &targets,
        HedgeCostConfig::default(),
        0.0,
        RiskObjective::SquaredLoss,
        &grid,
        4096,
    )
    .unwrap();
    assert_eq!(no_cost.selected_candidate, 0);
}

#[test]
fn band_is_causal_and_rejects_invalid_widths_shapes_and_overflow() {
    let targets = [0.5, 0.7, -0.1];
    let before = banded_holdings(&targets, 1, 3, &[0.1], 4096).unwrap();
    let after = banded_holdings(&[0.5, 0.7, 1e6], 1, 3, &[0.1], 4096).unwrap();
    assert_eq!(&before[..2], &after[..2]);
    assert!(banded_holdings(&targets, 1, 3, &[-0.1], 4096).is_err());
    assert!(banded_holdings(&targets, 1, 3, &[f64::NAN], 4096).is_err());
    assert!(banded_holdings(&[f64::MAX], 1, 1, &[f64::MAX], 4096).is_err());
    assert!(banded_holdings(&targets, 2, 3, &[0.1], 4096).is_err());
}
