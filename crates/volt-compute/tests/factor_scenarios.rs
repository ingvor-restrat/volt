use volt_compute::{
    ExecutionTarget, OptionSide, Precision,
    factor_scenarios::*,
    hedge_policy::HedgePolicyConfig,
    joint_option_market::ScenarioRange,
    native_learner::{Command, LearnerConfig, NativeLearner, Objective, Split},
};
fn fixture() -> (FactorModel, FactorConfig) {
    let m = FactorModel {
        asset_ids: vec!["A".into(), "B".into()],
        factor_ids: vec!["F".into()],
        initial_prices: vec![100., 80.],
        loadings: vec![vec![1.], vec![0.8]],
        annual_factor_covariance: vec![vec![0.0256]],
        annual_idiosyncratic_volatility: vec![0.12, 0.1],
        dividend_yields: vec![0.01, 0.02],
        annual_base_excess_return: vec![0.01, -0.01],
        annual_signal_coefficients: vec![0.2, -0.12],
        coefficient_uncertainty_std: vec![0.03, 0.02],
        signal_initial: 0.,
        signal_persistence: 0.95,
        signal_innovation_std: 0.3,
        provenance: FactorProvenance {
            model_id: "synthetic-test".into(),
            source_sha256: "a".repeat(64),
            synthetic: true,
            decision_origin_ns: None,
            known_at_ns: None,
            assumptions:
                "Invented test values; annual decimal returns and covariance; no data forecast"
                    .into(),
        },
    };
    let c = FactorConfig {
        seed: 731805,
        steps: 3,
        dt: 1. / 252.,
        rate: 0.03,
        liabilities: vec![FactorOption {
            asset: 0,
            side: OptionSide::Call,
            strike: 100.,
            maturity_steps: 3,
            quantity: 1,
            multiplier: 1,
        }],
        observation: FactorObservation::MarketAndSignal,
        trading_costs: vec![0.0005; 2],
        liquidation_costs: vec![0.; 2],
        max_working_bytes: 64 * 1024 * 1024,
    };
    (m, c)
}
fn range(first: u64, n: usize) -> ScenarioRange {
    ScenarioRange {
        split: Split::Train,
        first_path: first,
        paths: n,
    }
}
fn near(a: f64, b: f64, e: f64) {
    assert!((a - b).abs() <= e, "{a} versus {b}");
}
#[test]
fn covariance_and_supplied_shocks_use_annual_return_units() {
    let (m, c) = fixture();
    let risk = m.risk().unwrap();
    near(risk.annual_stock_covariance[0][0], 0.04, 1e-15);
    near(risk.annual_stock_covariance[0][1], 0.02048, 1e-15);
    near(risk.annual_stock_covariance[1][1], 0.026384, 1e-15);
    let b = [0.22, -0.1];
    let z = [0.5];
    let idio = [-0.2, 0.3];
    let out = advance_given_shocks(
        &m,
        &risk,
        &c,
        FactorStep {
            prices: &m.initial_prices,
            signal: 0.7,
            coefficient: &b,
            factor_normal: &z,
            idio_normal: &idio,
        },
    )
    .unwrap();
    for i in 0..2 {
        let sigma = risk.annual_stock_volatility[i];
        let shock = m.loadings[i][0] * 0.16 * z[0] + m.annual_idiosyncratic_volatility[i] * idio[i];
        let ret = (c.rate - m.dividend_yields[i] + m.annual_base_excess_return[i] + b[i] * 0.7
            - 0.5 * sigma * sigma)
            * c.dt
            + c.dt.sqrt() * shock;
        near(out[i], m.initial_prices[i] * ret.exp(), 1e-13);
    }
}
#[test]
fn singular_psd_factors_are_supported_but_indefinite_covariance_rejected() {
    let (mut m, _) = fixture();
    m.factor_ids = vec!["F1".into(), "F2".into()];
    m.loadings = vec![vec![1., 0.5], vec![0.8, -1.]];
    m.annual_factor_covariance = vec![vec![0.04, 0.04], vec![0.04, 0.04]];
    let r = m.risk().unwrap();
    near(r.annual_stock_covariance[0][0], 0.1044, 1e-14);
    near(r.annual_stock_covariance[1][1], 0.0116, 1e-14);
    m.annual_factor_covariance[1][1] = 0.03;
    assert!(m.validate().is_err());
}
#[test]
fn chunking_and_observation_masks_preserve_physical_paths() {
    let (m, mut c) = fixture();
    let all = generate(&m, &c, range(20, 4)).unwrap();
    let a = generate(&m, &c, range(20, 2)).unwrap();
    let b = generate(&m, &c, range(22, 2)).unwrap();
    assert_eq!(all.batch.marks, [a.batch.marks, b.batch.marks].concat());
    assert_eq!(
        all.batch.features,
        [a.batch.features, b.batch.features].concat()
    );
    assert_eq!(
        all.batch.provenance.dataset_id,
        a.batch.provenance.dataset_id
    );
    c.observation = FactorObservation::MarketOnly;
    let masked = generate(&m, &c, range(20, 4)).unwrap();
    assert_eq!(masked.physical_sha256, all.physical_sha256);
    assert_eq!(masked.batch.marks, all.batch.marks);
    assert_ne!(masked.content_sha256, all.content_sha256);
    for f in masked.batch.features.chunks(13) {
        assert_eq!(&f[8..], &[0.; 5]);
    }
}
#[test]
fn latent_uncertainty_is_excluded_from_initial_observations() {
    let (m, mut c) = fixture();
    let a = generate(&m, &c, range(0, 2)).unwrap();
    c.seed += 1;
    let b = generate(&m, &c, range(0, 2)).unwrap();
    assert_ne!(a.latent_coefficients, b.latent_coefficients);
    assert_eq!(&a.batch.features[..13], &b.batch.features[..13]);
    assert_eq!(
        &a.batch.features[3 * 13..4 * 13],
        &b.batch.features[3 * 13..4 * 13]
    );
}
#[test]
fn european_cashflow_and_zero_liability_trading_are_distinct() {
    let (m, mut c) = fixture();
    let hedged = generate(&m, &c, range(0, 2)).unwrap();
    for p in 0..2 {
        near(
            hedged.batch.payoffs[p],
            (hedged.batch.marks[(p * 4 + 3) * 2] - 100.).max(0.),
            1e-14,
        );
    }
    assert!(hedged.economics.premium > 0.);
    c.liabilities.clear();
    let trading = generate(&m, &c, range(0, 2)).unwrap();
    assert_eq!(hedged.physical_sha256, trading.physical_sha256);
    assert_eq!(trading.economics.premium, 0.);
    assert_eq!(trading.batch.payoffs, vec![0.; 2]);
    assert_ne!(
        hedged.batch.provenance.dataset_id,
        trading.batch.provenance.dataset_id
    );
}
#[test]
fn finite_budgets_identity_and_availability_fail_closed() {
    let (mut m, mut c) = fixture();
    c.max_working_bytes = 1;
    assert!(generate(&m, &c, range(0, 2)).is_err());
    c.max_working_bytes = 64 * 1024 * 1024;
    assert!(plan(&m, &c, &range(u64::MAX, 2)).is_err());
    m.provenance.synthetic = false;
    assert!(m.validate().is_err());
    m.provenance.known_at_ns = Some(99);
    m.provenance.decision_origin_ns = Some(100);
    assert!(m.validate().is_ok());
    m.provenance.known_at_ns = Some(101);
    assert!(m.validate().is_err());
    m.provenance.synthetic = true;
    m.provenance.known_at_ns = None;
    m.provenance.decision_origin_ns = None;
    m.asset_ids[1] = "A".into();
    assert!(m.validate().is_err());
}
#[test]
fn native_learner_executes_same_funded_cashflows() {
    let (m, c) = fixture();
    let x = generate(&m, &c, range(0, 2)).unwrap();
    let mut policy = HedgePolicyConfig::paper_heston(2);
    policy.time_steps = 3;
    policy.feature_count = 13;
    policy.hidden_width = 4;
    policy.batch_norm.enabled = false;
    let cfg = LearnerConfig {
        target: ExecutionTarget::Cpu,
        precision: Precision::F64,
        policy,
        objective: Objective::Entropic { risk_aversion: 0.1 },
        premium: x.economics.premium,
        trading_costs: x.economics.trading_costs.clone(),
        liquidation_costs: x.economics.liquidation_costs.clone(),
        cash_growth: x.economics.cash_growth.clone(),
        income_growth: x.economics.income_growth.clone(),
        ..Default::default()
    };
    let mut learner = NativeLearner::new(cfg).unwrap();
    learner
        .handle(Command::RegisterDataset {
            dataset_id: x.batch.provenance.dataset_id.clone(),
            partitions: partitions(&[range(0, 2)]).unwrap(),
        })
        .unwrap();
    learner
        .handle(Command::Upload {
            batch_id: "factor".into(),
            split: Split::Train,
            batch: x.batch.clone(),
        })
        .unwrap();
    let result = learner
        .handle(Command::Evaluate {
            batch_id: "factor".into(),
            retain_holdings: true,
            sharpen_gate: None,
        })
        .unwrap();
    for p in 0..2 {
        let mut value = x.economics.premium;
        let mut previous = [0_f64; 2];
        let mut costs = 0.;
        for t in 0..3 {
            let target = [
                result["holdings"][t][p * 2].as_f64().unwrap(),
                result["holdings"][t][p * 2 + 1].as_f64().unwrap(),
            ];
            let at = (p * 4 + t) * 2;
            let next = at + 2;
            let fee = (0..2)
                .map(|i| {
                    (target[i] - previous[i]).abs()
                        * x.batch.marks[at + i]
                        * x.economics.trading_costs[i]
                })
                .sum::<f64>();
            value = (value
                - (0..2)
                    .map(|i| target[i] * x.batch.marks[at + i])
                    .sum::<f64>()
                - fee)
                * x.economics.cash_growth[t]
                + (0..2)
                    .map(|i| {
                        target[i] * x.batch.marks[next + i] * x.economics.income_growth[t * 2 + i]
                    })
                    .sum::<f64>();
            costs += fee;
            previous = target;
        }
        near(
            value - x.batch.payoffs[p],
            result["pnl"][p].as_f64().unwrap(),
            1e-10,
        );
        near(costs, result["costs"][p].as_f64().unwrap(), 1e-12);
    }
    assert_eq!(
        learner
            .handle(Command::Update {
                batch_id: "factor".into(),
                expected_version: 0
            })
            .unwrap()["version"],
        1
    );
}
#[test]
fn simulated_return_covariance_recovers_supplied_factor_risk() {
    let (mut m, mut c) = fixture();
    m.annual_signal_coefficients = vec![0.; 2];
    m.coefficient_uncertainty_std = vec![0.; 2];
    m.annual_base_excess_return = vec![0.; 2];
    m.signal_innovation_std = 0.;
    m.dividend_yields = vec![0.; 2];
    c.rate = 0.;
    c.steps = 1;
    c.liabilities.clear();
    let x = generate(&m, &c, range(100, 4096)).unwrap();
    let mut first = [0.; 2];
    let mut second = [[0.; 2]; 2];
    for p in 0..4096 {
        let row = [
            (x.batch.marks[p * 4 + 2] / 100.).ln() + 0.5 * 0.04 * c.dt,
            (x.batch.marks[p * 4 + 3] / 80.).ln() + 0.5 * 0.026384 * c.dt,
        ];
        for i in 0..2 {
            first[i] += row[i];
            for j in 0..2 {
                second[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let cov = (second[i][j] / 4096. - first[i] * first[j] / 4096_f64.powi(2)) / c.dt;
            near(cov, x.risk.annual_stock_covariance[i][j], 0.0035);
        }
    }
}
