use burn::tensor::{DType, Tensor, TensorCreationOptions, TensorData};
use volt_compute::hedge_application::{
    HedgeCostConfig, HedgeLiability, HedgePathBatch, PreparedHedgePaths, plan_hedging,
    policy_rollout, tensor_risk,
};
use volt_compute::hedge_policy::{HedgePolicy, HedgePolicyConfig};
use volt_compute::risk_objective::{RiskObjective, weighted_risk};
use volt_compute::{ExecutionTarget, Precision, Session, budget::MemoryBudget};
fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}
fn precisions() -> Vec<Precision> {
    if target() == ExecutionTarget::Metal {
        vec![Precision::F32]
    } else {
        vec![Precision::F32, Precision::F64]
    }
}
fn vector(s: &Session, x: Vec<f64>) -> Tensor<1> {
    let len = x.len();
    Tensor::from_data(
        TensorData::new(x, [len]),
        TensorCreationOptions::new(s.device().clone()).with_dtype(
            if s.precision() == Precision::F64 {
                DType::F64
            } else {
                DType::F32
            },
        ),
    )
}
fn read(t: Tensor<1>) -> Vec<f64> {
    t.without_autodiff().try_to_vec_as::<f64>().unwrap()
}
fn close(a: f64, b: f64, abs: f64, rel: f64) {
    assert!(
        (a - b).abs() <= abs + rel * b.abs(),
        "{a:.16e} vs {b:.16e}, error{}",
        (a - b).abs()
    );
}
fn paths() -> HedgePathBatch {
    HedgePathBatch {
        paths: 2,
        steps: 2,
        hedge_count: 2,
        feature_count: 2,
        marks: vec![
            100.0, 2.0, 105.0, 3.0, 102.0, 4.0, 100.0, 2.0, 98.0, 1.5, 101.0, 1.0,
        ],
        features: vec![
            100f64.ln(),
            0.04,
            105f64.ln(),
            0.05,
            100f64.ln(),
            0.04,
            98f64.ln(),
            0.03,
        ],
        payoffs: vec![3.0, 4.0],
        source_hash: "hand-calculated-two-path-control".into(),
    }
}
fn config() -> HedgePolicyConfig {
    let mut c = HedgePolicyConfig::paper_heston(8);
    c.time_steps = 2;
    c.hidden_width = 3;
    c.batch_norm.enabled = false;
    c
}

#[test]
fn hand_calculated_self_financing_gains_costs_and_terminal_liquidation() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let prepared = PreparedHedgePaths::upload(&s, &paths()).unwrap();
        let mut snap = HedgePolicy::new(&s, config(), 9)
            .unwrap()
            .snapshot()
            .unwrap();
        for (step, holdings) in [[1.0, -2.0], [0.5, 0.25]].into_iter().enumerate() {
            snap.parameters[step * 10 + 8].values.fill(0.0);
            snap.parameters[step * 10 + 9].values = holdings.to_vec();
        }
        let mut policy = HedgePolicy::from_snapshot(&s, snap).unwrap();
        let result = policy_rollout(
            &s,
            &prepared,
            &mut policy,
            HedgeCostConfig {
                proportional: 0.01,
                terminal_proportional: 0.02,
            },
            5.0,
            false,
            true,
        )
        .unwrap();
        assert_eq!(result.holdings.len(), 2);
        assert!(result.holdings.iter().all(|x| !x.is_autodiff()));
        let tolerance = if precision == Precision::F64 {
            1e-12
        } else {
            3e-6
        };
        for (a, b) in read(result.pnl).into_iter().zip([1.0775, -1.20375]) {
            close(a, b, tolerance, 0.0);
        }
        for (a, b) in read(result.costs).into_iter().zip([2.6725, 2.57875]) {
            close(a, b, tolerance, 0.0);
        }
        for (a, b) in read(result.turnover).into_iter().zip([215.25, 207.125]) {
            close(a, b, tolerance, 0.0);
        }
        let result = policy_rollout(
            &s,
            &prepared,
            &mut policy,
            HedgeCostConfig::default(),
            5.0,
            false,
            false,
        )
        .unwrap();
        assert!(result.holdings.is_empty());
        assert_eq!(read(result.costs), [0.0, 0.0]);
        assert_eq!(read(result.turnover), [215.25, 207.125]);
        for (a, b) in read(result.pnl).into_iter().zip([3.75, 1.375]) {
            close(a, b, tolerance, 0.0);
        }
    }
}

#[test]
fn application_pnl_gradient_includes_recurrent_holdings_and_trading_costs() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let prepared = PreparedHedgePaths::upload(&s, &paths()).unwrap();
        let mut snap = HedgePolicy::new(&s, config(), 9)
            .unwrap()
            .snapshot()
            .unwrap();
        for (i, p) in snap.parameters.iter_mut().enumerate() {
            p.values.fill(match i % 10 {
                0 | 4 | 8 => 0.2,
                2 | 6 => 1.0,
                _ => 0.3,
            });
        }
        let cost = HedgeCostConfig {
            proportional: 0.001,
            terminal_proportional: 0.002,
        };
        let mut policy = HedgePolicy::from_snapshot(&s, snap.clone()).unwrap();
        let leaf = policy.parameters()[0].clone();
        let result = policy_rollout(&s, &prepared, &mut policy, cost, 5.0, true, false).unwrap();
        let gradients = tensor_risk(result.pnl, RiskObjective::SquaredLoss, None)
            .unwrap()
            .backward();
        let actual = leaf
            .grad(&gradients)
            .unwrap()
            .try_to_vec_as::<f64>()
            .unwrap()[0];
        assert!(actual.abs() > 1e-3);
        let at = |dx: f64| {
            let mut snap = snap.clone();
            snap.parameters[0].values[0] += dx;
            let mut policy = HedgePolicy::from_snapshot(&s, snap).unwrap();
            let result =
                policy_rollout(&s, &prepared, &mut policy, cost, 5.0, false, false).unwrap();
            weighted_risk(&read(result.pnl), None, RiskObjective::SquaredLoss)
                .unwrap()
                .value
        };
        let h = if precision == Precision::F64 {
            1e-5
        } else {
            1e-3
        };
        close(
            actual,
            (at(h) - at(-h)) / (2.0 * h),
            if precision == Precision::F64 {
                3e-8
            } else {
                2e-3
            },
            if precision == Precision::F64 {
                2e-8
            } else {
                2e-3
            },
        );
    }
}

#[test]
fn tensor_objectives_match_independent_scalar_values_and_derivatives() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let values = vec![-4.0, -1.0, 0.0, 2.0, 3.0];
        let abs = if precision == Precision::F64 {
            2e-12
        } else {
            2e-6
        };
        for objective in [
            RiskObjective::SquaredLoss,
            RiskObjective::Cvar { alpha: 0.7 },
            RiskObjective::Entropic { risk_aversion: 1.7 },
        ] {
            let pnl = vector(&s, values.clone()).autodiff().require_grad();
            let eta = if matches!(objective, RiskObjective::Cvar { .. }) {
                Some(vector(&s, vec![1.0]).autodiff().require_grad())
            } else {
                None
            };
            let risk = tensor_risk(pnl.clone(), objective, eta).unwrap();
            close(
                read(risk.clone())[0],
                weighted_risk(&values, None, objective).unwrap().value,
                abs,
                if precision == Precision::F64 {
                    1e-12
                } else {
                    1e-6
                },
            );
            let grads = risk.backward();
            let gradient = read(pnl.grad(&grads).unwrap());
            match objective {
                RiskObjective::SquaredLoss => {
                    for (a, p) in gradient.iter().zip(&values) {
                        close(*a, 2.0 * p / 5.0, abs, 0.0)
                    }
                }
                RiskObjective::Entropic {
                    risk_aversion: lambda,
                } => {
                    let denominator: f64 = values.iter().map(|p| (-lambda * p).exp()).sum();
                    for (a, p) in gradient.iter().zip(&values) {
                        close(*a, -(-lambda * p).exp() / denominator, abs, 0.0);
                    }
                }
                RiskObjective::Cvar { .. } => {
                    close(gradient[0], -2.0 / 3.0, abs, 0.0);
                    for g in &gradient[2..] {
                        close(*g, 0.0, abs, 0.0);
                    }
                }
            }
        }
        // CVaR eta away from the kink has a well-defined derivative of -1/3.
        let pnl = vector(&s, values);
        let eta = vector(&s, vec![0.5]).autodiff().require_grad();
        let risk = tensor_risk(pnl, RiskObjective::Cvar { alpha: 0.7 }, Some(eta.clone())).unwrap();
        let grads = risk.backward();
        close(read(eta.grad(&grads).unwrap())[0], -1.0 / 3.0, abs, 0.0);
        // Shift-stabilized entropy must survive naive exp(1000*1000) overflow.
        let values = vec![-1000.0, -999.0, 1000.0];
        let objective = RiskObjective::Entropic {
            risk_aversion: 1000.0,
        };
        let actual = tensor_risk(vector(&s, values.clone()), objective, None).unwrap();
        close(
            read(actual)[0],
            weighted_risk(&values, None, objective).unwrap().value,
            if precision == Precision::F64 {
                1e-10
            } else {
                1e-4
            },
            0.0,
        );
    }
}

#[test]
fn hostile_shapes_precision_and_admission_are_rejected_before_upload() {
    let s = Session::new(target(), Precision::F32, 4096).unwrap();
    let good = paths();
    assert!(good.validate().is_ok());
    let mut bad = good.clone();
    bad.steps = usize::MAX;
    assert!(bad.validate().is_err());
    let mut bad = good.clone();
    bad.features.pop();
    assert!(PreparedHedgePaths::upload(&s, &bad).is_err());
    for field in 0..3 {
        let mut bad = good.clone();
        match field {
            0 => bad.marks[0] = f64::MAX,
            1 => bad.features[0] = f64::MAX,
            _ => bad.payoffs[0] = f64::MAX,
        };
        assert!(PreparedHedgePaths::upload(&s, &bad).is_err());
    }
    let c = config();
    let budget = MemoryBudget {
        max_batch_rows: 1,
        ..MemoryBudget::default()
    };
    assert!(plan_hedging(&c, 2, Precision::F32, target(), &budget, true).is_err());
    let budget = MemoryBudget {
        compute_bytes: 1,
        host_working_bytes: 1,
        compute_reserve_bytes: 0,
        ..MemoryBudget::default()
    };
    assert!(plan_hedging(&c, 2, Precision::F32, target(), &budget, true).is_err());
    assert!(
        tensor_risk(
            vector(&s, vec![1.0]),
            RiskObjective::Cvar { alpha: 0.9 },
            Some(vector(&s, vec![0.0, 0.0]))
        )
        .is_err()
    );
    if target() == ExecutionTarget::Cpu {
        let f64session = Session::new(target(), Precision::F64, 4096).unwrap();
        assert!(
            tensor_risk(
                vector(&s, vec![1.0]),
                RiskObjective::Cvar { alpha: 0.9 },
                Some(vector(&f64session, vec![0.0]))
            )
            .is_err()
        );
    }
}

#[test]
fn normalized_spread_remains_bounded_for_huge_spots() {
    let spread = HedgeLiability::NormalizedCallSpread {
        lower: 100.0,
        upper: 101.0,
    };
    assert_eq!(spread.payoff(1e100).unwrap(), 1.0);
    assert_eq!(spread.payoff(100.5).unwrap(), 0.5);
    assert_eq!(spread.payoff(99.0).unwrap(), 0.0);
    assert!(spread.payoff(f64::INFINITY).is_err());
}

#[test]
fn small_entropic_scale_keeps_mean_limit_and_nonzero_gradients() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        for lambda in [1e-3, 1e-10, 1e-20] {
            let pnl = vector(&s, vec![-1.0, 2.0, 5.0]).autodiff().require_grad();
            let loss = tensor_risk(
                pnl.clone(),
                RiskObjective::Entropic {
                    risk_aversion: lambda,
                },
                None,
            )
            .unwrap();
            let actual = loss
                .clone()
                .without_autodiff()
                .try_to_vec_as::<f64>()
                .unwrap()[0];
            let expected = volt_compute::risk_objective::weighted_risk(
                &[-1.0, 2.0, 5.0],
                None,
                RiskObjective::Entropic {
                    risk_aversion: lambda,
                },
            )
            .unwrap()
            .value;
            let tol = if precision == Precision::F64 {
                1e-11
            } else {
                2e-6
            };
            assert!(
                (actual - expected).abs() < tol,
                "{precision:?} lambda{lambda}: {actual} vs {expected}"
            );
            let gradients = loss.backward();
            let observed = pnl
                .grad(&gradients)
                .unwrap()
                .try_to_vec_as::<f64>()
                .unwrap();
            let exps: Vec<_> = [-1.0, 2.0, 5.0]
                .iter()
                .map(|p: &f64| (-lambda * (*p + 1.0)).exp())
                .collect();
            let total: f64 = exps.iter().sum();
            for (a, b) in observed.iter().zip(exps) {
                assert!((a + b / total).abs() < tol);
            }
        }
    }
}
