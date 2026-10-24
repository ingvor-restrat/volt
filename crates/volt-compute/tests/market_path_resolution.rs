use volt_compute::hedge_application::HedgeLiability;
use volt_compute::market_path_resolution::into_hedge_batch;
use volt_compute::market_path_resolution::{MarketPathResolutionConfig, generate_resolved_paths};
use volt_compute::market_paths::{
    FUNDED_VARIANCE_CLAIM, INTEGRATED_VARIANCE, MarketModel, MarketPathConfig, STOCK, VARIANCE,
    generate_paths,
};

fn config(substeps: usize) -> MarketPathResolutionConfig {
    MarketPathResolutionConfig {
        market: {
            let mut market = MarketPathConfig::paper_heston(7, 730019);
            market.integrated_variance_rule =
                volt_compute::market_paths::IntegratedVarianceRule::Trapezoid;
            market
        },
        substeps,
    }
}

#[test]
fn one_substep_is_bit_identical_including_base_provenance() {
    let c = config(1);
    let base = generate_paths(&c.market).unwrap();
    let refined = generate_resolved_paths(&c).unwrap();
    assert_eq!(base.values, refined.market.values);
    assert_eq!(base.content_hash, refined.market.content_hash);
    assert_eq!(base.inputs_hash, refined.market.inputs_hash);
    refined.validate().unwrap();
}

#[test]
fn fine_states_keep_horizon_and_all_four_current_fields_on_coarse_dates() {
    for n in [2, 4] {
        let c = config(n);
        let result = generate_resolved_paths(&c).unwrap();
        let fine = generate_paths(&result.fine_config).unwrap();
        assert_eq!(result.market.shape(), [7, 31, 4]);
        assert_eq!(result.fine_config.horizon(), c.market.horizon());
        for p in 0..7 {
            for t in 0..31 {
                for f in 0..4 {
                    assert_eq!(result.market.value(p, t, f), fine.value(p, t * n, f));
                }
                let v = result.market.value(p, t, VARIANCE).unwrap();
                let a = result.market.value(p, t, INTEGRATED_VARIANCE).unwrap();
                let tau = (30 - t) as f64 * c.market.dt;
                let expected = a + (v - 0.04) * (-(-tau).exp_m1()) + 0.04 * tau;
                assert!(
                    (expected - result.market.value(p, t, FUNDED_VARIANCE_CLAIM).unwrap()).abs()
                        < 1e-14
                );
            }
        }
        assert_eq!(
            result.market.times,
            (0..31).map(|i| i as f64 * c.market.dt).collect::<Vec<_>>()
        );
        // Terminal claim contains the fine accumulated integral, not a new
        // integral estimated from only the thirty retained endpoints.
        assert_eq!(
            result.market.value(0, 30, FUNDED_VARIANCE_CLAIM),
            result.market.value(0, 30, INTEGRATED_VARIANCE)
        );
    }
}

#[test]
fn path_identity_is_independent_of_chunking() {
    for n in [1, 2, 4] {
        let c = config(n);
        let all = generate_resolved_paths(&c).unwrap();
        let mut values = Vec::new();
        for (first, paths) in [(0, 3), (3, 3), (6, 1)] {
            let mut chunk = c.clone();
            chunk.market.first_path_id = first;
            chunk.market.paths = paths;
            let r = generate_resolved_paths(&chunk).unwrap();
            r.validate().unwrap();
            values.extend(r.market.values);
        }
        assert_eq!(all.market.values, values);
    }
}

#[test]
fn current_observations_do_not_use_later_random_draws() {
    // BS stock/current A at time t depend on the prefix, regardless of the
    // later maturity. H for BS intentionally quotes total deterministic A_T.
    let mut c = config(4);
    c.market.model = MarketModel::BlackScholes { volatility: 0.2 };
    let full = generate_resolved_paths(&c).unwrap();
    c.market.steps = 15;
    let short = generate_resolved_paths(&c).unwrap();
    for p in 0..7 {
        for t in 0..16 {
            for f in [STOCK, VARIANCE, INTEGRATED_VARIANCE] {
                assert_eq!(full.market.value(p, t, f), short.market.value(p, t, f));
            }
        }
    }
}

#[test]
fn invalid_shapes_and_mutated_provenance_are_rejected() {
    for n in [0, usize::MAX, 4097] {
        assert!(config(n).validate().is_err());
    }
    let mut c = config(4);
    c.market.first_path_id = u64::MAX;
    assert!(c.validate().is_err());
    c = config(4);
    c.market.paths = usize::MAX;
    assert!(c.validate().is_err());
    let r = generate_resolved_paths(&config(2)).unwrap();
    let mut x = r.clone();
    x.config.substeps = 4;
    assert!(x.validate().is_err());
    let mut x = r.clone();
    x.market.values[4] += 1.;
    assert!(x.validate().is_err());
    let mut x = r;
    x.fine_content_hash = "0".repeat(64);
    assert!(x.validate().is_err());
}

#[test]
fn existing_policy_adapter_keeps_thirty_decisions_and_current_features() {
    let r = generate_resolved_paths(&config(4)).unwrap();
    let market = r.market.clone();
    let batch = into_hedge_batch(vec![r], HedgeLiability::Call { strike: 100.0 }).unwrap();
    assert_eq!(
        (
            batch.paths,
            batch.steps,
            batch.hedge_count,
            batch.feature_count
        ),
        (7, 30, 2, 2)
    );
    for p in 0..7 {
        for t in 0..30 {
            let i = (p * 30 + t) * 2;
            assert_eq!(batch.features[i], market.value(p, t, STOCK).unwrap().ln());
            assert_eq!(batch.features[i + 1], market.value(p, t, VARIANCE).unwrap());
            assert_eq!(
                batch.marks[(p * 31 + t) * 2 + 1],
                market.value(p, t, FUNDED_VARIANCE_CLAIM).unwrap()
            );
        }
    }
}

#[test]
fn later_random_inputs_cannot_change_earlier_heston_marks_or_features() {
    use volt_compute::market_paths::{MarketPathInputs, evolve_host, generate_inputs};
    let c = config(4);
    let fine = c.validate().unwrap().fine_config;
    let inputs = generate_inputs(&fine).unwrap();
    let base = evolve_host(&inputs).unwrap();
    let mut endpoints = inputs.variance_endpoints.clone();
    let mut normals = inputs.independent_normals.clone();
    for p in 0..fine.paths {
        for t in 41..=fine.steps {
            endpoints[p * (fine.steps + 1) + t] += 0.1;
        }
        for t in 40..fine.steps {
            normals[p * fine.steps + t] += 0.3;
        }
    }
    let replay = MarketPathInputs::from_explicit(fine, endpoints, normals).unwrap();
    let changed = evolve_host(&replay).unwrap();
    for p in 0..7 {
        for coarse in 0..=10 {
            for field in 0..4 {
                assert_eq!(
                    base.value(p, coarse * 4, field),
                    changed.value(p, coarse * 4, field)
                );
            }
        }
    }
    assert_ne!(base.value(0, 120, STOCK), changed.value(0, 120, STOCK));
}
