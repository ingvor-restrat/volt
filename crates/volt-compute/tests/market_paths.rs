use volt_compute::{
    ExecutionTarget, Precision, Session,
    market_paths::{
        FUNDED_VARIANCE_CLAIM, INTEGRATED_VARIANCE, IntegratedVarianceRule, MARKET_FIELDS,
        MarketModel, MarketPathConfig, MarketPathInputs, STOCK, VARIANCE, evolve_host,
        evolve_on_device, generate_inputs, generate_paths,
    },
    monte_carlo::OnlineMoments,
};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}
fn small() -> MarketPathConfig {
    let mut c = MarketPathConfig::paper_heston(17, 9182);
    c.steps = 7;
    c
}
fn moments(values: impl Iterator<Item = f64>) -> OnlineMoments {
    let mut m = OnlineMoments::default();
    for v in values {
        m.add(v);
    }
    m
}

#[test]
fn original_parameters_shape_hashes_and_chunk_identity_are_explicit() {
    let config = small();
    let all = generate_paths(&config).unwrap();
    assert_eq!(all.shape(), [17, 8, 4]);
    assert_eq!(all.config.dt, 1.0 / 365.0);
    assert_eq!(all.config.initial_spot, 100.0);
    assert!(all.rng_contract.contains("philox4x32-10"));
    assert_eq!(all.values.len(), config.validate().unwrap().output_values);
    assert_eq!(all.path_id(16), Some(16));
    assert_eq!(all.path_id(17), None);
    assert_eq!(
        all.content_hash,
        generate_paths(&config).unwrap().content_hash
    );
    let mut joined = Vec::new();
    for (start, count) in [(0, 5), (5, 7), (12, 5)] {
        let mut chunk = config.clone();
        chunk.first_path_id = start;
        chunk.paths = count;
        joined.extend(generate_paths(&chunk).unwrap().values);
    }
    assert_eq!(all.values, joined);
    assert!(all.value(0, 8, STOCK).is_none());
    assert!(all.value(0, 0, 4).is_none());
    for path in 0..config.paths {
        assert_eq!(all.value(path, 0, STOCK), Some(100.0));
        assert_eq!(all.value(path, 0, VARIANCE), Some(0.04));
        assert_eq!(all.value(path, 0, INTEGRATED_VARIANCE), Some(0.0));
        assert_eq!(
            all.value(path, config.steps, FUNDED_VARIANCE_CLAIM),
            all.value(path, config.steps, INTEGRATED_VARIANCE)
        );
    }
    let mut changed = config.clone();
    changed.seed += 1;
    assert_ne!(
        generate_paths(&changed).unwrap().content_hash,
        all.content_hash
    );
    let serialized = serde_json::to_value(&config).unwrap();
    let mut unknown = serialized;
    unknown["trend"] = serde_json::json!(1.0);
    assert!(serde_json::from_value::<MarketPathConfig>(unknown).is_err());
    // Cross the automatic parallelization threshold; a tiny independently
    // generated tail must reproduce the parallel worker's absolute path IDs.
    let mut parallel = config.clone();
    parallel.paths = 4097;
    parallel.steps = 2;
    let all_inputs = generate_inputs(&parallel).unwrap();
    parallel.first_path_id = 4090;
    parallel.paths = 7;
    let tail = generate_inputs(&parallel).unwrap();
    assert_eq!(
        &all_inputs.variance_endpoints[4090 * 3..],
        tail.variance_endpoints
    );
    assert_eq!(
        &all_inputs.independent_normals[4090 * 2..],
        tail.independent_normals
    );
}

#[test]
fn cir_conditional_first_and_second_moments_include_small_shape_and_large_poisson() {
    for (initial, xi, theta, count) in [
        (0.04, 2.0, 0.04, 30_000),
        (0.0, 2.0, 0.04, 30_000),
        (1000.0, 0.2, 0.04, 8000),
    ] {
        for seed in [17, 201] {
            let mut config = MarketPathConfig::paper_heston(count, seed);
            config.steps = 1;
            if let MarketModel::Heston { ref mut params } = config.model {
                params.initial_variance = initial;
                params.vol_of_variance = xi;
                params.long_run_variance = theta;
            }
            let inputs = generate_inputs(&config).unwrap();
            let final_v = inputs
                .variance_endpoints
                .as_chunks::<2>()
                .0
                .iter()
                .map(|row| row[1])
                .collect::<Vec<_>>();
            let e = (-config.dt).exp();
            let expected_mean = theta + (initial - theta) * e;
            let expected_variance =
                initial * xi * xi * e * (1.0 - e) + theta * xi * xi * 0.5 * (1.0 - e).powi(2);
            for (sample, expected) in [
                (moments(final_v.iter().copied()), expected_mean),
                (
                    moments(final_v.iter().map(|v| v * v)),
                    expected_variance + expected_mean * expected_mean,
                ),
            ] {
                let se = sample.estimate().standard_error.unwrap();
                assert!(
                    (sample.mean - expected).abs() < 6.0 * se + 1e-8 * expected.abs() + 1e-12,
                    "seed{seed} V0{initial} xi{xi}: {} vs {expected}, SE{se}",
                    sample.mean
                );
            }
            assert!(final_v.iter().all(|v| *v >= 0.0 && v.is_finite()));
        }
    }
}

#[test]
fn zero_boundary_small_gamma_underflow_and_deterministic_xi_are_distinct() {
    let mut absorbing = MarketPathConfig::paper_heston(512, 91);
    absorbing.steps = 3;
    if let MarketModel::Heston { ref mut params } = absorbing.model {
        params.initial_variance = 0.0;
        params.long_run_variance = 0.0;
    }
    let inputs = generate_inputs(&absorbing).unwrap();
    assert!(inputs.variance_endpoints.iter().all(|v| *v == 0.0));
    assert_eq!(inputs.diagnostics.gamma_underflow_endpoints, 0);
    assert_eq!(inputs.diagnostics.zero_variance_endpoints, 1536);
    let paths = evolve_host(&inputs).unwrap();
    assert!(
        paths
            .values
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| (p[0] - 100.0).abs() < 1e-12 && p[1..] == [0.0, 0.0, 0.0])
    );
    let mut tiny = absorbing.clone();
    if let MarketModel::Heston { ref mut params } = tiny.model {
        params.long_run_variance = 1e-8;
    }
    let tiny = generate_inputs(&tiny).unwrap();
    assert!(tiny.diagnostics.gamma_underflow_endpoints > 0);
    assert!(tiny.diagnostics.zero_variance_endpoints >= tiny.diagnostics.gamma_underflow_endpoints);
    let mut deterministic = small();
    if let MarketModel::Heston { ref mut params } = deterministic.model {
        params.initial_variance = 0.0625;
        params.long_run_variance = 0.0625;
        params.vol_of_variance = 0.0;
    }
    let heston = generate_paths(&deterministic).unwrap();
    deterministic.model = MarketModel::BlackScholes { volatility: 0.25 };
    let bs = generate_paths(&deterministic).unwrap();
    for (a, b) in heston.values.iter().zip(bs.values) {
        assert!((a - b).abs() < 1e-11);
    }
}

#[test]
fn explicit_replay_checks_bk_coupling_quadrature_and_funded_claim_timing() {
    let mut config = MarketPathConfig::paper_heston(1, 7);
    config.steps = 2;
    config.dt = 0.5;
    let endpoints = vec![0.04, 0.09, 0.01];
    let normals = vec![0.2, -1.0];
    let left = evolve_host(
        &MarketPathInputs::from_explicit(config.clone(), endpoints.clone(), normals.clone())
            .unwrap(),
    )
    .unwrap();
    assert!((left.value(0, 1, INTEGRATED_VARIANCE).unwrap() - 0.02).abs() < 1e-15);
    assert!((left.value(0, 2, INTEGRATED_VARIANCE).unwrap() - 0.065).abs() < 1e-15);
    let claim = 0.02 + (0.09 - 0.04) * (1.0 - (-0.5_f64).exp()) + 0.04 * 0.5;
    assert!((left.value(0, 1, FUNDED_VARIANCE_CLAIM).unwrap() - claim).abs() < 1e-15);
    let rho = -0.7_f64;
    let integral = 0.02;
    let log_increment = rho / 2.0 * (0.09 - 0.04 - 0.04 * 0.5)
        + (rho / 2.0 - 0.5) * integral
        + ((1.0 - rho * rho) * integral).sqrt() * 0.2;
    assert!((left.value(0, 1, STOCK).unwrap() - 100.0 * log_increment.exp()).abs() < 1e-12);
    config.integrated_variance_rule = IntegratedVarianceRule::Trapezoid;
    let trap = evolve_host(
        &MarketPathInputs::from_explicit(config.clone(), endpoints.clone(), normals.clone())
            .unwrap(),
    )
    .unwrap();
    assert!((trap.value(0, 2, INTEGRATED_VARIANCE).unwrap() - 0.0575).abs() < 1e-15);
    assert_ne!(left.content_hash, trap.content_hash);
    config.rate = 0.03;
    let funded =
        evolve_host(&MarketPathInputs::from_explicit(config, endpoints, normals).unwrap()).unwrap();
    assert!(
        (funded.value(0, 0, FUNDED_VARIANCE_CLAIM).unwrap() - 0.04 * (-0.03_f64).exp()).abs()
            < 1e-15
    );
}

#[test]
fn bs_stock_martingale_and_funded_variance_expectations_have_sampling_controls() {
    for seed in [101, 509] {
        let mut config = MarketPathConfig::paper_heston(20_000, seed);
        config.model = MarketModel::BlackScholes { volatility: 0.2 };
        config.rate = 0.03;
        config.dividend_yield = 0.01;
        let paths = generate_paths(&config).unwrap();
        let stock =
            moments((0..config.paths).map(|p| paths.value(p, config.steps, STOCK).unwrap()));
        let expected =
            config.initial_spot * ((config.rate - config.dividend_yield) * config.horizon()).exp();
        assert!((stock.mean - expected).abs() < 6.0 * stock.estimate().standard_error.unwrap());
        let mut heston = MarketPathConfig::paper_heston(20_000, seed);
        heston.integrated_variance_rule = IntegratedVarianceRule::Trapezoid;
        let heston_paths = generate_paths(&heston).unwrap();
        let expected = 0.04 * heston.horizon();
        for time in [10, 20, 30] {
            let claim = moments(
                (0..heston.paths)
                    .map(|p| heston_paths.value(p, time, FUNDED_VARIANCE_CLAIM).unwrap()),
            );
            assert!(
                (claim.mean - expected).abs()
                    < 6.0 * claim.estimate().standard_error.unwrap() + 1e-10
            );
        }
        // The simplified Heston stock is NOT asserted to be an exact martingale.
        let stock = moments((0..heston.paths).map(|p| heston_paths.value(p, 30, STOCK).unwrap()));
        eprintln!(
            "simplifiedBK seed{seed} terminal spot mean{}, SE{}",
            stock.mean,
            stock.estimate().standard_error.unwrap()
        );
        assert!(stock.mean.is_finite() && stock.estimate().standard_error.unwrap() > 0.0);
    }
}

#[test]
fn explicit_shared_inputs_match_host_and_selected_device_arithmetic() {
    let precisions = if target() == ExecutionTarget::Metal {
        eprintln!("SKIP market replay f64: Metal lacks native f64");
        vec![Precision::F32]
    } else {
        vec![Precision::F32, Precision::F64]
    };
    for model in [small().model, MarketModel::BlackScholes { volatility: 0.2 }] {
        for rule in [
            IntegratedVarianceRule::Left,
            IntegratedVarianceRule::Trapezoid,
        ] {
            let mut config = small();
            config.model = model;
            config.integrated_variance_rule = rule;
            let inputs = generate_inputs(&config).unwrap();
            let host = evolve_host(&inputs).unwrap();
            for precision in &precisions {
                let session = Session::new(target(), *precision, host.values.len()).unwrap();
                let device = evolve_on_device(&session, &inputs).unwrap();
                assert_eq!(device.shape(), host.shape());
                assert_eq!(device.inputs_hash, host.inputs_hash);
                for (i, (actual, expected)) in device.values.iter().zip(&host.values).enumerate() {
                    let tolerance = if *precision == Precision::F32 {
                        3e-5 + 2e-5 * expected.abs()
                    } else {
                        1e-11 + 1e-11 * expected.abs()
                    };
                    assert!(
                        (actual - expected).abs() < tolerance,
                        "field{} {precision:?}: {actual} vs {expected}",
                        i % MARKET_FIELDS
                    );
                }
            }
        }
    }
}

#[test]
fn shape_limits_replay_tampering_and_device_underflows_are_checked() {
    let mut config = small();
    config.paths = usize::MAX;
    assert!(config.validate().is_err());
    config = small();
    config.first_path_id = u64::MAX;
    assert!(config.validate().is_err());
    config = small();
    config.paths = 1_000_000;
    config.steps = 30;
    assert_eq!(config.validate().unwrap().output_values, 124_000_000);
    config.paths = 2_000_000;
    assert!(config.validate().is_err());
    config = small();
    config.dt = f64::NAN;
    assert!(generate_inputs(&config).is_err());
    config = small();
    if let MarketModel::Heston { ref mut params } = config.model {
        params.initial_variance = -1.0;
    }
    assert!(generate_inputs(&config).is_err());
    let mut inputs = generate_inputs(&small()).unwrap();
    inputs.independent_normals[0] += 0.1;
    assert!(evolve_host(&inputs).is_err());
    let original = generate_paths(&small()).unwrap();
    original.validate().unwrap();
    for mutation in 0..5 {
        let mut altered = original.clone();
        match mutation {
            0 => altered.values[4] += 1e-7,
            1 => altered.times[1] += 1e-7,
            2 => altered.config.rate = 0.1,
            3 => altered.content_hash = "0".repeat(64),
            _ => altered.diagnostics.gamma_underflow_endpoints += 1,
        }
        assert!(altered.validate().is_err());
    }
    let mut config = small();
    config.paths = 1;
    config.steps = 2;
    let inputs =
        MarketPathInputs::from_explicit(config, vec![0.04, 1e-100, 0.09], vec![0.0, 0.0]).unwrap();
    let session = Session::new(target(), Precision::F32, 12).unwrap();
    let output = evolve_on_device(&session, &inputs).unwrap();
    assert_eq!(output.diagnostics.device_variance_underflow_endpoints, 1);
    assert_eq!(output.value(0, 1, VARIANCE), Some(0.0));
    let cap = Session::new(target(), Precision::F32, 11).unwrap();
    assert!(evolve_on_device(&cap, &inputs).is_err());
}
