use volt_compute::{
    ComputeError, ExecutionTarget, Precision,
    matrix::{MatrixBudget, MatrixWorkspace},
    matrix_workload::{self, MatrixWorkloadConfig},
};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}

fn small_budget(compute_bytes: u64) -> MatrixBudget {
    MatrixBudget {
        compute_bytes,
        host_working_bytes: 1_000_000,
        compute_reserve_bytes: 0,
        headroom_percent: 0,
        matmul_scratch_multiplier: 2,
    }
}

#[test]
fn output_and_scratch_admission_rolls_back_without_leaking_leases() {
    let workspace =
        MatrixWorkspace::new(ExecutionTarget::Cpu, Precision::F32, small_budget(180)).unwrap();
    let a = workspace.upload(2, 3, &[1., 2., 3., 4., 5., 6.]).unwrap();
    let b = workspace
        .upload(3, 2, &[7., 8., 9., 10., 11., 12.])
        .unwrap();
    assert_eq!(workspace.snapshot().live_tensor_bytes, 48);
    // 48 live operand bytes + 16 output + 128 modeled scratch cannot fit 180.
    assert!(matches!(
        workspace.matmul(&a, &b),
        Err(ComputeError::BudgetExceeded(_))
    ));
    let snapshot = workspace.snapshot();
    assert_eq!(snapshot.live_tensor_count, 2);
    assert_eq!(snapshot.live_tensor_bytes, 48);
    assert_eq!(snapshot.active_scratch_bytes, 0);
    assert_eq!(snapshot.live_host_working_bytes, 0);
    drop((a, b));
    assert_eq!(workspace.snapshot().live_tensor_count, 0);
    assert_eq!(workspace.snapshot().live_tensor_bytes, 0);
}

#[test]
fn download_and_user_host_storage_retain_leases_until_drop() {
    let workspace =
        MatrixWorkspace::new(ExecutionTarget::Cpu, Precision::F32, small_budget(1000)).unwrap();
    let host = workspace.reserve_host_bytes(123).unwrap();
    let a = workspace.upload(2, 2, &[1., 2., 3., 4.]).unwrap();
    let product = workspace.matmul(&a, &a).unwrap();
    let readback = workspace.download(&product).unwrap();
    assert_eq!(readback.values(), [7., 10., 15., 22.]);
    assert_eq!(readback.shape(), [2, 2]);
    assert_eq!(workspace.snapshot().live_host_working_bytes, 123 + 4 * 8);
    drop(readback);
    assert_eq!(workspace.snapshot().live_host_working_bytes, 123);
    drop(host);
    assert_eq!(workspace.snapshot().live_host_working_bytes, 0);
}

#[test]
fn independent_odd_rectangular_chain_matches_f64_reference() {
    let target = target();
    let precisions = if target == ExecutionTarget::Metal {
        vec![Precision::F32]
    } else {
        vec![Precision::F32, Precision::F64]
    };
    for precision in precisions {
        let workspace = MatrixWorkspace::new(target, precision, MatrixBudget::default()).unwrap();
        let a_values = values(17 * 23, 3);
        let b_values = values(23 * 11, 7);
        let c_values = values(11 * 5, 11);
        let a = workspace.upload(17, 23, &a_values).unwrap();
        let b = workspace.upload(23, 11, &b_values).unwrap();
        let c = workspace.upload(11, 5, &c_values).unwrap();
        let ab = workspace.matmul(&a, &b).unwrap();
        let abc = workspace.matmul(&ab, &c).unwrap();
        let actual = workspace.download(&abc).unwrap();
        let ab_expected = independent_matmul(&a_values, &b_values, 17, 23, 11);
        let expected = independent_matmul(&ab_expected, &c_values, 17, 11, 5);
        assert_eq!(actual.shape(), [17, 5]);
        for (actual, expected) in actual.values().iter().zip(expected) {
            let tolerance = if precision == Precision::F64 {
                1e-11
            } else {
                1e-5
            };
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} vs {expected}"
            );
        }
        let snapshot = workspace.snapshot();
        assert_eq!(snapshot.live_tensor_count, 5);
        assert!(snapshot.peak_modeled_compute_bytes > snapshot.live_tensor_bytes);
        assert!(snapshot.peak_modeled_compute_bytes <= snapshot.compute_admission_limit_bytes);
        drop((a, b, c, ab, abc, actual));
        assert_eq!(workspace.snapshot().live_tensor_count, 0);
    }
}

#[test]
fn rejects_unowned_matrices_invalid_values_and_arithmetic_overflow() {
    let a = MatrixWorkspace::new(
        ExecutionTarget::Cpu,
        Precision::F32,
        MatrixBudget::default(),
    )
    .unwrap();
    let b = MatrixWorkspace::new(
        ExecutionTarget::Cpu,
        Precision::F64,
        MatrixBudget::default(),
    )
    .unwrap();
    let owned_by_a = a.upload(1, 2, &[1., 2.]).unwrap();
    let owned_by_b = b.upload(2, 1, &[3., 4.]).unwrap();
    assert!(a.matmul(&owned_by_a, &owned_by_b).is_err());
    assert!(a.download(&owned_by_b).is_err());
    assert!(a.matmul(&owned_by_a, &owned_by_a).is_err());
    assert!(a.upload(usize::MAX, 2, &[]).is_err());
    assert!(a.upload(1, 1, &[f64::INFINITY]).is_err());
    assert!(a.upload(1, 1, &[f64::MAX]).is_err());
    assert!(a.reserve_host_bytes(u64::MAX).is_err());
    assert_eq!(a.snapshot().live_host_working_bytes, 0);
    assert!(
        MatrixWorkspace::new(
            ExecutionTarget::Metal,
            Precision::F64,
            MatrixBudget::default()
        )
        .is_err()
    );
    let invalid_budget = MatrixBudget {
        headroom_percent: 100,
        ..Default::default()
    };
    assert!(MatrixWorkspace::new(ExecutionTarget::Cpu, Precision::F32, invalid_budget).is_err());
}

#[test]
fn cloned_workspace_coordinates_concurrent_admission() {
    let workspace = MatrixWorkspace::new(
        ExecutionTarget::Cpu,
        Precision::F32,
        MatrixBudget {
            compute_bytes: 1000,
            host_working_bytes: 8192 + 128,
            compute_reserve_bytes: 0,
            headroom_percent: 0,
            matmul_scratch_multiplier: 2,
        },
    )
    .unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let workspace = workspace.clone();
        let barrier = barrier.clone();
        workers.push(std::thread::spawn(move || {
            let reservation = workspace.reserve_host_bytes(80);
            barrier.wait();
            reservation.is_ok()
        }));
    }
    barrier.wait();
    let admitted = workers
        .into_iter()
        .map(|worker| usize::from(worker.join().unwrap()))
        .sum::<usize>();
    assert_eq!(admitted, 1);
    let snapshot = workspace.snapshot();
    assert_eq!(snapshot.live_host_working_bytes, 0);
    assert!(snapshot.peak_modeled_host_bytes <= snapshot.host_admission_limit_bytes);
}

#[test]
fn dense_workload_reports_full_independent_f64_reference() {
    let target = target();
    let precisions = if target == ExecutionTarget::Metal {
        vec![Precision::F32]
    } else {
        vec![Precision::F32, Precision::F64]
    };
    for precision in precisions {
        let report = matrix_workload::run(MatrixWorkloadConfig {
            target,
            precision,
            m: 257,
            k: 193,
            n: 129,
            repeats: 2,
            budget: MatrixBudget::default(),
        })
        .unwrap();
        assert!(report.passed, "max error {}", report.max_absolute_error);
        assert_eq!(report.fixture, "dense_deterministic_nondyadic");
        assert_eq!(
            report.reference_method,
            "independent_scalar_f64_dot_products"
        );
        assert_eq!(report.warm_matmul_seconds.len(), 2);
        assert_eq!(report.source_sha256.len(), 64);
        assert_eq!(report.input_sha256.len(), 64);
        if precision == Precision::F64 {
            assert!(report.max_absolute_error < 1e-10);
        }
        if matches!(target, ExecutionTarget::Cuda { .. }) {
            assert_eq!(
                report.matmul_compute_policy,
                "cubek_simple_unit_full_precision"
            );
        }
    }
}

#[test]
fn larger_workload_uses_declared_closed_form_for_every_output() {
    let report = matrix_workload::run(MatrixWorkloadConfig {
        target: target(),
        precision: Precision::F32,
        m: 257,
        k: 257,
        n: 257,
        repeats: 1,
        budget: MatrixBudget::default(),
    })
    .unwrap();
    assert!(report.passed, "max error {}", report.max_absolute_error);
    assert_eq!(report.fixture, "structured_rank_two_dyadic");
    assert_eq!(
        report.reference_method,
        "independent_structured_closed_form_f64"
    );
    assert_eq!(report.mismatched_values, 0);
    assert_eq!(report.modeled_memory_at_readback.live_tensor_count, 3);
}

fn values(count: usize, seed: usize) -> Vec<f64> {
    (0..count)
        .map(|index| ((index * seed + 5) % 31) as f64 / 8.0 - 1.75)
        .collect()
}

fn independent_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut output = vec![0.; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[i * k + inner] * b[inner * n + j];
            }
            output[i * n + j] = sum;
        }
    }
    output
}
