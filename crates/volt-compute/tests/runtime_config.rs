#[cfg(all(feature = "cuda", feature = "fusion"))]
#[test]
fn cuda_fusion_is_rejected_before_device_initialization() {
    use volt_compute::{ComputeError, ExecutionTarget, Precision, Session};
    for precision in [Precision::F32, Precision::F64] {
        assert!(matches!(
            Session::new(ExecutionTarget::Cuda { index: 0 }, precision, 10),
            Err(ComputeError::Unsupported(_))
        ));
    }
}
