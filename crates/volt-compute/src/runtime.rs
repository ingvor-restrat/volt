use burn::tensor::{DType, Device, Tensor, TensorCreationOptions, TensorData};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, thiserror::Error)]
pub enum ComputeError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("unsupported configuration: {0}")]
    Unsupported(String),
    #[error("device execution failed: {0}")]
    Execution(String),
    #[error("memory budget exceeded: {0}")]
    BudgetExceeded(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTarget {
    Cpu,
    Metal,
    Cuda { index: usize },
}

impl FromStr for ExecutionTarget {
    type Err = ComputeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda { index: 0 }),
            _ => value
                .strip_prefix("cuda:")
                .and_then(|i| i.parse().ok())
                .map(|index| Self::Cuda { index })
                .ok_or_else(|| {
                    ComputeError::InvalidInput(format!(
                        "unknown device {value:?}; use cpu, metal, or cuda:N"
                    ))
                }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    F32,
    F64,
}

/// A session owns the execution device. Resident tensors retain their allocations
/// across calls; intermediates are managed by Burn and the selected runtime.
pub struct Session {
    device: Device,
    target: ExecutionTarget,
    precision: Precision,
    max_batch_rows: usize,
}

impl Session {
    pub fn new(
        target: ExecutionTarget,
        precision: Precision,
        max_batch_rows: usize,
    ) -> Result<Self, ComputeError> {
        if max_batch_rows == 0 {
            return Err(ComputeError::InvalidInput(
                "max_batch_rows must be positive".into(),
            ));
        }
        if target == ExecutionTarget::Metal && precision == Precision::F64 {
            return Err(ComputeError::Unsupported(
                "Metal does not provide native f64 arithmetic".into(),
            ));
        }
        if cfg!(all(feature = "cuda", feature = "fusion"))
            && matches!(target, ExecutionTarget::Cuda { .. })
        {
            return Err(ComputeError::Unsupported(
                "CUDA fusion is disabled after shape-dependent pricing failed cancellation checks; build with --features cuda without fusion".into(),
            ));
        }
        let device = match target {
            ExecutionTarget::Cpu => Device::flex(),
            ExecutionTarget::Metal => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    Device::metal(Default::default())
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    return Err(ComputeError::Unsupported(
                        "Metal requires macOS and --features metal".into(),
                    ));
                }
            }
            ExecutionTarget::Cuda { index } => {
                #[cfg(feature = "cuda")]
                {
                    Device::cuda(index)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = index;
                    return Err(ComputeError::Unsupported(
                        "CUDA requires --features cuda".into(),
                    ));
                }
            }
        };
        Ok(Self {
            device,
            target,
            precision,
            max_batch_rows,
        })
    }

    pub fn target(&self) -> ExecutionTarget {
        self.target
    }
    pub fn precision(&self) -> Precision {
        self.precision
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn max_batch_rows(&self) -> usize {
        self.max_batch_rows
    }

    pub fn synchronize(&self) -> Result<(), ComputeError> {
        self.device
            .sync()
            .map_err(|e| ComputeError::Execution(e.to_string()))
    }

    pub(crate) fn vector(&self, values: Vec<f64>) -> Tensor<1> {
        let length = values.len();
        let dtype = match self.precision {
            Precision::F32 => DType::F32,
            Precision::F64 => DType::F64,
        };
        Tensor::from_data(
            TensorData::new(values, [length]),
            TensorCreationOptions::new(self.device.clone()).with_dtype(dtype),
        )
    }

    pub fn read_vector(&self, tensor: &Tensor<1>) -> Result<Vec<f64>, ComputeError> {
        if tensor.device() != self.device {
            return Err(ComputeError::InvalidInput(
                "tensor belongs to a different execution device".into(),
            ));
        }
        tensor
            .try_to_vec_as::<f64>()
            .map_err(|e| ComputeError::Execution(e.to_string()))
    }
}

/// Best-effort allocator release on every exit, including failed dispatches.
/// This does not bound reservations belonging to other sessions/processes.
pub(crate) struct DeviceCleanupGuard<'a>(pub(crate) &'a Session);
impl Drop for DeviceCleanupGuard<'_> {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
        self.0.device().memory_cleanup();
    }
}
