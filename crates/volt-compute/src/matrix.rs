use crate::{ComputeError, ExecutionTarget, Precision, Session};
use burn::tensor::{DType, Tensor, TensorCreationOptions, TensorData};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, MutexGuard};

/// Checked row-major host exchange with persistent, device-owned tensor handles.
/// Upload and download are explicit; matmul intermediates remain on the device.
pub struct ResidentMatrix {
    tensor: Tensor<2>,
    rows: usize,
    columns: usize,
    target: ExecutionTarget,
}

impl ResidentMatrix {
    /// Bound an individual allocation to 128 MiB at f64. This is a per-tensor
    /// bound, not a claim about peak workspace or total resident device memory.
    pub const MAX_ELEMENTS: usize = 16_777_216;

    pub fn upload(
        session: &Session,
        rows: usize,
        columns: usize,
        values: Vec<f64>,
    ) -> Result<Self, ComputeError> {
        let count = checked_elements(rows, columns)?;
        if values.len() != count || values.iter().any(|v| !v.is_finite()) {
            return Err(ComputeError::InvalidInput(
                "matrix length must match shape and all values must be finite".into(),
            ));
        }
        if session.precision() == Precision::F32 && values.iter().any(|v| !(*v as f32).is_finite())
        {
            return Err(ComputeError::InvalidInput(
                "matrix values overflow f32".into(),
            ));
        }
        let dtype = match session.precision() {
            Precision::F32 => DType::F32,
            Precision::F64 => DType::F64,
        };
        let tensor = Tensor::from_data(
            TensorData::new(values, [rows, columns]),
            TensorCreationOptions::new(session.device().clone()).with_dtype(dtype),
        );
        Ok(Self {
            tensor,
            rows,
            columns,
            target: session.target(),
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self, ComputeError> {
        if self.columns != rhs.rows
            || self.tensor.device() != rhs.tensor.device()
            || self.tensor.dtype() != rhs.tensor.dtype()
        {
            return Err(ComputeError::InvalidInput(
                "matrix product requires compatible shapes, devices, and dtypes".into(),
            ));
        }
        checked_elements(self.rows, rhs.columns)?;
        let tensor = if matches!(self.target, ExecutionTarget::Cuda { .. }) {
            #[cfg(all(feature = "cuda", not(feature = "fusion")))]
            {
                strict_cuda_matmul(self.tensor.clone(), rhs.tensor.clone())?
            }
            #[cfg(any(not(feature = "cuda"), feature = "fusion"))]
            {
                return Err(ComputeError::Unsupported(
                    "strict CUDA matrix multiplication requires an unfused CUDA build".into(),
                ));
            }
        } else {
            self.tensor.clone().matmul(rhs.tensor.clone())
        };
        Ok(Self {
            tensor,
            rows: self.rows,
            columns: rhs.columns,
            target: self.target,
        })
    }

    pub fn download(&self) -> Result<Vec<f64>, ComputeError> {
        self.tensor
            .try_to_vec_as::<f64>()
            .map_err(|e| ComputeError::Execution(e.to_string()))
    }
}

/// CubeK's automatic accelerated path changes f32 register/stage operands to
/// TF32 on NVIDIA. The explicit register/unit strategy preserves the requested
/// arithmetic dtype. This extension is deliberately non-autodiff; extracting an
/// autodiff primitive would fail rather than silently severing its graph.
#[cfg(all(feature = "cuda", not(feature = "fusion")))]
fn strict_cuda_matmul(left: Tensor<2>, right: Tensor<2>) -> Result<Tensor<2>, ComputeError> {
    use burn::backend::Cube;
    use cubecl::prelude::Scalar;
    use cubek::{
        matmul::{definition::MatmulElems, multi_level, strategy::Strategy},
        std::InputBinding,
    };
    let dtype = left.dtype();
    let device = left.device();
    let shape = [left.dims()[0], right.dims()[1]];
    let element = match dtype {
        DType::F32 => f32::elem_type_native(),
        DType::F64 => f64::elem_type_native(),
        _ => {
            return Err(ComputeError::Unsupported(
                "strict CUDA matmul supports f32 and f64".into(),
            ));
        }
    };
    let primitive_error = |error| {
        ComputeError::Unsupported(format!(
            "strict CUDA matmul requires unfused non-autodiff Cube tensors: {error:?}"
        ))
    };
    let left = left.try_into_primitive::<Cube>().map_err(primitive_error)?;
    let right = right
        .try_into_primitive::<Cube>()
        .map_err(primitive_error)?;
    let output = Tensor::<2>::empty(shape, TensorCreationOptions::new(device).with_dtype(dtype))
        .try_into_primitive::<Cube>()
        .map_err(primitive_error)?;
    let mut elements = MatmulElems::from_single_dtype(element);
    cubek::matmul::launch::launch_ref(
        &Strategy::MultiLevel(multi_level::Strategy::SimpleUnit(Default::default())),
        &output.client,
        InputBinding::new(left.binding(), element),
        InputBinding::new(right.binding(), element),
        output.clone().binding(),
        &mut elements,
    )
    .map_err(|error| ComputeError::Execution(format!("strict CUDA matmul failed: {error:?}")))?;
    Ok(Tensor::from_primitive::<Cube>(output))
}

pub(crate) fn checked_elements(rows: usize, columns: usize) -> Result<usize, ComputeError> {
    rows.checked_mul(columns)
        .filter(|&n| n > 0 && n <= ResidentMatrix::MAX_ELEMENTS)
        .ok_or_else(|| {
            ComputeError::InvalidInput(
                "matrix dimensions must be positive and product <= 16777216".into(),
            )
        })
}

/// Logical numerical-storage admission. Scratch is a conservative model, not a
/// bound supplied by the backend; JIT/compiler and driver memory are excluded.
/// Other workspaces and raw ResidentMatrix users are not reserved against this
/// budget. Device allocator telemetry is shared by the backend and process.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MatrixBudget {
    pub compute_bytes: u64,
    pub host_working_bytes: u64,
    pub compute_reserve_bytes: u64,
    pub headroom_percent: u8,
    /// Scratch estimate is multiplier times the combined operand/output bytes.
    pub matmul_scratch_multiplier: u32,
}

impl Default for MatrixBudget {
    fn default() -> Self {
        Self {
            compute_bytes: 256 * 1024 * 1024,
            host_working_bytes: 256 * 1024 * 1024,
            compute_reserve_bytes: 32 * 1024 * 1024,
            headroom_percent: 20,
            matmul_scratch_multiplier: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixMemorySnapshot {
    pub schema: String,
    pub live_tensor_count: u64,
    pub live_tensor_bytes: u64,
    pub active_scratch_bytes: u64,
    pub live_host_working_bytes: u64,
    pub modeled_compute_bytes: u64,
    pub modeled_host_bytes: u64,
    pub peak_modeled_compute_bytes: u64,
    pub peak_modeled_host_bytes: u64,
    pub compute_admission_limit_bytes: u64,
    pub host_admission_limit_bytes: u64,
}

#[derive(Default, Clone, Copy)]
struct Accounting {
    tensor_count: u64,
    tensor_bytes: u64,
    scratch_bytes: u64,
    host_bytes: u64,
    peak_compute: u64,
    peak_host: u64,
}

struct WorkspaceInner {
    session: Session,
    budget: MatrixBudget,
    compute_limit: u64,
    host_limit: u64,
    accounting: Mutex<Accounting>,
}

/// Clones share the same admission ledger. Completed uploads and products are
/// synchronized before temporary host/scratch leases are released. Inputs and
/// outputs remain charged until their final handle is dropped.
#[derive(Clone)]
pub struct MatrixWorkspace {
    inner: Arc<WorkspaceInner>,
}

pub struct BudgetedMatrix {
    // Field order releases the tensor before releasing its admission lease.
    matrix: ResidentMatrix,
    lease: Arc<MemoryLease>,
}

/// Host readback remains charged to the workspace while this object is alive.
/// Copies explicitly made by a caller are outside the workspace's ownership.
pub struct DownloadedMatrix {
    values: Vec<f64>,
    shape: [usize; 2],
    _lease: HostMemoryLease,
}

/// Reserve caller-owned numerical storage before allocating it, and keep this
/// guard alive as long as that storage is in use. This models storage; it cannot
/// prevent an arbitrary caller from allocating memory outside the API.
pub struct HostMemoryLease {
    lease: MemoryLease,
}

#[derive(Clone, Copy)]
enum LeaseKind {
    Tensor,
    Scratch,
    Host,
}

struct MemoryLease {
    inner: Arc<WorkspaceInner>,
    bytes: u64,
    kind: LeaseKind,
}

impl MatrixWorkspace {
    pub fn new(
        target: ExecutionTarget,
        precision: Precision,
        budget: MatrixBudget,
    ) -> Result<Self, ComputeError> {
        if budget.compute_bytes == 0
            || budget.host_working_bytes == 0
            || budget.headroom_percent >= 100
            || budget.matmul_scratch_multiplier == 0
        {
            return Err(ComputeError::InvalidInput(
                "matrix budgets and scratch multiplier must be positive and headroom below 100 percent".into(),
            ));
        }
        let compute_limit = usable(budget.compute_bytes, budget.headroom_percent);
        let host_limit = usable(budget.host_working_bytes, budget.headroom_percent);
        let session = Session::new(target, precision, ResidentMatrix::MAX_ELEMENTS)?;
        let inner = Arc::new(WorkspaceInner {
            session,
            budget,
            compute_limit,
            host_limit,
            accounting: Mutex::new(Accounting::default()),
        });
        let mut initial = Accounting::default();
        inner.admit(&mut initial)?;
        *inner.lock() = initial;
        Ok(Self { inner })
    }

    pub fn device_description(&self) -> String {
        format!("{:?}", self.inner.session.device())
    }

    pub fn reserve_host_bytes(&self, bytes: u64) -> Result<HostMemoryLease, ComputeError> {
        Ok(HostMemoryLease {
            lease: self.reserve(LeaseKind::Host, bytes)?,
        })
    }

    pub fn upload(
        &self,
        rows: usize,
        columns: usize,
        values: &[f64],
    ) -> Result<BudgetedMatrix, ComputeError> {
        let elements = checked_elements(rows, columns)?;
        if values.len() != elements || values.iter().any(|v| !v.is_finite()) {
            return Err(ComputeError::InvalidInput(
                "matrix values must match shape and be finite".into(),
            ));
        }
        if self.inner.session.precision() == Precision::F32
            && values.iter().any(|v| !(*v as f32).is_finite())
        {
            return Err(ComputeError::InvalidInput(
                "matrix values overflow f32".into(),
            ));
        }
        let bytes = checked_bytes(elements, self.scalar_bytes())?;
        let staging = self.reserve_host_bytes(checked_bytes(elements, 16)?)?;
        let lease = Arc::new(self.reserve(LeaseKind::Tensor, bytes)?);
        let matrix = ResidentMatrix::upload(&self.inner.session, rows, columns, values.to_vec())?;
        self.synchronize()?;
        drop(staging);
        Ok(BudgetedMatrix { matrix, lease })
    }

    pub fn matmul(
        &self,
        left: &BudgetedMatrix,
        right: &BudgetedMatrix,
    ) -> Result<BudgetedMatrix, ComputeError> {
        self.check_owner(left)?;
        self.check_owner(right)?;
        if left.shape()[1] != right.shape()[0] {
            return Err(ComputeError::InvalidInput(
                "incompatible matrix product shapes".into(),
            ));
        }
        let count = checked_elements(left.shape()[0], right.shape()[1])?;
        let bytes = checked_bytes(count, self.scalar_bytes())?;
        let scratch_bytes = left
            .lease
            .bytes
            .checked_add(right.lease.bytes)
            .and_then(|n| n.checked_add(bytes))
            .and_then(|n| n.checked_mul(u64::from(self.inner.budget.matmul_scratch_multiplier)))
            .ok_or_else(overflow)?;
        // Both charges are established before dispatch. Failure rolls back prior
        // reservations through RAII, including when the product cannot fit.
        let lease = Arc::new(self.reserve(LeaseKind::Tensor, bytes)?);
        let scratch = self.reserve(LeaseKind::Scratch, scratch_bytes)?;
        let matrix = left.matrix.matmul(&right.matrix)?;
        self.synchronize()?;
        drop(scratch);
        Ok(BudgetedMatrix { matrix, lease })
    }

    pub fn download(&self, matrix: &BudgetedMatrix) -> Result<DownloadedMatrix, ComputeError> {
        self.check_owner(matrix)?;
        let count = checked_elements(matrix.shape()[0], matrix.shape()[1])?;
        let mut lease = self.reserve_host_bytes(checked_bytes(count, 16)?)?;
        let values = matrix.matrix.download()?;
        if values.len() != count || values.iter().any(|value| !value.is_finite()) {
            return Err(ComputeError::Execution("invalid matrix readback".into()));
        }
        lease.shrink(checked_bytes(count, 8)?);
        Ok(DownloadedMatrix {
            values,
            shape: matrix.shape(),
            _lease: lease,
        })
    }

    pub fn synchronize(&self) -> Result<(), ComputeError> {
        self.inner.session.synchronize()
    }

    pub fn snapshot(&self) -> MatrixMemorySnapshot {
        let accounting = self.inner.lock();
        let (compute, host) = self
            .inner
            .modeled(&accounting)
            .expect("admitted accounting cannot overflow");
        MatrixMemorySnapshot {
            schema: "volt.compute.matrix_memory.v1".into(),
            live_tensor_count: accounting.tensor_count,
            live_tensor_bytes: accounting.tensor_bytes,
            active_scratch_bytes: accounting.scratch_bytes,
            live_host_working_bytes: accounting.host_bytes,
            modeled_compute_bytes: compute,
            modeled_host_bytes: host,
            peak_modeled_compute_bytes: accounting.peak_compute,
            peak_modeled_host_bytes: accounting.peak_host,
            compute_admission_limit_bytes: self.inner.compute_limit,
            host_admission_limit_bytes: self.inner.host_limit,
        }
    }

    pub fn allocator_usage(&self) -> Option<(u64, u64)> {
        self.inner
            .session
            .device()
            .memory_pool_usage()
            .map(|u| (u.bytes_in_use, u.bytes_reserved))
    }

    fn scalar_bytes(&self) -> u64 {
        match self.inner.session.precision() {
            Precision::F32 => 4,
            Precision::F64 => 8,
        }
    }

    fn check_owner(&self, matrix: &BudgetedMatrix) -> Result<(), ComputeError> {
        if !Arc::ptr_eq(&self.inner, &matrix.lease.inner) {
            return Err(ComputeError::InvalidInput(
                "matrix belongs to a different workspace, device, or precision".into(),
            ));
        }
        Ok(())
    }

    fn reserve(&self, kind: LeaseKind, bytes: u64) -> Result<MemoryLease, ComputeError> {
        let mut accounting = self.inner.lock();
        let mut candidate = *accounting;
        let counter = match kind {
            LeaseKind::Tensor => &mut candidate.tensor_bytes,
            LeaseKind::Scratch => &mut candidate.scratch_bytes,
            LeaseKind::Host => &mut candidate.host_bytes,
        };
        *counter = counter.checked_add(bytes).ok_or_else(overflow)?;
        if matches!(kind, LeaseKind::Tensor) {
            candidate.tensor_count = candidate.tensor_count.checked_add(1).ok_or_else(overflow)?;
        }
        self.inner.admit(&mut candidate)?;
        *accounting = candidate;
        Ok(MemoryLease {
            inner: self.inner.clone(),
            bytes,
            kind,
        })
    }
}

impl BudgetedMatrix {
    pub fn shape(&self) -> [usize; 2] {
        self.matrix.shape()
    }
}

impl DownloadedMatrix {
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl HostMemoryLease {
    fn shrink(&mut self, bytes: u64) {
        assert!(bytes <= self.lease.bytes);
        let mut accounting = self.lease.inner.lock();
        accounting.host_bytes -= self.lease.bytes - bytes;
        self.lease.bytes = bytes;
    }
}

impl WorkspaceInner {
    fn lock(&self) -> MutexGuard<'_, Accounting> {
        self.accounting
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn modeled(&self, a: &Accounting) -> Result<(u64, u64), ComputeError> {
        let compute = self
            .budget
            .compute_reserve_bytes
            .checked_add(a.tensor_bytes)
            .and_then(|n| n.checked_add(a.scratch_bytes))
            .ok_or_else(overflow)?;
        let host = 8192_u64
            .checked_add(a.host_bytes)
            .and_then(|n| {
                n.checked_add(
                    if matches!(
                        self.session.target(),
                        ExecutionTarget::Cpu | ExecutionTarget::Metal
                    ) {
                        compute
                    } else {
                        0
                    },
                )
            })
            .ok_or_else(overflow)?;
        Ok((compute, host))
    }

    fn admit(&self, a: &mut Accounting) -> Result<(), ComputeError> {
        let (compute, host) = self.modeled(a)?;
        if compute > self.compute_limit || host > self.host_limit {
            return Err(ComputeError::BudgetExceeded(format!(
                "matrix admission needs compute {compute}/{} and host {host}/{} bytes",
                self.compute_limit, self.host_limit
            )));
        }
        a.peak_compute = a.peak_compute.max(compute);
        a.peak_host = a.peak_host.max(host);
        Ok(())
    }
}

impl Drop for MemoryLease {
    fn drop(&mut self) {
        let mut accounting = self.inner.lock();
        match self.kind {
            LeaseKind::Tensor => {
                accounting.tensor_bytes -= self.bytes;
                accounting.tensor_count -= 1;
            }
            LeaseKind::Scratch => accounting.scratch_bytes -= self.bytes,
            LeaseKind::Host => accounting.host_bytes -= self.bytes,
        }
    }
}

fn usable(bytes: u64, headroom: u8) -> u64 {
    let percent = 100 - u64::from(headroom);
    bytes / 100 * percent + bytes % 100 * percent / 100
}

fn checked_bytes(elements: usize, bytes: u64) -> Result<u64, ComputeError> {
    u64::try_from(elements)
        .ok()
        .and_then(|n| n.checked_mul(bytes))
        .ok_or_else(overflow)
}

fn overflow() -> ComputeError {
    ComputeError::InvalidInput("matrix memory arithmetic overflow".into())
}
