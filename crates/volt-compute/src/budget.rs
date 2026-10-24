use crate::{ComputeError, ExecutionTarget, OptionRow, Precision};
use serde::{Deserialize, Serialize};

const MIB: u64 = 1024 * 1024;

/// Workload admission estimates, not OS process limits or reservations against
/// other GPU users. Autodiff includes retained activations and backward storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryBudget {
    pub compute_bytes: u64,
    pub host_working_bytes: u64,
    pub compute_reserve_bytes: u64,
    pub headroom_percent: u8,
    pub max_batch_rows: usize,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            compute_bytes: 256 * MIB,
            host_working_bytes: 256 * MIB,
            compute_reserve_bytes: 32 * MIB,
            headroom_percent: 20,
            max_batch_rows: 65_536,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub schema: String,
    pub total_rows: usize,
    pub chunk_rows: usize,
    pub chunks: usize,
    pub estimated_compute_bytes: u64,
    pub estimated_host_working_bytes: u64,
    pub estimated_compute_bytes_per_row: u64,
    pub estimated_host_bytes_per_chunk_row: u64,
    pub host_global_bytes: u64,
    pub compute_admission_limit_bytes: u64,
    pub host_admission_limit_bytes: u64,
}

/// Conservative row-local storage model. This does not cover path-dependent
/// simulation graphs; those need a model that also accounts for time steps.
#[derive(Debug, Clone, Copy)]
pub struct WorkloadMemoryModel {
    pub scalar_slots_per_row: u64,
    pub host_staging_bytes_per_row: u64,
    pub host_output_bytes_per_row: u64,
}

impl WorkloadMemoryModel {
    pub const VALUATION: Self = Self {
        scalar_slots_per_row: 64,
        host_staging_bytes_per_row: 256,
        host_output_bytes_per_row: 8,
    };

    /// Black–Scholes reverse graph, adjoints, and six returned f64 arrays.
    pub const BLACK_SCHOLES_AAD: Self = Self {
        scalar_slots_per_row: 256,
        host_staging_bytes_per_row: 512,
        host_output_bytes_per_row: 48,
    };

    /// Sequential reverse-mode first derivatives and second-order spot jet.
    pub const BLACK_SCHOLES_AAD_GAMMA: Self = Self {
        scalar_slots_per_row: 384,
        host_staging_bytes_per_row: 768,
        host_output_bytes_per_row: 56,
    };
}

impl MemoryBudget {
    pub fn plan(
        &self,
        rows: &[OptionRow],
        target: ExecutionTarget,
        precision: Precision,
    ) -> Result<MemoryPlan, ComputeError> {
        self.plan_for_workload(rows, target, precision, WorkloadMemoryModel::VALUATION)
    }

    pub fn plan_for_workload(
        &self,
        rows: &[OptionRow],
        target: ExecutionTarget,
        precision: Precision,
        model: WorkloadMemoryModel,
    ) -> Result<MemoryPlan, ComputeError> {
        self.validate()?;
        if model.scalar_slots_per_row == 0
            || model.host_staging_bytes_per_row == 0
            || model.host_output_bytes_per_row == 0
        {
            return Err(ComputeError::InvalidInput(
                "workload memory dimensions must be positive".into(),
            ));
        }
        if rows.is_empty() {
            return Err(ComputeError::InvalidInput(
                "valuation needs at least one row".into(),
            ));
        }
        if target == ExecutionTarget::Metal && precision == Precision::F64 {
            return Err(ComputeError::Unsupported(
                "Metal does not provide native f64 arithmetic".into(),
            ));
        }
        let count = u64::try_from(rows.len()).map_err(|_| overflow())?;
        let scalar_bytes = match precision {
            Precision::F32 => 4,
            Precision::F64 => 8,
        };
        // Eight input arrays plus conservative allowance for unfused expression
        // temporaries, outputs, reduction workspaces, and repeated evaluations.
        let compute_per_row = model
            .scalar_slots_per_row
            .checked_mul(scalar_bytes)
            .ok_or_else(overflow)?;
        let max_id_bytes = rows.iter().map(|row| row.row_id.len()).max().unwrap_or(0) as u64;
        // Staging f64 arrays, readback, cloned row IDs and validation structures.
        let host_staging_per_row = model
            .host_staging_bytes_per_row
            .checked_add(max_id_bytes)
            .ok_or_else(overflow)?;
        // Full ordered output plus a borrowed-key index for global duplicate
        // validation. Caller-owned input rows and identifier storage are excluded.
        let global_host = count
            .checked_mul(
                model
                    .host_output_bytes_per_row
                    .checked_add(std::mem::size_of::<&str>() as u64)
                    .ok_or_else(overflow)?,
            )
            .and_then(|n| n.checked_add(8192))
            .ok_or_else(overflow)?;
        let compute_limit = usable(self.compute_bytes, self.headroom_percent)?;
        let host_limit = usable(self.host_working_bytes, self.headroom_percent)?;
        let compute_rows = compute_limit
            .checked_sub(self.compute_reserve_bytes)
            .ok_or_else(|| {
                ComputeError::BudgetExceeded("compute reserve consumes the admission limit".into())
            })?
            / compute_per_row;
        // CPU and Metal compute share physical host RAM. Count numeric working
        // storage in both admission checks while avoiding a double physical sum.
        let shared_memory = matches!(target, ExecutionTarget::Cpu | ExecutionTarget::Metal);
        let host_fixed = global_host
            .checked_add(if shared_memory {
                self.compute_reserve_bytes
            } else {
                0
            })
            .ok_or_else(overflow)?;
        let host_per_row = host_staging_per_row
            .checked_add(if shared_memory { compute_per_row } else { 0 })
            .ok_or_else(overflow)?;
        let host_rows = host_limit.checked_sub(host_fixed)
            .ok_or_else(|| ComputeError::BudgetExceeded("ordered output, validation index, and shared reserve exceed host budget; use a streaming caller".into()))? / host_per_row;
        let chunk_rows = count
            .min(compute_rows)
            .min(host_rows)
            .min(self.max_batch_rows as u64);
        if chunk_rows == 0 {
            return Err(ComputeError::BudgetExceeded(
                "budget cannot admit one valuation row with working space".into(),
            ));
        }
        Ok(MemoryPlan {
            schema: "volt.compute.memory_plan.v1".into(),
            total_rows: rows.len(),
            chunk_rows: chunk_rows as usize,
            chunks: rows.len().div_ceil(chunk_rows as usize),
            estimated_compute_bytes: self.compute_reserve_bytes + chunk_rows * compute_per_row,
            estimated_host_working_bytes: host_fixed + chunk_rows * host_per_row,
            estimated_compute_bytes_per_row: compute_per_row,
            estimated_host_bytes_per_chunk_row: host_per_row,
            host_global_bytes: host_fixed,
            compute_admission_limit_bytes: compute_limit,
            host_admission_limit_bytes: host_limit,
        })
    }

    fn validate(&self) -> Result<(), ComputeError> {
        if self.headroom_percent >= 100
            || self.max_batch_rows == 0
            || self.compute_bytes == 0
            || self.host_working_bytes == 0
        {
            return Err(ComputeError::InvalidInput(
                "budgets and row cap must be positive; headroom must be below 100 percent".into(),
            ));
        }
        Ok(())
    }
}

fn usable(bytes: u64, headroom: u8) -> Result<u64, ComputeError> {
    // Division before multiplication avoids overflow at u64::MAX.
    let pct = 100 - u64::from(headroom);
    (bytes / 100)
        .checked_mul(pct)
        .and_then(|n| n.checked_add((bytes % 100) * pct / 100))
        .ok_or_else(overflow)
}

fn overflow() -> ComputeError {
    ComputeError::InvalidInput("memory estimate overflow".into())
}
