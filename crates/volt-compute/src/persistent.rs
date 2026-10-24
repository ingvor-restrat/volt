use crate::{
    ComputeError, ExecutionTarget, OptionRow, Precision, ResidentOptions, Session,
    budget::{MemoryBudget, MemoryPlan},
    chunked::{ChunkedStats, ChunkedValuation, CompensatedSum, execution_guard, plan_rows},
};
use std::time::Instant;

/// Prepared, immutable portfolio inputs on a live device. Admission must fit the
/// complete portfolio in one resident batch. Streaming jobs use chunked::value_rows.
/// Borrowing the source rows keeps provenance stable while resident inputs live.
pub struct PreparedValuation<'a> {
    rows: &'a [OptionRow],
    session: Session,
    resident: ResidentOptions,
    terms: Vec<(f64, f64)>,
    plan: MemoryPlan,
    budget: MemoryBudget,
    pub packing_and_upload_seconds: f64,
}

impl<'a> PreparedValuation<'a> {
    pub fn new(
        rows: &'a [OptionRow],
        target: ExecutionTarget,
        precision: Precision,
        budget: &MemoryBudget,
    ) -> Result<Self, ComputeError> {
        let plan = plan_rows(rows, target, precision, budget)?;
        if plan.chunk_rows != rows.len() {
            return Err(ComputeError::BudgetExceeded(format!(
                "persistent valuation needs {} rows resident; plan admits {}. Increase the allowance or use chunked execution",
                rows.len(),
                plan.chunk_rows
            )));
        }
        if rows.iter().any(|row| match precision {
            Precision::F32 => !(row.spot as f32).is_normal(),
            Precision::F64 => !row.spot.is_normal(),
        }) {
            return Err(ComputeError::InvalidInput(
                "persistent input spots must be positive normal values at device precision".into(),
            ));
        }
        let _guard = execution_guard()?;
        let session = Session::new(target, precision, rows.len())?;
        let started = Instant::now();
        let resident = ResidentOptions::upload(&session, rows)?;
        session.synchronize()?;
        let packing_and_upload_seconds = started.elapsed().as_secs_f64();
        check_allocator(&session, budget)?;
        // This replaces the validation index (16 bytes/row) after it is dropped;
        // ordered output + terms stay within the global host allowance in plan.
        let terms = rows
            .iter()
            .map(|row| (row.quantity as f64, row.multiplier as f64))
            .collect();
        Ok(Self {
            rows,
            session,
            resident,
            terms,
            plan,
            budget: budget.clone(),
            packing_and_upload_seconds,
        })
    }

    pub fn plan(&self) -> &MemoryPlan {
        &self.plan
    }
    pub fn row_ids(&self) -> &[String] {
        self.resident.row_ids()
    }
    pub fn source_rows(&self) -> &[OptionRow] {
        self.rows
    }

    /// Fully materialized result: GPU evaluation, synchronized readback, and
    /// deterministic host aggregation are all included. No input re-upload.
    pub fn evaluate(&self, spot_scale: f64) -> Result<ChunkedValuation, ComputeError> {
        let _guard = execution_guard()?;
        let mut stats = ChunkedStats::default();
        let started = Instant::now();
        let marks = self.resident.prices_with_spot_scale(spot_scale)?;
        #[cfg(feature = "fusion")]
        self.session.synchronize()?;
        let diagnostic = self.resident.portfolio_value(marks.clone());
        self.session.synchronize()?;
        stats.evaluation_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let prices = self.session.read_vector(&marks)?;
        let diagnostic = self.session.read_vector(&diagnostic)?;
        stats.readback_seconds = started.elapsed().as_secs_f64();
        if prices.len() != self.terms.len() || diagnostic.len() != 1 {
            return Err(ComputeError::Execution(
                "unexpected persistent output shape".into(),
            ));
        }
        let mut sum = CompensatedSum::default();
        for (price, &(quantity, multiplier)) in prices.iter().zip(&self.terms) {
            let value = price * quantity * multiplier;
            if !price.is_finite() || !value.is_finite() {
                return Err(ComputeError::Execution(
                    "nonfinite persistent valuation".into(),
                ));
            }
            sum.add(value);
        }
        let portfolio_value = sum.value();
        if !portfolio_value.is_finite() {
            return Err(ComputeError::Execution(
                "portfolio aggregate overflow".into(),
            ));
        }
        check_allocator(&self.session, &self.budget)?;
        if let Some(usage) = self.session.device().memory_pool_usage() {
            stats.max_sampled_allocator_live_bytes = Some(usage.bytes_in_use);
            stats.max_sampled_allocator_reserved_bytes = Some(usage.bytes_reserved);
        }
        stats.completed_chunks = 1;
        Ok(ChunkedValuation {
            prices,
            portfolio_value,
            device_reduced_value: diagnostic[0].is_finite().then_some(diagnostic[0]),
            plan: self.plan.clone(),
            stats,
        })
    }
}

fn check_allocator(session: &Session, budget: &MemoryBudget) -> Result<(), ComputeError> {
    if let Some(usage) = session.device().memory_pool_usage()
        && usage.bytes_reserved > budget.compute_bytes
    {
        return Err(ComputeError::BudgetExceeded(format!(
            "device allocator reserved {} bytes, above allowance {}",
            usage.bytes_reserved, budget.compute_bytes
        )));
    }
    Ok(())
}
