use crate::{ComputeError, OptionRow, Precision, chunked::ChunkedValuation};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{self, Write};

#[derive(Debug, Serialize)]
pub struct ValidationSummary {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub max_absolute_price_error: f64,
    pub mismatched_prices: usize,
    pub reference_portfolio_value: f64,
    pub portfolio_absolute_error: f64,
    pub portfolio_tolerance: f64,
    pub passed: bool,
}

/// Stream rows into the provenance digest without building another full JSON
/// copy. The domain, row count, row order, and JSON record boundaries are explicit.
pub fn input_hash(rows: &[OptionRow]) -> Result<String, ComputeError> {
    let mut writer = DigestWriter(Sha256::new());
    writer.0.update(b"volt.compute.numeric_input.v1\0");
    writer.0.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        serde_json::to_writer(&mut writer, row)
            .map_err(|e| ComputeError::InvalidInput(e.to_string()))?;
        writer.0.update(b"\n");
    }
    Ok(format!("{:x}", writer.0.finalize()))
}

pub fn validate_result(
    rows: &[OptionRow],
    result: &ChunkedValuation,
    precision: Precision,
) -> Result<ValidationSummary, ComputeError> {
    validate_scaled_result(rows, result, precision, 1.0)
}

pub fn validate_scaled_result(
    rows: &[OptionRow],
    result: &ChunkedValuation,
    precision: Precision,
    spot_scale: f64,
) -> Result<ValidationSummary, ComputeError> {
    let (absolute_tolerance, relative_tolerance) = match precision {
        Precision::F32 => (0.0002, 0.0001),
        Precision::F64 => (1e-10, 1e-10),
    };
    let mut max_absolute_price_error = 0.0_f64;
    let mut mismatched_prices = 0;
    let mut reference_sum = crate::chunked::CompensatedSum::default();
    let mut portfolio_tolerance = 0.0;
    let mut gross = 0.0;
    for (row, observed) in rows.iter().zip(&result.prices) {
        let expected = crate::pricing::reference_price_at_spot_scale(row, spot_scale)?;
        let error = (observed - expected).abs();
        let tolerance = absolute_tolerance + relative_tolerance * expected.abs();
        max_absolute_price_error = max_absolute_price_error.max(error);
        mismatched_prices += usize::from(!observed.is_finite() || error > tolerance);
        let weight = row.quantity as f64 * row.multiplier as f64;
        let position_value = expected * row.quantity as f64 * row.multiplier as f64;
        reference_sum.add(position_value);
        gross += position_value.abs();
        portfolio_tolerance += tolerance * weight.abs();
        if !position_value.is_finite() || !gross.is_finite() || !portfolio_tolerance.is_finite() {
            return Err(ComputeError::Execution(
                "reference valuation or tolerance exceeds finite f64 range".into(),
            ));
        }
    }
    portfolio_tolerance += gross
        * match precision {
            Precision::F32 => 1e-5,
            Precision::F64 => 1e-12,
        };
    let reference_portfolio_value = reference_sum.value();
    if !reference_portfolio_value.is_finite() || !portfolio_tolerance.is_finite() {
        return Err(ComputeError::Execution(
            "reference portfolio or tolerance overflow".into(),
        ));
    }
    let portfolio_absolute_error = (reference_portfolio_value - result.portfolio_value).abs();
    let passed = rows.len() == result.prices.len()
        && mismatched_prices == 0
        && portfolio_absolute_error.is_finite()
        && portfolio_absolute_error <= portfolio_tolerance;
    Ok(ValidationSummary {
        absolute_tolerance,
        relative_tolerance,
        max_absolute_price_error,
        mismatched_prices,
        reference_portfolio_value,
        portfolio_absolute_error,
        portfolio_tolerance,
        passed,
    })
}

struct DigestWriter(Sha256);
impl Write for DigestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.update(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
