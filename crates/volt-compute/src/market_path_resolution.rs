//! Refine simulation time while preserving the policy's observation/trading grid.
//!
//! CIR endpoints remain exact; the joint stock/integral scheme remains simplified
//! Broadie–Kaya with the declared quadrature. Refinement is a sensitivity control,
//! not an exact joint-Heston sampler. Different resolutions are independent
//! discretization couplings of the same counter contract, not Brownian bridges.
use crate::{
    ComputeError,
    market_paths::{self, MARKET_FIELDS, MarketPathConfig, MarketPaths},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarketPathResolutionConfig {
    /// Original policy observation grid, including its original dt and steps.
    pub market: MarketPathConfig,
    pub substeps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionPlan {
    pub fine_config: MarketPathConfig,
    /// Conservative simultaneous host input/fine-output/coarse-output admission.
    pub working_bytes: u64,
}

impl MarketPathResolutionConfig {
    pub fn validate(&self) -> Result<ResolutionPlan, ComputeError> {
        let coarse = self.market.validate()?;
        if self.substeps == 0 {
            return Err(invalid("resolution substeps must be positive"));
        }
        let mut fine = self.market.clone();
        fine.steps = fine
            .steps
            .checked_mul(self.substeps)
            .ok_or_else(|| invalid("resolution step count overflow"))?;
        fine.dt /= self.substeps as f64;
        let shape = fine.validate()?;
        if fine.horizon() != self.market.horizon() {
            return Err(invalid(
                "resolution cannot preserve the exact configured horizon",
            ));
        }
        let bytes = shape
            .input_bytes
            .checked_add(shape.output_bytes)
            .and_then(|b| b.checked_add(coarse.output_bytes))
            .ok_or_else(|| invalid("resolution working byte overflow"))?;
        Ok(ResolutionPlan {
            fine_config: fine,
            working_bytes: bytes,
        })
    }
}

/// Coarse observations plus separately hashed fine-grid provenance. The market
/// adapter can use `market` without changing any network or hedge-date config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolutionPaths {
    pub schema: String,
    pub config: MarketPathResolutionConfig,
    pub fine_config: MarketPathConfig,
    pub fine_content_hash: String,
    pub market: MarketPaths,
    pub provenance_hash: String,
}
impl ResolutionPaths {
    /// Check hashes/shapes then independently regenerate the fine inputs and
    /// verify every selected state. Hashes alone are not sampling provenance.
    pub fn validate(&self) -> Result<ResolutionPlan, ComputeError> {
        let plan = self.validate_metadata()?;
        let fine = market_paths::generate_paths(&plan.fine_config)?;
        if fine.content_hash != self.fine_content_hash
            || fine.inputs_hash != self.market.inputs_hash
            || market_paths::hash_json(&fine.diagnostics)?
                != market_paths::hash_json(&self.market.diagnostics)?
            || self.market.input_source
                != if self.config.substeps == 1 {
                    fine.input_source.clone()
                } else {
                    resolution_source(self.config.substeps, &fine)
                }
        {
            return Err(invalid("resolution fine replay identity mismatch"));
        }
        for path in 0..self.market.config.paths {
            for date in 0..=self.market.config.steps {
                for field in 0..MARKET_FIELDS {
                    if self.market.value(path, date, field)
                        != fine.value(path, date * self.config.substeps, field)
                    {
                        return Err(invalid(
                            "resolution coarse state differs from current fine state",
                        ));
                    }
                }
            }
        }
        Ok(plan)
    }
    fn validate_metadata(&self) -> Result<ResolutionPlan, ComputeError> {
        let plan = self.config.validate()?;
        self.market.validate()?;
        if self.schema != "volt.compute.market_path_resolution.v1"
            || market_paths::hash_json(&plan.fine_config)?
                != market_paths::hash_json(&self.fine_config)?
            || market_paths::hash_json(&self.config.market)? != self.market.config_hash
            || self.fine_content_hash.len() != 64
            || !self
                .fine_content_hash
                .bytes()
                .all(|c| c.is_ascii_hexdigit())
            || self.provenance_hash != provenance(self)?
        {
            return Err(invalid("resolution provenance/config hash mismatch"));
        }
        Ok(plan)
    }
}

pub fn generate_resolved_paths(
    config: &MarketPathResolutionConfig,
) -> Result<ResolutionPaths, ComputeError> {
    let plan = config.validate()?;
    let fine = market_paths::generate_paths(&plan.fine_config)?;
    let fine_content_hash = fine.content_hash.clone();
    let market = if config.substeps == 1 {
        fine
    } else {
        let points = config.market.steps + 1;
        let fine_points = fine.config.steps + 1;
        let mut values = Vec::with_capacity(config.market.paths * points * MARKET_FIELDS);
        for path in 0..config.market.paths {
            for date in 0..points {
                let start = (path * fine_points + date * config.substeps) * MARKET_FIELDS;
                // Select only the current fine state: A contains all preceding
                // substeps, while H is quoted at this date from that A and V.
                values.extend_from_slice(&fine.values[start..start + MARKET_FIELDS]);
            }
        }
        let config_hash = market_paths::hash_json(&config.market)?;
        let input_source = resolution_source(config.substeps, &fine);
        let content_hash = market_paths::hash_output(
            &config_hash,
            &fine.inputs_hash,
            &input_source,
            &fine.diagnostics,
            &values,
        )?;
        MarketPaths {
            schema: fine.schema,
            config: config.market.clone(),
            times: (0..points).map(|i| i as f64 * config.market.dt).collect(),
            values,
            diagnostics: fine.diagnostics,
            rng_contract: fine.rng_contract,
            input_source,
            inputs_hash: fine.inputs_hash,
            config_hash,
            content_hash,
        }
    };
    let mut result = ResolutionPaths {
        schema: "volt.compute.market_path_resolution.v1".into(),
        config: config.clone(),
        fine_config: plan.fine_config,
        fine_content_hash,
        market,
        provenance_hash: String::new(),
    };
    result.provenance_hash = provenance(&result)?;
    result.validate_metadata()?;
    Ok(result)
}

/// Consume generated/resampled systems directly into the existing hedge adapter.
/// Observation counts, features and trade dates stay on the original grid.
/// Includes independent fine-input replay validation, including for deserialized
/// records, before constructing policy features.
pub fn into_hedge_batch(
    paths: Vec<ResolutionPaths>,
    liability: crate::hedge_application::HedgeLiability,
) -> Result<crate::hedge_application::HedgePathBatch, ComputeError> {
    for path in &paths {
        path.validate()?;
    }
    let markets: Vec<_> = paths.into_iter().map(|p| p.market).collect();
    crate::hedge_application::HedgePathBatch::from_markets(&markets, liability)
}

fn provenance(paths: &ResolutionPaths) -> Result<String, ComputeError> {
    market_paths::hash_json(&(
        &paths.schema,
        &paths.config,
        &paths.fine_config,
        &paths.fine_content_hash,
        &paths.market.content_hash,
        &paths.market.inputs_hash,
        &paths.market.rng_contract,
    ))
}
fn resolution_source(substeps: usize, fine: &MarketPaths) -> String {
    format!(
        "resolution_subsample.v1:substeps={substeps};fine_content={};source={}",
        fine.content_hash, fine.input_source
    )
}
fn invalid(message: &str) -> ComputeError {
    ComputeError::InvalidInput(message.into())
}
