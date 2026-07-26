//! Convert declared contract quantities into Volt-owned canonical graph samples.
//!
//! The command reads one JSON request and writes one JSON result:
//!
//! `cargo run --bin graph_export -- --input request.json --output result.json`

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use volt::{
    CanonicalPortfolio, ContractId, Currency, GraphContext, OptionContract,
    PortfolioGraphSample, Position, PriceMicros, TimestampNs, UnderlyingId,
};

const REQUEST_SCHEMA_VERSION: &str = "volt.graph_export_request.v1";
const RESULT_SCHEMA_VERSION: &str = "volt.graph_export_result.v1";

#[derive(Debug, Deserialize)]
struct MarkedContract {
    contract: OptionContract,
    mark_micros: i64,
}

#[derive(Debug, Deserialize)]
struct RequestedPosition {
    contract_id: ContractId,
    quantity: i64,
}

#[derive(Debug, Deserialize)]
struct RequestedCandidate {
    source_candidate_id: String,
    positions: Vec<RequestedPosition>,
}

#[derive(Debug, Deserialize)]
struct GraphExportRequest {
    schema_version: String,
    underlying: UnderlyingId,
    currency: Currency,
    valuation_time_ns: i64,
    expiry_ns: i64,
    reference_spot_micros: i64,
    contracts: Vec<MarkedContract>,
    candidates: Vec<RequestedCandidate>,
}

#[derive(Debug, Serialize)]
struct ExportedCandidate {
    source_candidate_id: String,
    sample: PortfolioGraphSample,
}

#[derive(Debug, Serialize)]
struct GraphExportResult {
    schema_version: String,
    records: Vec<ExportedCandidate>,
}

fn export_graphs(
    request: GraphExportRequest,
) -> Result<GraphExportResult, Box<dyn std::error::Error>> {
    if request.schema_version != REQUEST_SCHEMA_VERSION {
        return Err(format!(
            "unsupported graph-export request schema {}",
            request.schema_version
        )
        .into());
    }
    if request.contracts.is_empty() || request.candidates.is_empty() {
        return Err("graph export requires contracts and candidates".into());
    }

    let mut contracts = BTreeMap::new();
    let mut marks = BTreeMap::new();
    for marked in request.contracts {
        marked.contract.validate()?;
        if marked.mark_micros < 0 {
            return Err("graph marks must be non-negative".into());
        }
        let contract_id = marked.contract.contract_id.clone();
        if contracts
            .insert(contract_id.clone(), marked.contract)
            .is_some()
        {
            return Err(format!("duplicate contract {}", contract_id.0).into());
        }
        marks.insert(contract_id, PriceMicros(marked.mark_micros));
    }
    let context = GraphContext {
        underlying: request.underlying,
        currency: request.currency,
        valuation_time: TimestampNs(request.valuation_time_ns),
        expiry: TimestampNs(request.expiry_ns),
        reference_spot: PriceMicros(request.reference_spot_micros),
        marks,
    };
    context.validate()?;

    let mut records = Vec::with_capacity(request.candidates.len());
    for candidate in request.candidates {
        if candidate.source_candidate_id.trim().is_empty() {
            return Err("source candidate id must not be empty".into());
        }
        let positions = candidate
            .positions
            .into_iter()
            .map(|position| {
                let contract = contracts.get(&position.contract_id).ok_or_else(|| {
                    format!(
                        "candidate {} references unknown contract {}",
                        candidate.source_candidate_id, position.contract_id.0
                    )
                })?;
                Position::new(contract.clone(), position.quantity)
                    .map_err(|error| -> Box<dyn std::error::Error> { Box::new(error) })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let portfolio = CanonicalPortfolio::from_positions(positions)?;
        let sample = PortfolioGraphSample::from_portfolio(&portfolio, &context, None)?;
        records.push(ExportedCandidate {
            source_candidate_id: candidate.source_candidate_id,
            sample,
        });
    }
    Ok(GraphExportResult {
        schema_version: RESULT_SCHEMA_VERSION.to_owned(),
        records,
    })
}

struct Options {
    input: PathBuf,
    output: PathBuf,
}

fn options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let mut input = None;
    let mut output = None;
    while let Some(flag) = args.next() {
        if flag == "--input" {
            input = Some(PathBuf::from(args.next().ok_or("--input requires a path")?));
        } else if flag == "--output" {
            output = Some(PathBuf::from(args.next().ok_or("--output requires a path")?));
        } else {
            return Err("usage: graph_export --input REQUEST.json --output RESULT.json".into());
        }
    }
    Ok(Options {
        input: input.ok_or("--input is required")?,
        output: output.ok_or("--output is required")?,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = options()?;
    let request: GraphExportRequest =
        serde_json::from_reader(BufReader::new(File::open(options.input)?))?;
    let result = export_graphs(request)?;
    serde_json::to_writer_pretty(BufWriter::new(File::create(options.output)?), &result)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt::{ExerciseSchedule, OptionRight, Settlement};

    fn contract(id: &str, strike: i64, expiry: i64) -> OptionContract {
        OptionContract::new(
            id,
            "SPY",
            "USD",
            OptionRight::Call,
            PriceMicros(strike),
            100,
            ExerciseSchedule::American {
                first_exercise: TimestampNs(10),
                expiry: TimestampNs(expiry),
            },
            Settlement::Physical,
        )
        .unwrap()
    }

    #[test]
    fn export_uses_volt_canonicalization() {
        let expiry = 100;
        let request = GraphExportRequest {
            schema_version: REQUEST_SCHEMA_VERSION.to_owned(),
            underlying: UnderlyingId("SPY".to_owned()),
            currency: Currency("USD".to_owned()),
            valuation_time_ns: 10,
            expiry_ns: expiry,
            reference_spot_micros: 100_000_000,
            contracts: vec![
                MarkedContract {
                    contract: contract("late", 110_000_000, expiry),
                    mark_micros: 2_500_000,
                },
                MarkedContract {
                    contract: contract("early", 90_000_000, expiry),
                    mark_micros: 12_000_000,
                },
            ],
            candidates: vec![RequestedCandidate {
                source_candidate_id: "source-order".to_owned(),
                positions: vec![
                    RequestedPosition {
                        contract_id: ContractId("late".to_owned()),
                        quantity: -1,
                    },
                    RequestedPosition {
                        contract_id: ContractId("early".to_owned()),
                        quantity: 1,
                    },
                ],
            }],
        };

        let result = export_graphs(request).unwrap();

        assert_eq!(result.records.len(), 1);
        assert_eq!(result.records[0].source_candidate_id, "source-order");
        let position_ids = result.records[0]
            .sample
            .nodes
            .iter()
            .filter_map(|node| match &node.payload {
                volt::GraphNodePayload::OptionPosition { contract_id, .. } => {
                    Some(contract_id.0.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(position_ids, vec!["early", "late"]);
    }
}
