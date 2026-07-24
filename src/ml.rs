//! Deterministic graph views for machine-learning and optimization systems.
//!
//! This module never replaces the exact portfolio model. It derives a stable,
//! versioned graph whose identifiers retain links to both exact contract
//! semantics and terminal-payoff equivalence.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::canonical::CanonicalPortfolio;
use crate::contract::{
    ContractId, Currency, ExerciseSchedule, OptionRight, PriceMicros, Settlement, TimestampNs,
    UnderlyingId,
};
use crate::hash::canonical_hash;
use crate::{CogError, CogResult};

/// Schema tag included in every [`PortfolioGraphSample`].
pub const GRAPH_SAMPLE_SCHEMA_VERSION: &str = "volt.portfolio_graph.v1";

/// Market context required to turn an exact portfolio into an ML graph view.
///
/// Marks are per underlying unit in price micros, not multiplied contract
/// notionals. Keeping them outside the contract definition prevents market
/// observations from changing contract identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphContext {
    /// Underlying shared by every contract in the sample.
    pub underlying: UnderlyingId,
    /// Settlement currency shared by every contract in the sample.
    pub currency: Currency,
    /// Time at which the sample is valued, in nanoseconds since the Unix epoch.
    pub valuation_time: TimestampNs,
    /// Common contract expiry, in nanoseconds since the Unix epoch.
    pub expiry: TimestampNs,
    /// Underlying reference price in millionths of one currency unit.
    pub reference_spot: PriceMicros,
    /// Per-unit option marks, keyed by exact contract identifier.
    pub marks: BTreeMap<ContractId, PriceMicros>,
}

impl GraphContext {
    /// Checks identifiers, time ordering, spot, and option marks.
    pub fn validate(&self) -> CogResult<()> {
        if self.underlying.0.trim().is_empty() {
            return Err(CogError::InvalidDefinition(
                "graph underlying must not be empty".to_owned(),
            ));
        }
        if self.currency.0.trim().is_empty() {
            return Err(CogError::InvalidDefinition(
                "graph currency must not be empty".to_owned(),
            ));
        }
        if self.valuation_time.0 < 0 || self.expiry.0 < self.valuation_time.0 {
            return Err(CogError::InvalidDefinition(
                "graph times must satisfy 0 <= valuation_time <= expiry".to_owned(),
            ));
        }
        if self.reference_spot.0 <= 0 {
            return Err(CogError::InvalidDefinition(
                "graph reference spot must be positive".to_owned(),
            ));
        }
        if self.marks.values().any(|mark| mark.0 < 0) {
            return Err(CogError::InvalidDefinition(
                "graph option marks must be non-negative".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Optional supervised-learning target attached to a graph sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphTarget {
    /// Human- and machine-readable name of the optimized objective.
    pub objective_name: String,
    /// Scalar objective value produced by the evaluator.
    pub objective: f64,
    /// Scenario spots in millionths of one currency unit.
    pub scenario_spots_micros: Vec<i64>,
    /// Portfolio P&L values aligned one-to-one with `scenario_spots_micros`.
    pub scenario_pnl_micros: Vec<i128>,
    /// Free-form provenance or interpretation note.
    pub note: String,
}

impl GraphTarget {
    /// Checks that the objective is finite and the scenario vectors align.
    pub fn validate(&self) -> CogResult<()> {
        if self.objective_name.trim().is_empty() {
            return Err(CogError::InvalidDefinition(
                "graph objective_name must not be empty".to_owned(),
            ));
        }
        if !self.objective.is_finite() {
            return Err(CogError::InvalidDefinition(
                "graph objective must be finite".to_owned(),
            ));
        }
        if self.scenario_spots_micros.is_empty()
            || self.scenario_spots_micros.len() != self.scenario_pnl_micros.len()
        {
            return Err(CogError::InvalidDefinition(
                "graph target scenario spots and PnL must be non-empty and aligned".to_owned(),
            ));
        }
        if self.scenario_spots_micros.iter().any(|spot| *spot < 0) {
            return Err(CogError::InvalidDefinition(
                "graph scenario spots must be non-negative".to_owned(),
            ));
        }
        Ok(())
    }
}

/// One identified node in a portfolio graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNode {
    /// Identifier unique within its graph sample.
    pub node_id: String,
    /// Typed attributes carried by the node.
    #[serde(flatten)]
    pub payload: GraphNodePayload,
}

/// Typed data carried by a [`GraphNode`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "node_type", rename_all = "snake_case")]
pub enum GraphNodePayload {
    /// Root node summarizing the complete portfolio.
    Portfolio {
        /// Number of non-zero canonical positions.
        leg_count: u32,
        /// Sum of absolute position quantities.
        gross_quantity: u64,
    },
    /// Shared underlying and its reference market context.
    Underlying {
        /// Instrument on which the options are written.
        underlying: UnderlyingId,
        /// Currency used by prices and cash settlement.
        currency: Currency,
        /// Reference spot in millionths of one currency unit.
        reference_spot_micros: i64,
    },
    /// Common expiry context for the graph.
    Expiry {
        /// Expiry in nanoseconds since the Unix epoch.
        expiry_ns: i64,
        /// Non-negative duration from valuation to expiry, in nanoseconds.
        time_to_expiry_ns: i64,
    },
    /// Exact option position represented as model-ready scalar features.
    OptionPosition {
        /// Exact contract identifier.
        contract_id: ContractId,
        /// Call or put right.
        right: OptionRight,
        /// Cash or physical settlement.
        settlement: Settlement,
        /// Coarse exercise-schedule category.
        exercise_style: GraphExerciseStyle,
        /// Strike in millionths of one currency unit per underlying unit.
        strike_micros: i64,
        /// Option mark in millionths of one currency unit per underlying unit.
        mark_micros: i64,
        /// Signed number of contracts.
        quantity: i64,
        /// Underlying units controlled by each contract.
        multiplier: u32,
        /// First possible exercise time, in nanoseconds since the Unix epoch.
        first_exercise_ns: i64,
        /// Number of discrete opportunities, or zero for an American interval.
        exercise_opportunity_count: u32,
    },
}

/// Exercise-schedule category exposed as a graph feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphExerciseStyle {
    /// Exercise is allowed only at expiry.
    European,
    /// Exercise is allowed at a finite set of dates.
    Bermudan,
    /// Exercise is allowed throughout a time interval.
    American,
}

/// Semantic relationship between two graph nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdgeType {
    /// An option-position node belongs to the portfolio root.
    PositionIn,
    /// An option-position node is written on the underlying node.
    WrittenOn,
    /// An option-position node expires at the expiry node.
    ExpiresAt,
}

/// Directed, typed edge in a portfolio graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node identifier.
    pub source: String,
    /// Target node identifier.
    pub target: String,
    /// Meaning of the source-to-target relationship.
    pub edge_type: GraphEdgeType,
}

/// Versioned graph view for optimization and learning consumers.
///
/// This is a derived ML representation. `portfolio_hash` and
/// `terminal_payoff_hash` connect every prediction back to Rust-owned exact
/// semantics and the deliberately weaker terminal-payoff equivalence class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortfolioGraphSample {
    /// Serialization schema used by this sample.
    pub schema_version: String,
    /// Content-derived identifier for graph inputs, excluding the target.
    pub sample_id: String,
    /// Hash of the exact canonical portfolio.
    pub portfolio_hash: String,
    /// Hash of terminal-payoff semantics, absent for an empty portfolio.
    pub terminal_payoff_hash: Option<String>,
    /// Deterministically ordered graph nodes.
    pub nodes: Vec<GraphNode>,
    /// Deterministically ordered graph edges.
    pub edges: Vec<GraphEdge>,
    /// Absent for candidates that have not yet been evaluated by the simulator.
    pub target: Option<GraphTarget>,
}

impl PortfolioGraphSample {
    /// Derives a graph sample from an exact portfolio and compatible context.
    ///
    /// The optional target is validated but deliberately excluded from
    /// `sample_id`, so labels may be revised without changing feature identity.
    pub fn from_portfolio(
        portfolio: &CanonicalPortfolio,
        context: &GraphContext,
        target: Option<GraphTarget>,
    ) -> CogResult<Self> {
        context.validate()?;
        if let Some(target) = &target {
            target.validate()?;
        }

        let portfolio_hash = portfolio.semantic_hash()?;
        let terminal_payoff_hash = if portfolio.positions.is_empty() {
            None
        } else {
            Some(portfolio.terminal_payoff_projection()?.semantic_hash()?)
        };

        let leg_count = u32::try_from(portfolio.positions.len())
            .map_err(|_| CogError::NumericOverflow("graph leg count".to_owned()))?;
        let gross_quantity = portfolio
            .positions
            .iter()
            .try_fold(0_u64, |total, position| {
                total
                    .checked_add(position.quantity.unsigned_abs())
                    .ok_or_else(|| CogError::NumericOverflow("graph gross quantity".to_owned()))
            })?;

        let mut nodes = vec![
            GraphNode {
                node_id: "portfolio".to_owned(),
                payload: GraphNodePayload::Portfolio {
                    leg_count,
                    gross_quantity,
                },
            },
            GraphNode {
                node_id: "underlying".to_owned(),
                payload: GraphNodePayload::Underlying {
                    underlying: context.underlying.clone(),
                    currency: context.currency.clone(),
                    reference_spot_micros: context.reference_spot.0,
                },
            },
            GraphNode {
                node_id: "expiry".to_owned(),
                payload: GraphNodePayload::Expiry {
                    expiry_ns: context.expiry.0,
                    time_to_expiry_ns: context
                        .expiry
                        .0
                        .checked_sub(context.valuation_time.0)
                        .ok_or_else(|| {
                            CogError::NumericOverflow("graph time to expiry".to_owned())
                        })?,
                },
            },
        ];
        let mut edges = Vec::with_capacity(portfolio.positions.len() * 3);

        for (index, position) in portfolio.positions.iter().enumerate() {
            let contract = &position.contract;
            if contract.underlying != context.underlying
                || contract.currency != context.currency
                || contract.expiry() != context.expiry
            {
                return Err(CogError::IncompatiblePortfolio(
                    "graph context and portfolio contract do not match".to_owned(),
                ));
            }
            let mark = context.marks.get(&contract.contract_id).ok_or_else(|| {
                CogError::InvalidDefinition(format!(
                    "missing graph mark for contract {}",
                    contract.contract_id.0
                ))
            })?;
            let (exercise_style, first_exercise_ns, exercise_opportunity_count) =
                exercise_features(&contract.exercise)?;
            let node_id = format!("position:{index}");
            nodes.push(GraphNode {
                node_id: node_id.clone(),
                payload: GraphNodePayload::OptionPosition {
                    contract_id: contract.contract_id.clone(),
                    right: contract.right,
                    settlement: contract.settlement,
                    exercise_style,
                    strike_micros: contract.strike.0,
                    mark_micros: mark.0,
                    quantity: position.quantity,
                    multiplier: contract.multiplier,
                    first_exercise_ns,
                    exercise_opportunity_count,
                },
            });
            edges.extend([
                GraphEdge {
                    source: node_id.clone(),
                    target: "portfolio".to_owned(),
                    edge_type: GraphEdgeType::PositionIn,
                },
                GraphEdge {
                    source: node_id.clone(),
                    target: "underlying".to_owned(),
                    edge_type: GraphEdgeType::WrittenOn,
                },
                GraphEdge {
                    source: node_id,
                    target: "expiry".to_owned(),
                    edge_type: GraphEdgeType::ExpiresAt,
                },
            ]);
        }

        #[derive(Serialize)]
        struct SampleIdentity<'a> {
            portfolio_hash: &'a str,
            terminal_payoff_hash: &'a Option<String>,
            nodes: &'a [GraphNode],
            edges: &'a [GraphEdge],
        }
        let sample_id = canonical_hash(
            GRAPH_SAMPLE_SCHEMA_VERSION,
            &SampleIdentity {
                portfolio_hash: &portfolio_hash,
                terminal_payoff_hash: &terminal_payoff_hash,
                nodes: &nodes,
                edges: &edges,
            },
        )?;

        Ok(Self {
            schema_version: GRAPH_SAMPLE_SCHEMA_VERSION.to_owned(),
            sample_id,
            portfolio_hash,
            terminal_payoff_hash,
            nodes,
            edges,
            target,
        })
    }
}

fn exercise_features(schedule: &ExerciseSchedule) -> CogResult<(GraphExerciseStyle, i64, u32)> {
    match schedule {
        ExerciseSchedule::European { expiry } => Ok((GraphExerciseStyle::European, expiry.0, 1)),
        ExerciseSchedule::American { first_exercise, .. } => {
            // Zero denotes an interval rather than a finite set of dates.
            Ok((GraphExerciseStyle::American, first_exercise.0, 0))
        }
        ExerciseSchedule::Bermudan { opportunities } => Ok((
            GraphExerciseStyle::Bermudan,
            opportunities
                .first()
                .ok_or_else(|| {
                    CogError::InvalidDefinition(
                        "bermudan graph schedule has no opportunities".to_owned(),
                    )
                })?
                .0,
            u32::try_from(opportunities.len())
                .map_err(|_| CogError::NumericOverflow("exercise opportunities".to_owned()))?,
        )),
    }
}
