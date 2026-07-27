//! Deterministic graph views for machine-learning and optimization systems.
//!
//! This module never replaces the exact portfolio model. It derives a stable,
//! versioned graph whose identifiers retain links to both exact contract
//! semantics and terminal-payoff equivalence.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::canonical::CanonicalPortfolio;
use crate::compiler::{CompiledActionCandidate, CompiledActionSet, CompiledCog};
use crate::contract::{
    ContractId, Currency, ExerciseSchedule, OptionRight, PriceMicros, Settlement, TimestampNs,
    UnderlyingId,
};
use crate::hash::canonical_hash;
use crate::runtime::{ContractLifecycle, PortfolioState};
use crate::search::{CandidateAction, CompiledCandidateAction};
use crate::Position;
use crate::{CogError, CogResult};

/// Schema tag included in every [`PortfolioGraphSample`].
pub const GRAPH_SAMPLE_SCHEMA_VERSION: &str = "volt.portfolio_graph.v1";
/// Schema tag included in every [`CompiledPortfolioGraphSample`].
pub const COMPILED_PORTFOLIO_GRAPH_SCHEMA_VERSION: &str = "volt.compiled_portfolio_graph.v1";
/// Schema tag included in every [`StateActionGraphSample`].
pub const STATE_ACTION_GRAPH_SCHEMA_VERSION: &str = "volt.state_action_graph.v1";
/// Schema tag included in every [`RuntimeStateActionGraphSample`].
pub const RUNTIME_STATE_ACTION_GRAPH_SCHEMA_VERSION: &str = "volt.runtime_state_action_graph.v1";

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

/// Portfolio graph bound to the source and artifact that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledPortfolioGraphSample {
    /// Must equal [`COMPILED_PORTFOLIO_GRAPH_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Content hash of compiler provenance plus graph feature identity.
    pub sample_id: String,
    /// Immutable compiled COG artifact hash.
    pub compiled_cog_hash: String,
    /// Canonical structured-source hash.
    pub source_hash: String,
    /// Existing exact portfolio graph and optional target.
    pub portfolio_graph: PortfolioGraphSample,
}

impl CompiledPortfolioGraphSample {
    /// Bind an exact graph sample to its validated compiled COG artifact.
    pub fn from_compiled_cog(
        compiled_cog: &CompiledCog,
        portfolio_graph: PortfolioGraphSample,
    ) -> CogResult<Self> {
        compiled_cog.validate()?;
        if portfolio_graph.schema_version != GRAPH_SAMPLE_SCHEMA_VERSION {
            return Err(CogError::InvalidDefinition(format!(
                "unsupported nested portfolio graph schema {}",
                portfolio_graph.schema_version
            )));
        }
        #[derive(Serialize)]
        struct Identity<'a> {
            compiled_cog_hash: &'a str,
            source_hash: &'a str,
            portfolio_graph_sample_id: &'a str,
        }
        let sample_id = canonical_hash(
            COMPILED_PORTFOLIO_GRAPH_SCHEMA_VERSION,
            &Identity {
                compiled_cog_hash: &compiled_cog.artifact_hash,
                source_hash: &compiled_cog.source_hash,
                portfolio_graph_sample_id: &portfolio_graph.sample_id,
            },
        )?;
        Ok(Self {
            schema_version: COMPILED_PORTFOLIO_GRAPH_SCHEMA_VERSION.to_owned(),
            sample_id,
            compiled_cog_hash: compiled_cog.artifact_hash.clone(),
            source_hash: compiled_cog.source_hash.clone(),
            portfolio_graph,
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

/// Runtime-only values attached to a state/action graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateGraphContext {
    /// Number of accepted lifecycle transitions.
    pub sequence: u64,
    /// Latest observation time.
    pub as_of: Option<TimestampNs>,
    /// Accumulated cash ledgers.
    pub cash_micros: BTreeMap<Currency, i128>,
    /// Accumulated delivered underlying units.
    pub underlying_units: BTreeMap<UnderlyingId, i64>,
}

/// Coarse action category exposed to graph-learning consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphActionKind {
    /// Add one exact option branch.
    AddLeg,
    /// Remove one exact open leg.
    RemoveLeg,
    /// Resize one exact open leg.
    ResizeLeg,
    /// Preserve state while advancing observation time.
    Wait,
    /// Exercise an exact holder right.
    Exercise,
}

/// One compiler-owned action node, including rejected action slots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateActionGraphNode {
    /// Identifier unique within the state/action graph.
    pub node_id: String,
    /// Source rule that proposed this slot.
    pub rule_id: String,
    /// Rule priority.
    pub priority: i32,
    /// Branch index after exact listed-universe resolution.
    pub branch_index: u32,
    /// Content-addressed action identifier.
    pub action_hash: String,
    /// Coarse action type.
    pub action_kind: GraphActionKind,
    /// Exact contract when resolution reached one.
    pub contract_id: Option<ContractId>,
    /// Option right when known.
    pub option_right: Option<OptionRight>,
    /// Exact strike when known.
    pub strike_micros: Option<i64>,
    /// Signed add/resize quantity or positive exercise quantity.
    pub quantity: i64,
    /// Hard legality bit supplied by the compiler.
    pub legal: bool,
    /// Stable explanation for a rejected slot.
    pub rejection_reason: Option<String>,
}

/// Relationship from an action node to state graph semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateActionGraphEdgeType {
    /// Action is a candidate transition from the portfolio root.
    TransitionFrom,
    /// Action targets one currently open exact position.
    TargetsPosition,
    /// Add-leg action is written on the graph underlying.
    UsesUnderlying,
}

/// One directed edge whose source is an action node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateActionGraphEdge {
    /// Action-node identifier.
    pub source: String,
    /// Portfolio-graph node identifier.
    pub target: String,
    /// Meaning of the action-to-state relationship.
    pub edge_type: StateActionGraphEdgeType,
}

/// Heterogeneous state/action graph with a compiler-owned legal action mask.
///
/// The nested `portfolio_graph` preserves existing exact portfolio and payoff
/// identities. Action nodes are an additional view and never replace compiler
/// or runtime authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateActionGraphSample {
    /// Must equal [`STATE_ACTION_GRAPH_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Content hash of graph inputs.
    pub sample_id: String,
    /// Compiled COG artifact that produced the candidate set.
    pub compiled_cog_hash: String,
    /// Runtime state hash.
    pub state_hash: String,
    /// Point-in-time compiled action-set hash.
    pub action_set_hash: String,
    /// Exact static portfolio graph derived from live positions.
    pub portfolio_graph: PortfolioGraphSample,
    /// Runtime ledgers and observation sequence.
    pub state_context: StateGraphContext,
    /// Stable action slots.
    pub action_nodes: Vec<StateActionGraphNode>,
    /// Typed action-to-state relationships.
    pub action_edges: Vec<StateActionGraphEdge>,
    /// Hard compiler legality aligned with `action_nodes`.
    pub legal_action_mask: Vec<bool>,
}

impl StateActionGraphSample {
    /// Compile one runtime state and action set into a versioned graph sample.
    pub fn from_state_and_actions(
        state: &PortfolioState,
        graph_context: &GraphContext,
        action_set: &CompiledActionSet,
    ) -> CogResult<Self> {
        state.check_invariants()?;
        graph_context.validate()?;
        action_set.validate()?;
        let state_hash = state.state_hash()?;
        if action_set.state_hash != state_hash {
            return Err(CogError::InvalidDefinition(
                "state/action graph received an action set for another state".to_owned(),
            ));
        }
        if action_set.candidates.len() != action_set.legal_action_mask.len() {
            return Err(CogError::InvalidDefinition(
                "compiled action candidates and legal mask are not aligned".to_owned(),
            ));
        }
        if action_set
            .candidates
            .iter()
            .map(|candidate| candidate.legal)
            .ne(action_set.legal_action_mask.iter().copied())
        {
            return Err(CogError::InvalidDefinition(
                "compiled action legal bits disagree with the legal mask".to_owned(),
            ));
        }

        let portfolio = live_state_portfolio(state)?;
        let portfolio_graph =
            PortfolioGraphSample::from_portfolio(&portfolio, graph_context, None)?;
        let state_context = StateGraphContext {
            sequence: state.sequence,
            as_of: state.as_of,
            cash_micros: state.cash_micros.clone(),
            underlying_units: state.underlying_units.clone(),
        };
        let position_nodes = portfolio_graph
            .nodes
            .iter()
            .filter_map(|node| match &node.payload {
                GraphNodePayload::OptionPosition { contract_id, .. } => {
                    Some((contract_id.clone(), node.node_id.clone()))
                }
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let mut action_nodes = Vec::with_capacity(action_set.candidates.len());
        let mut action_edges = Vec::new();
        for (index, candidate) in action_set.candidates.iter().enumerate() {
            let node_id = format!("action:{index}");
            let descriptor = action_descriptor(candidate);
            action_nodes.push(StateActionGraphNode {
                node_id: node_id.clone(),
                rule_id: candidate.rule_id.clone(),
                priority: candidate.priority,
                branch_index: candidate.branch_index,
                action_hash: candidate.action_hash.clone(),
                action_kind: descriptor.kind,
                contract_id: descriptor.contract_id.clone(),
                option_right: descriptor.option_right,
                strike_micros: descriptor.strike_micros,
                quantity: descriptor.quantity,
                legal: candidate.legal,
                rejection_reason: candidate.rejection_reason.clone(),
            });
            action_edges.push(StateActionGraphEdge {
                source: node_id.clone(),
                target: "portfolio".to_owned(),
                edge_type: StateActionGraphEdgeType::TransitionFrom,
            });
            if let Some(contract_id) = &descriptor.contract_id {
                if let Some(position_node) = position_nodes.get(contract_id) {
                    action_edges.push(StateActionGraphEdge {
                        source: node_id.clone(),
                        target: position_node.clone(),
                        edge_type: StateActionGraphEdgeType::TargetsPosition,
                    });
                }
            }
            if descriptor.kind == GraphActionKind::AddLeg {
                action_edges.push(StateActionGraphEdge {
                    source: node_id,
                    target: "underlying".to_owned(),
                    edge_type: StateActionGraphEdgeType::UsesUnderlying,
                });
            }
        }
        let legal_action_mask = action_set.legal_action_mask.clone();

        #[derive(Serialize)]
        struct Identity<'a> {
            compiled_cog_hash: &'a str,
            state_hash: &'a str,
            action_set_hash: &'a str,
            portfolio_graph: &'a PortfolioGraphSample,
            state_context: &'a StateGraphContext,
            action_nodes: &'a [StateActionGraphNode],
            action_edges: &'a [StateActionGraphEdge],
            legal_action_mask: &'a [bool],
        }
        let sample_id = canonical_hash(
            STATE_ACTION_GRAPH_SCHEMA_VERSION,
            &Identity {
                compiled_cog_hash: &action_set.compiled_cog_hash,
                state_hash: &state_hash,
                action_set_hash: &action_set.action_set_hash,
                portfolio_graph: &portfolio_graph,
                state_context: &state_context,
                action_nodes: &action_nodes,
                action_edges: &action_edges,
                legal_action_mask: &legal_action_mask,
            },
        )?;
        Ok(Self {
            schema_version: STATE_ACTION_GRAPH_SCHEMA_VERSION.to_owned(),
            sample_id,
            compiled_cog_hash: action_set.compiled_cog_hash.clone(),
            state_hash,
            action_set_hash: action_set.action_set_hash.clone(),
            portfolio_graph,
            state_context,
            action_nodes,
            action_edges,
            legal_action_mask,
        })
    }
}

#[derive(Debug)]
struct ActionDescriptor {
    kind: GraphActionKind,
    contract_id: Option<ContractId>,
    option_right: Option<OptionRight>,
    strike_micros: Option<i64>,
    quantity: i64,
}

fn action_descriptor(candidate: &CompiledActionCandidate) -> ActionDescriptor {
    match &candidate.compiled_action {
        Some(CompiledCandidateAction::AddLeg { contract, quantity }) => ActionDescriptor {
            kind: GraphActionKind::AddLeg,
            contract_id: Some(contract.contract_id.clone()),
            option_right: Some(contract.right),
            strike_micros: Some(contract.strike.0),
            quantity: *quantity,
        },
        Some(CompiledCandidateAction::RemoveLeg { contract }) => ActionDescriptor {
            kind: GraphActionKind::RemoveLeg,
            contract_id: Some(contract.clone()),
            option_right: None,
            strike_micros: None,
            quantity: 0,
        },
        Some(CompiledCandidateAction::ResizeLeg {
            contract,
            delta_quantity,
        }) => ActionDescriptor {
            kind: GraphActionKind::ResizeLeg,
            contract_id: Some(contract.clone()),
            option_right: None,
            strike_micros: None,
            quantity: *delta_quantity,
        },
        Some(CompiledCandidateAction::Wait) => ActionDescriptor {
            kind: GraphActionKind::Wait,
            contract_id: None,
            option_right: None,
            strike_micros: None,
            quantity: 0,
        },
        Some(CompiledCandidateAction::Exercise { contract, quantity }) => ActionDescriptor {
            kind: GraphActionKind::Exercise,
            contract_id: Some(contract.clone()),
            option_right: None,
            strike_micros: None,
            quantity: *quantity,
        },
        None => source_action_descriptor(&candidate.source_action),
    }
}

fn source_action_descriptor(action: &CandidateAction) -> ActionDescriptor {
    match action {
        CandidateAction::AddLeg {
            right, quantity, ..
        } => ActionDescriptor {
            kind: GraphActionKind::AddLeg,
            contract_id: None,
            option_right: Some(*right),
            strike_micros: None,
            quantity: *quantity,
        },
        CandidateAction::RemoveLeg { contract } => ActionDescriptor {
            kind: GraphActionKind::RemoveLeg,
            contract_id: Some(contract.clone()),
            option_right: None,
            strike_micros: None,
            quantity: 0,
        },
        CandidateAction::ResizeLeg {
            contract,
            delta_quantity,
        } => ActionDescriptor {
            kind: GraphActionKind::ResizeLeg,
            contract_id: Some(contract.clone()),
            option_right: None,
            strike_micros: None,
            quantity: *delta_quantity,
        },
        CandidateAction::Wait => ActionDescriptor {
            kind: GraphActionKind::Wait,
            contract_id: None,
            option_right: None,
            strike_micros: None,
            quantity: 0,
        },
        CandidateAction::Exercise { contract, quantity } => ActionDescriptor {
            kind: GraphActionKind::Exercise,
            contract_id: Some(contract.clone()),
            option_right: None,
            strike_micros: None,
            quantity: *quantity,
        },
    }
}

fn live_state_portfolio(state: &PortfolioState) -> CogResult<CanonicalPortfolio> {
    CanonicalPortfolio::from_positions(
        state
            .positions
            .values()
            .filter(|position| position.lifecycle == ContractLifecycle::Live)
            .map(|position| Position::new(position.contract.clone(), position.open_quantity))
            .collect::<CogResult<Vec<_>>>()?,
    )
}

/// Hard account limits used to refine compiler legality for one runtime graph.
///
/// The limits are deliberately small and observable. A production margin
/// engine can supply tighter masks without changing the graph schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphAccountLimits {
    /// Maximum number of live option positions.
    pub maximum_live_positions: u32,
    /// Maximum sum of absolute live option quantities.
    pub maximum_gross_option_quantity: u64,
    /// Maximum absolute delivered units for each underlying.
    pub maximum_absolute_underlying_units: u64,
    /// Minimum permitted cash balance by currency.
    pub minimum_cash_micros: BTreeMap<Currency, i128>,
    /// Deterministic cost charged per traded option contract.
    pub transaction_cost_per_contract_micros: i128,
}

impl GraphAccountLimits {
    /// Validate positive position limits, currencies, and transaction costs.
    pub fn validate(&self) -> CogResult<()> {
        if self.maximum_live_positions == 0
            || self.maximum_gross_option_quantity == 0
            || self.maximum_absolute_underlying_units == 0
        {
            return Err(CogError::InvalidDefinition(
                "graph account limits must be positive".to_owned(),
            ));
        }
        if self.transaction_cost_per_contract_micros < 0 {
            return Err(CogError::InvalidDefinition(
                "graph account transaction cost must be non-negative".to_owned(),
            ));
        }
        if self
            .minimum_cash_micros
            .keys()
            .any(|currency| currency.0.trim().is_empty())
        {
            return Err(CogError::InvalidDefinition(
                "graph account cash currencies must not be empty".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Point-in-time marks and account limits for a runtime state/action graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeGraphContext {
    /// Decision clock as Unix epoch nanoseconds.
    pub valuation_time: TimestampNs,
    /// Reference spot for every underlying represented by the state or actions.
    pub reference_spots: BTreeMap<UnderlyingId, PriceMicros>,
    /// Per-unit option marks for every represented exact contract.
    pub marks: BTreeMap<ContractId, PriceMicros>,
    /// Account constraints used to refine the compiler action mask.
    pub account_limits: GraphAccountLimits,
}

impl RuntimeGraphContext {
    /// Validate the clock, marks, spots, and account limits.
    pub fn validate(&self) -> CogResult<()> {
        if self.valuation_time.0 < 0 {
            return Err(CogError::InvalidDefinition(
                "runtime graph valuation time must be non-negative".to_owned(),
            ));
        }
        if self.reference_spots.is_empty()
            || self
                .reference_spots
                .iter()
                .any(|(underlying, spot)| underlying.0.trim().is_empty() || spot.0 <= 0)
        {
            return Err(CogError::InvalidDefinition(
                "runtime graph reference spots must be identified and positive".to_owned(),
            ));
        }
        if self
            .marks
            .iter()
            .any(|(contract, mark)| contract.0.trim().is_empty() || mark.0 < 0)
        {
            return Err(CogError::InvalidDefinition(
                "runtime graph marks must be identified and non-negative".to_owned(),
            ));
        }
        self.account_limits.validate()
    }
}

/// Typed data carried by a multi-expiry runtime graph node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "node_type", rename_all = "snake_case")]
pub enum RuntimeGraphNodePayload {
    /// Root for the exact option positions.
    Portfolio {
        /// Number of live option positions.
        live_leg_count: u32,
        /// Sum of absolute live option quantities.
        gross_option_quantity: u64,
        /// Runtime journal sequence.
        sequence: u64,
    },
    /// Account constraints and point-in-time action counts.
    Account {
        /// Number of actions admitted by the compiler.
        compiler_legal_action_count: u32,
        /// Number remaining after account constraints.
        legal_action_count: u32,
        /// Maximum number of live option positions.
        maximum_live_positions: u32,
        /// Maximum gross option quantity.
        maximum_gross_option_quantity: u64,
        /// Maximum absolute underlying inventory.
        maximum_absolute_underlying_units: u64,
    },
    /// One currency ledger.
    Cash {
        /// Currency of this ledger.
        currency: Currency,
        /// Current signed cash balance.
        balance_micros: i128,
        /// Minimum permitted cash balance.
        minimum_balance_micros: i128,
    },
    /// One underlying market and inventory ledger.
    Underlying {
        /// Exact underlying identity.
        underlying: UnderlyingId,
        /// Point-in-time reference spot.
        reference_spot_micros: i64,
        /// Current signed physical inventory.
        units: i64,
        /// Maximum permitted absolute inventory.
        maximum_absolute_units: u64,
    },
    /// One exact expiry shared by any number of position or action nodes.
    Expiry {
        /// Expiry time.
        expiry_ns: i64,
        /// Signed duration from valuation; negative values identify elapsed expiries.
        time_to_expiry_ns: i64,
    },
    /// One exact option position, including completed lifecycle states.
    OptionPosition {
        /// Exact listed contract identity.
        contract_id: ContractId,
        /// Current lifecycle.
        lifecycle: ContractLifecycle,
        /// Signed live quantity, or zero after consumption.
        quantity: i64,
        /// Call or put.
        right: OptionRight,
        /// Settlement convention.
        settlement: Settlement,
        /// Exercise schedule category.
        exercise_style: GraphExerciseStyle,
        /// Strike per underlying unit.
        strike_micros: i64,
        /// Point-in-time option mark.
        mark_micros: i64,
        /// Underlying units represented by one contract.
        multiplier: u32,
        /// First permitted exercise time.
        first_exercise_ns: i64,
        /// Number of discrete opportunities, or zero for an American interval.
        exercise_opportunity_count: u32,
    },
    /// One stable compiler action slot.
    Action {
        /// Source rule identifier.
        rule_id: String,
        /// Rule priority.
        priority: i32,
        /// Branch index after listed-contract resolution.
        branch_index: u32,
        /// Stable action identifier.
        action_hash: String,
        /// Coarse action type.
        action_kind: GraphActionKind,
        /// Exact contract when resolution reached one.
        contract_id: Option<ContractId>,
        /// Call or put when an exact contract is known.
        option_right: Option<OptionRight>,
        /// Settlement convention when an exact contract is known.
        settlement: Option<Settlement>,
        /// Strike when an exact contract is known.
        strike_micros: Option<i64>,
        /// Point-in-time mark when an exact contract is known.
        mark_micros: Option<i64>,
        /// Expiry when an exact contract is known.
        expiry_ns: Option<i64>,
        /// Contract multiplier when an exact contract is known.
        multiplier: Option<u32>,
        /// Signed add/resize quantity or positive exercise quantity.
        quantity: i64,
        /// Compiler guard and grammar decision.
        compiler_legal: bool,
        /// Account-limit decision, conditional on compiler legality.
        account_legal: bool,
        /// Combined legal-action bit.
        legal: bool,
        /// Stable explanation for a rejected action.
        rejection_reason: Option<String>,
    },
}

/// One identified node in a multi-expiry runtime state/action graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeGraphNode {
    /// Identifier unique within the sample.
    pub node_id: String,
    /// Typed node attributes.
    #[serde(flatten)]
    pub payload: RuntimeGraphNodePayload,
}

/// Semantic relationship in a multi-expiry runtime state/action graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeGraphEdgeType {
    /// Option position belongs to the portfolio.
    PositionIn,
    /// Option position or action references an underlying.
    WrittenOn,
    /// Option position or action references an expiry.
    ExpiresAt,
    /// Portfolio belongs to the account.
    PortfolioInAccount,
    /// Cash ledger belongs to the account.
    CashInAccount,
    /// Underlying inventory belongs to the account.
    UnderlyingInAccount,
    /// Action is a possible transition from the portfolio.
    TransitionFrom,
    /// Action targets an exact current position.
    TargetsPosition,
    /// Action is subject to the account limits.
    ConstrainedBy,
}

/// One directed typed edge in a runtime state/action graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeGraphEdge {
    /// Source node identifier.
    pub source: String,
    /// Target node identifier.
    pub target: String,
    /// Semantic edge type.
    pub edge_type: RuntimeGraphEdgeType,
}

/// Multi-expiry runtime graph with cash, inventory, account, and action nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStateActionGraphSample {
    /// Must equal [`RUNTIME_STATE_ACTION_GRAPH_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Content hash of the complete graph input.
    pub sample_id: String,
    /// Compiled COG artifact that produced the candidate set.
    pub compiled_cog_hash: String,
    /// Exact runtime-state hash.
    pub state_hash: String,
    /// Compiler action-set hash before account refinement.
    pub action_set_hash: String,
    /// Heterogeneous, deterministically ordered nodes.
    pub nodes: Vec<RuntimeGraphNode>,
    /// Directed typed edges.
    pub edges: Vec<RuntimeGraphEdge>,
    /// Combined compiler and account legality aligned with action nodes.
    pub legal_action_mask: Vec<bool>,
}

impl RuntimeStateActionGraphSample {
    /// Build the graph and refine compiler legality with explicit account limits.
    pub fn from_state_and_actions(
        state: &PortfolioState,
        context: &RuntimeGraphContext,
        action_set: &CompiledActionSet,
    ) -> CogResult<Self> {
        state.check_invariants()?;
        context.validate()?;
        action_set.validate()?;
        let state_hash = state.state_hash()?;
        if action_set.state_hash != state_hash {
            return Err(CogError::InvalidDefinition(
                "runtime graph received an action set for another state".to_owned(),
            ));
        }

        let mut action_legality = Vec::with_capacity(action_set.candidates.len());
        for candidate in &action_set.candidates {
            if !candidate.legal {
                action_legality.push((false, false, candidate.rejection_reason.clone()));
                continue;
            }
            let (account_legal, reason) = account_action_legality(state, context, candidate)?;
            action_legality.push((true, account_legal, reason));
        }
        let legal_action_mask = action_legality
            .iter()
            .map(|(compiler_legal, account_legal, _)| *compiler_legal && *account_legal)
            .collect::<Vec<_>>();

        let live_positions = state
            .positions
            .values()
            .filter(|position| position.lifecycle == ContractLifecycle::Live)
            .collect::<Vec<_>>();
        let live_leg_count = u32::try_from(live_positions.len())
            .map_err(|_| CogError::NumericOverflow("runtime graph live positions".to_owned()))?;
        let gross_option_quantity = live_positions.iter().try_fold(0_u64, |total, position| {
            total
                .checked_add(position.open_quantity.unsigned_abs())
                .ok_or_else(|| CogError::NumericOverflow("runtime graph gross quantity".to_owned()))
        })?;
        let compiler_legal_action_count = u32::try_from(
            action_legality
                .iter()
                .filter(|(compiler_legal, _, _)| *compiler_legal)
                .count(),
        )
        .map_err(|_| CogError::NumericOverflow("compiler legal action count".to_owned()))?;
        let legal_action_count =
            u32::try_from(legal_action_mask.iter().filter(|legal| **legal).count())
                .map_err(|_| CogError::NumericOverflow("legal action count".to_owned()))?;

        let mut nodes = vec![
            RuntimeGraphNode {
                node_id: "portfolio".to_owned(),
                payload: RuntimeGraphNodePayload::Portfolio {
                    live_leg_count,
                    gross_option_quantity,
                    sequence: state.sequence,
                },
            },
            RuntimeGraphNode {
                node_id: "account".to_owned(),
                payload: RuntimeGraphNodePayload::Account {
                    compiler_legal_action_count,
                    legal_action_count,
                    maximum_live_positions: context.account_limits.maximum_live_positions,
                    maximum_gross_option_quantity: context
                        .account_limits
                        .maximum_gross_option_quantity,
                    maximum_absolute_underlying_units: context
                        .account_limits
                        .maximum_absolute_underlying_units,
                },
            },
        ];
        let mut edges = vec![RuntimeGraphEdge {
            source: "portfolio".to_owned(),
            target: "account".to_owned(),
            edge_type: RuntimeGraphEdgeType::PortfolioInAccount,
        }];

        let mut currencies = state.cash_micros.keys().cloned().collect::<BTreeSet<_>>();
        currencies.extend(context.account_limits.minimum_cash_micros.keys().cloned());
        currencies.extend(
            state
                .positions
                .values()
                .map(|position| position.contract.currency.clone()),
        );
        let mut cash_nodes = BTreeMap::new();
        for (index, currency) in currencies.into_iter().enumerate() {
            let node_id = format!("cash:{index}");
            nodes.push(RuntimeGraphNode {
                node_id: node_id.clone(),
                payload: RuntimeGraphNodePayload::Cash {
                    balance_micros: state.cash_micros.get(&currency).copied().unwrap_or(0),
                    minimum_balance_micros: context
                        .account_limits
                        .minimum_cash_micros
                        .get(&currency)
                        .copied()
                        .unwrap_or(i128::MIN),
                    currency: currency.clone(),
                },
            });
            edges.push(RuntimeGraphEdge {
                source: node_id.clone(),
                target: "account".to_owned(),
                edge_type: RuntimeGraphEdgeType::CashInAccount,
            });
            cash_nodes.insert(currency, node_id);
        }

        let mut underlyings = state
            .underlying_units
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        underlyings.extend(
            state
                .positions
                .values()
                .map(|position| position.contract.underlying.clone()),
        );
        for candidate in &action_set.candidates {
            if let Some(contract) = resolved_action_contract(candidate, state) {
                underlyings.insert(contract.underlying.clone());
            }
        }
        let mut underlying_nodes = BTreeMap::new();
        for (index, underlying) in underlyings.into_iter().enumerate() {
            let spot = context.reference_spots.get(&underlying).ok_or_else(|| {
                CogError::InvalidDefinition(format!(
                    "runtime graph has no reference spot for {}",
                    underlying.0
                ))
            })?;
            let node_id = format!("underlying:{index}");
            nodes.push(RuntimeGraphNode {
                node_id: node_id.clone(),
                payload: RuntimeGraphNodePayload::Underlying {
                    underlying: underlying.clone(),
                    reference_spot_micros: spot.0,
                    units: state
                        .underlying_units
                        .get(&underlying)
                        .copied()
                        .unwrap_or(0),
                    maximum_absolute_units: context
                        .account_limits
                        .maximum_absolute_underlying_units,
                },
            });
            edges.push(RuntimeGraphEdge {
                source: node_id.clone(),
                target: "account".to_owned(),
                edge_type: RuntimeGraphEdgeType::UnderlyingInAccount,
            });
            underlying_nodes.insert(underlying, node_id);
        }

        let mut expiries = state
            .positions
            .values()
            .map(|position| position.contract.expiry())
            .collect::<BTreeSet<_>>();
        for candidate in &action_set.candidates {
            if let Some(contract) = resolved_action_contract(candidate, state) {
                expiries.insert(contract.expiry());
            }
        }
        let mut expiry_nodes = BTreeMap::new();
        for (index, expiry) in expiries.into_iter().enumerate() {
            let node_id = format!("expiry:{index}");
            nodes.push(RuntimeGraphNode {
                node_id: node_id.clone(),
                payload: RuntimeGraphNodePayload::Expiry {
                    expiry_ns: expiry.0,
                    time_to_expiry_ns: expiry.0.checked_sub(context.valuation_time.0).ok_or_else(
                        || CogError::NumericOverflow("runtime graph time to expiry".to_owned()),
                    )?,
                },
            });
            expiry_nodes.insert(expiry, node_id);
        }

        let mut position_nodes = BTreeMap::new();
        for (index, (contract_id, position)) in state.positions.iter().enumerate() {
            let contract = &position.contract;
            let mark = context.marks.get(contract_id).ok_or_else(|| {
                CogError::InvalidDefinition(format!(
                    "runtime graph has no mark for {}",
                    contract_id.0
                ))
            })?;
            let (exercise_style, first_exercise_ns, exercise_opportunity_count) =
                exercise_features(&contract.exercise)?;
            let node_id = format!("position:{index}");
            nodes.push(RuntimeGraphNode {
                node_id: node_id.clone(),
                payload: RuntimeGraphNodePayload::OptionPosition {
                    contract_id: contract_id.clone(),
                    lifecycle: position.lifecycle,
                    quantity: position.open_quantity,
                    right: contract.right,
                    settlement: contract.settlement,
                    exercise_style,
                    strike_micros: contract.strike.0,
                    mark_micros: mark.0,
                    multiplier: contract.multiplier,
                    first_exercise_ns,
                    exercise_opportunity_count,
                },
            });
            edges.extend([
                RuntimeGraphEdge {
                    source: node_id.clone(),
                    target: "portfolio".to_owned(),
                    edge_type: RuntimeGraphEdgeType::PositionIn,
                },
                RuntimeGraphEdge {
                    source: node_id.clone(),
                    target: underlying_nodes[&contract.underlying].clone(),
                    edge_type: RuntimeGraphEdgeType::WrittenOn,
                },
                RuntimeGraphEdge {
                    source: node_id.clone(),
                    target: expiry_nodes[&contract.expiry()].clone(),
                    edge_type: RuntimeGraphEdgeType::ExpiresAt,
                },
            ]);
            position_nodes.insert(contract_id.clone(), node_id);
        }

        for (index, (candidate, (compiler_legal, account_legal, account_reason))) in action_set
            .candidates
            .iter()
            .zip(action_legality.iter())
            .enumerate()
        {
            let descriptor = action_descriptor(candidate);
            let action_contract = resolved_action_contract(candidate, state);
            let action_mark = action_contract
                .map(|contract| {
                    context.marks.get(&contract.contract_id).ok_or_else(|| {
                        CogError::InvalidDefinition(format!(
                            "runtime graph has no mark for action contract {}",
                            contract.contract_id.0
                        ))
                    })
                })
                .transpose()?;
            let legal = *compiler_legal && *account_legal;
            let rejection_reason = if !*compiler_legal {
                candidate.rejection_reason.clone()
            } else if !*account_legal {
                account_reason.clone()
            } else {
                None
            };
            let node_id = format!("action:{index}");
            nodes.push(RuntimeGraphNode {
                node_id: node_id.clone(),
                payload: RuntimeGraphNodePayload::Action {
                    rule_id: candidate.rule_id.clone(),
                    priority: candidate.priority,
                    branch_index: candidate.branch_index,
                    action_hash: candidate.action_hash.clone(),
                    action_kind: descriptor.kind,
                    contract_id: descriptor.contract_id.clone(),
                    option_right: action_contract.map(|contract| contract.right),
                    settlement: action_contract.map(|contract| contract.settlement),
                    strike_micros: action_contract.map(|contract| contract.strike.0),
                    mark_micros: action_mark.map(|mark| mark.0),
                    expiry_ns: action_contract.map(|contract| contract.expiry().0),
                    multiplier: action_contract.map(|contract| contract.multiplier),
                    quantity: descriptor.quantity,
                    compiler_legal: *compiler_legal,
                    account_legal: *account_legal,
                    legal,
                    rejection_reason,
                },
            });
            edges.extend([
                RuntimeGraphEdge {
                    source: node_id.clone(),
                    target: "portfolio".to_owned(),
                    edge_type: RuntimeGraphEdgeType::TransitionFrom,
                },
                RuntimeGraphEdge {
                    source: node_id.clone(),
                    target: "account".to_owned(),
                    edge_type: RuntimeGraphEdgeType::ConstrainedBy,
                },
            ]);
            if let Some(contract_id) = &descriptor.contract_id {
                if let Some(position_node) = position_nodes.get(contract_id) {
                    edges.push(RuntimeGraphEdge {
                        source: node_id.clone(),
                        target: position_node.clone(),
                        edge_type: RuntimeGraphEdgeType::TargetsPosition,
                    });
                }
            }
            if let Some(contract) = action_contract {
                edges.extend([
                    RuntimeGraphEdge {
                        source: node_id.clone(),
                        target: underlying_nodes[&contract.underlying].clone(),
                        edge_type: RuntimeGraphEdgeType::WrittenOn,
                    },
                    RuntimeGraphEdge {
                        source: node_id,
                        target: expiry_nodes[&contract.expiry()].clone(),
                        edge_type: RuntimeGraphEdgeType::ExpiresAt,
                    },
                ]);
            }
        }

        #[derive(Serialize)]
        struct Identity<'a> {
            compiled_cog_hash: &'a str,
            state_hash: &'a str,
            action_set_hash: &'a str,
            nodes: &'a [RuntimeGraphNode],
            edges: &'a [RuntimeGraphEdge],
            legal_action_mask: &'a [bool],
        }
        let sample_id = canonical_hash(
            RUNTIME_STATE_ACTION_GRAPH_SCHEMA_VERSION,
            &Identity {
                compiled_cog_hash: &action_set.compiled_cog_hash,
                state_hash: &state_hash,
                action_set_hash: &action_set.action_set_hash,
                nodes: &nodes,
                edges: &edges,
                legal_action_mask: &legal_action_mask,
            },
        )?;
        Ok(Self {
            schema_version: RUNTIME_STATE_ACTION_GRAPH_SCHEMA_VERSION.to_owned(),
            sample_id,
            compiled_cog_hash: action_set.compiled_cog_hash.clone(),
            state_hash,
            action_set_hash: action_set.action_set_hash.clone(),
            nodes,
            edges,
            legal_action_mask,
        })
    }
}

fn resolved_action_contract<'a>(
    candidate: &'a CompiledActionCandidate,
    state: &'a PortfolioState,
) -> Option<&'a crate::OptionContract> {
    match candidate.compiled_action.as_ref()? {
        CompiledCandidateAction::AddLeg { contract, .. } => Some(contract),
        CompiledCandidateAction::RemoveLeg { contract }
        | CompiledCandidateAction::ResizeLeg { contract, .. }
        | CompiledCandidateAction::Exercise { contract, .. } => state
            .positions
            .get(contract)
            .map(|position| &position.contract),
        CompiledCandidateAction::Wait => None,
    }
}

fn account_action_legality(
    state: &PortfolioState,
    context: &RuntimeGraphContext,
    candidate: &CompiledActionCandidate,
) -> CogResult<(bool, Option<String>)> {
    let action = candidate.compiled_action.as_ref().ok_or_else(|| {
        CogError::InvariantViolation("legal compiler action has no exact action".to_owned())
    })?;
    let mut quantities = state
        .positions
        .iter()
        .filter(|(_, position)| position.lifecycle == ContractLifecycle::Live)
        .map(|(contract, position)| (contract.clone(), position.open_quantity))
        .collect::<BTreeMap<_, _>>();
    let mut cash = state.cash_micros.clone();
    let mut underlying = state.underlying_units.clone();

    match action {
        CompiledCandidateAction::AddLeg { contract, quantity } => {
            if state
                .positions
                .get(&contract.contract_id)
                .is_some_and(|position| position.lifecycle != ContractLifecycle::Live)
            {
                return Ok((
                    false,
                    Some("account runtime does not reopen a consumed contract".to_owned()),
                ));
            }
            apply_option_trade(&mut quantities, &mut cash, contract, *quantity, context)?;
        }
        CompiledCandidateAction::RemoveLeg { contract } => {
            let position = state.positions.get(contract).ok_or_else(|| {
                CogError::InvariantViolation(format!(
                    "remove action targets missing position {}",
                    contract.0
                ))
            })?;
            apply_option_trade(
                &mut quantities,
                &mut cash,
                &position.contract,
                position.open_quantity.checked_neg().ok_or_else(|| {
                    CogError::NumericOverflow("remove action quantity".to_owned())
                })?,
                context,
            )?;
        }
        CompiledCandidateAction::ResizeLeg {
            contract,
            delta_quantity,
        } => {
            let position = state.positions.get(contract).ok_or_else(|| {
                CogError::InvariantViolation(format!(
                    "resize action targets missing position {}",
                    contract.0
                ))
            })?;
            apply_option_trade(
                &mut quantities,
                &mut cash,
                &position.contract,
                *delta_quantity,
                context,
            )?;
        }
        CompiledCandidateAction::Wait => {}
        CompiledCandidateAction::Exercise { contract, quantity } => {
            let position = state.positions.get(contract).ok_or_else(|| {
                CogError::InvariantViolation(format!(
                    "exercise action targets missing position {}",
                    contract.0
                ))
            })?;
            apply_exercise_preview(
                &mut quantities,
                &mut cash,
                &mut underlying,
                &position.contract,
                *quantity,
                context,
            )?;
        }
    }

    let live_count = quantities
        .values()
        .filter(|quantity| **quantity != 0)
        .count();
    if live_count
        > usize::try_from(context.account_limits.maximum_live_positions)
            .map_err(|_| CogError::NumericOverflow("maximum live positions".to_owned()))?
    {
        return Ok((
            false,
            Some("account maximum live positions would be exceeded".to_owned()),
        ));
    }
    let gross = quantities.values().try_fold(0_u64, |total, quantity| {
        total
            .checked_add(quantity.unsigned_abs())
            .ok_or_else(|| CogError::NumericOverflow("projected gross quantity".to_owned()))
    })?;
    if gross > context.account_limits.maximum_gross_option_quantity {
        return Ok((
            false,
            Some("account maximum gross option quantity would be exceeded".to_owned()),
        ));
    }
    if underlying.values().any(|units| {
        units.unsigned_abs() > context.account_limits.maximum_absolute_underlying_units
    }) {
        return Ok((
            false,
            Some("account maximum absolute underlying inventory would be exceeded".to_owned()),
        ));
    }
    for (currency, minimum) in &context.account_limits.minimum_cash_micros {
        if cash.get(currency).copied().unwrap_or(0) < *minimum {
            return Ok((
                false,
                Some(format!(
                    "account minimum cash for {} would be breached",
                    currency.0
                )),
            ));
        }
    }
    Ok((true, None))
}

fn apply_option_trade(
    quantities: &mut BTreeMap<ContractId, i64>,
    cash: &mut BTreeMap<Currency, i128>,
    contract: &crate::OptionContract,
    delta_quantity: i64,
    context: &RuntimeGraphContext,
) -> CogResult<()> {
    let current = quantities.get(&contract.contract_id).copied().unwrap_or(0);
    let next = current
        .checked_add(delta_quantity)
        .ok_or_else(|| CogError::NumericOverflow("projected option quantity".to_owned()))?;
    if next == 0 {
        quantities.remove(&contract.contract_id);
    } else {
        quantities.insert(contract.contract_id.clone(), next);
    }
    let mark = context.marks.get(&contract.contract_id).ok_or_else(|| {
        CogError::InvalidDefinition(format!(
            "runtime graph has no mark for action contract {}",
            contract.contract_id.0
        ))
    })?;
    let premium = i128::from(delta_quantity)
        .checked_mul(i128::from(contract.multiplier))
        .and_then(|value| value.checked_mul(i128::from(mark.0)))
        .and_then(i128::checked_neg)
        .ok_or_else(|| CogError::NumericOverflow("projected option premium".to_owned()))?;
    let cost = i128::from(delta_quantity.unsigned_abs())
        .checked_mul(context.account_limits.transaction_cost_per_contract_micros)
        .ok_or_else(|| CogError::NumericOverflow("projected transaction cost".to_owned()))?;
    add_projected_cash(
        cash,
        &contract.currency,
        premium
            .checked_sub(cost)
            .ok_or_else(|| CogError::NumericOverflow("projected trade cash".to_owned()))?,
    )
}

fn apply_exercise_preview(
    quantities: &mut BTreeMap<ContractId, i64>,
    cash: &mut BTreeMap<Currency, i128>,
    underlying: &mut BTreeMap<UnderlyingId, i64>,
    contract: &crate::OptionContract,
    quantity: i64,
    context: &RuntimeGraphContext,
) -> CogResult<()> {
    let current = quantities
        .get(&contract.contract_id)
        .copied()
        .ok_or_else(|| {
            CogError::InvariantViolation("exercise preview has no live quantity".to_owned())
        })?;
    let next = current
        .checked_sub(quantity)
        .ok_or_else(|| CogError::NumericOverflow("projected exercise quantity".to_owned()))?;
    if next == 0 {
        quantities.remove(&contract.contract_id);
    } else {
        quantities.insert(contract.contract_id.clone(), next);
    }
    let units = quantity
        .checked_mul(i64::from(contract.multiplier))
        .ok_or_else(|| CogError::NumericOverflow("projected exercise units".to_owned()))?;
    match contract.settlement {
        Settlement::Cash => {
            let spot = context
                .reference_spots
                .get(&contract.underlying)
                .ok_or_else(|| {
                    CogError::InvalidDefinition(format!(
                        "runtime graph has no reference spot for {}",
                        contract.underlying.0
                    ))
                })?;
            let amount = i128::from(contract.intrinsic_per_unit(*spot))
                .checked_mul(i128::from(units))
                .ok_or_else(|| CogError::NumericOverflow("projected cash exercise".to_owned()))?;
            add_projected_cash(cash, &contract.currency, amount)?;
        }
        Settlement::Physical => {
            let strike_cash = i128::from(contract.strike.0)
                .checked_mul(i128::from(units))
                .ok_or_else(|| {
                    CogError::NumericOverflow("projected physical exercise cash".to_owned())
                })?;
            let (cash_delta, underlying_delta) = match contract.right {
                OptionRight::Call => (
                    strike_cash.checked_neg().ok_or_else(|| {
                        CogError::NumericOverflow("projected call exercise cash".to_owned())
                    })?,
                    units,
                ),
                OptionRight::Put => (
                    strike_cash,
                    units.checked_neg().ok_or_else(|| {
                        CogError::NumericOverflow("projected put exercise inventory".to_owned())
                    })?,
                ),
            };
            add_projected_cash(cash, &contract.currency, cash_delta)?;
            let current_units = underlying.get(&contract.underlying).copied().unwrap_or(0);
            let next_units = current_units.checked_add(underlying_delta).ok_or_else(|| {
                CogError::NumericOverflow("projected underlying inventory".to_owned())
            })?;
            if next_units == 0 {
                underlying.remove(&contract.underlying);
            } else {
                underlying.insert(contract.underlying.clone(), next_units);
            }
        }
    }
    Ok(())
}

fn add_projected_cash(
    cash: &mut BTreeMap<Currency, i128>,
    currency: &Currency,
    amount: i128,
) -> CogResult<()> {
    let current = cash.get(currency).copied().unwrap_or(0);
    let next = current
        .checked_add(amount)
        .ok_or_else(|| CogError::NumericOverflow("projected cash ledger".to_owned()))?;
    if next == 0 {
        cash.remove(currency);
    } else {
        cash.insert(currency.clone(), next);
    }
    Ok(())
}
