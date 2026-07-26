//! Deterministic graph views for machine-learning and optimization systems.
//!
//! This module never replaces the exact portfolio model. It derives a stable,
//! versioned graph whose identifiers retain links to both exact contract
//! semantics and terminal-payoff equivalence.

use std::collections::BTreeMap;

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
