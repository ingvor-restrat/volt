//! Typed, canonical, executable options logic for Volt.
//!
//! Volt keeps exact listed-contract identity, terminal payoff, lifecycle
//! state, path simulation, and graph export connected without treating those
//! views as equivalent. The usual path through the API is:
//!
//! 1. construct validated [`OptionContract`] and [`Position`] values;
//! 2. aggregate them into a [`CanonicalPortfolio`];
//! 3. derive a [`TerminalPayoffProjection`] or [`PortfolioGraphSample`]; or
//! 4. create a [`PortfolioState`] and apply actions through [`CogRuntime`].
//!
//! The terminal projection is useful for search-space reduction, but it is not
//! a claim that American contracts or their execution realizations are
//! dynamically equivalent.
//!
//! Money uses integer micro-units and time uses Unix-epoch nanoseconds.
//! Serialized public structures carry schema versions, while semantic hashes
//! use a versioned domain separator.

#![warn(missing_docs)]

mod canonical;
mod compiler;
mod contract;
mod hash;
mod market;
mod ml;
mod runtime;
mod scenario;
mod search;

pub use canonical::{
    CanonicalPortfolio, CanonicalPosition, PayoffKnot, TerminalPayoffProjection,
    CANONICAL_PORTFOLIO_SCHEMA_VERSION, TERMINAL_PAYOFF_SCHEMA_VERSION,
};
pub use compiler::{
    CogCompiler, CogPositionSource, CogRuleSource, CogSource, CompilationReport,
    CompiledActionCandidate, CompiledActionSet, CompiledCog, CompiledCogRule, CompiledStepResult,
    CompilerDiagnostic, DiagnosticSeverity, GuardExpr, PositionSide, ScenarioObjectiveSource,
    COG_SOURCE_SCHEMA_VERSION, COMPILED_ACTION_SET_SCHEMA_VERSION, COMPILED_COG_SCHEMA_VERSION,
};
pub use contract::{
    ContractId, Currency, ExerciseSchedule, OptionContract, OptionRight, Position, PriceMicros,
    Settlement, TimestampNs, UnderlyingId, CONTRACT_SCHEMA_VERSION,
};
pub use market::{
    BarInterval, BookLevel, InMemoryOptionMarketDataSource, MarkBasis, ObservationOrigin,
    OptionBar, OptionDataSourceManifest, OptionMark, OptionMarketDataSource,
    OptionMarketObservation, OptionObservationPayload, OptionQuote, OptionTrade, OrderSide,
    PriceAuthority, MARKET_OBSERVATION_SCHEMA_VERSION, MARKET_SOURCE_SCHEMA_VERSION,
};
pub use ml::{
    CompiledPortfolioGraphSample, GraphActionKind, GraphContext, GraphEdge, GraphEdgeType,
    GraphExerciseStyle, GraphNode, GraphNodePayload, GraphTarget, PortfolioGraphSample,
    StateActionGraphEdge, StateActionGraphEdgeType, StateActionGraphNode, StateActionGraphSample,
    StateGraphContext, COMPILED_PORTFOLIO_GRAPH_SCHEMA_VERSION, GRAPH_SAMPLE_SCHEMA_VERSION,
    STATE_ACTION_GRAPH_SCHEMA_VERSION,
};
pub use runtime::{
    CogRuntime, ContractLifecycle, Emission, JournalEntry, MarketObservation, PortfolioState,
    PositionState, ReplayResult, TransitionAction, TransitionInput, TransitionResult,
    JOURNAL_SCHEMA_VERSION, RUNTIME_STATE_SCHEMA_VERSION,
};
pub use scenario::{
    DeterministicFillModel, LegPathAttribution, PathPnlSnapshot, ScenarioAction, ScenarioFill,
    ScenarioOptionPrice, ScenarioPath, ScenarioPoint, ScenarioSimulator, ScheduledScenarioAction,
    SimulationManifest, SimulationRequest, SimulationResult, TransactionCostModel,
    SCENARIO_PATH_SCHEMA_VERSION, SIMULATION_REQUEST_SCHEMA_VERSION,
    SIMULATION_RESULT_SCHEMA_VERSION,
};
pub use search::{
    CandidateAction, CandidateActionEffect, CandidateGrammar, CandidateMarketContext,
    CompiledCandidateAction, ExpiryBucket, StrikeOrDeltaBucket, CANDIDATE_ACTION_SCHEMA_VERSION,
    CANDIDATE_GRAMMAR_SCHEMA_VERSION,
};

use thiserror::Error;

/// Errors raised by definition validation, canonicalization, and execution.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CogError {
    /// A contract, path, grammar, or other declaration is malformed.
    #[error("invalid definition: {0}")]
    InvalidDefinition(String),
    /// A requested projection combines contracts with incompatible semantics.
    #[error("incompatible portfolio: {0}")]
    IncompatiblePortfolio(String),
    /// An otherwise valid action is not permitted in the current state.
    #[error("invalid action: {0}")]
    InvalidAction(String),
    /// A runtime state violates one of Volt's internal consistency rules.
    #[error("invariant violation: {0}")]
    InvariantViolation(String),
    /// Checked integer arithmetic could not represent a result.
    #[error("numeric overflow: {0}")]
    NumericOverflow(String),
    /// Canonical serialization failed.
    #[error("serialization failed: {0}")]
    Serialization(String),
    /// A market or scenario source could not provide required evidence.
    #[error("data source failed: {0}")]
    DataSource(String),
}

/// Result type used by the cog/options API.
pub type CogResult<T> = Result<T, CogError>;

/// Public project-name alias for [`CogError`].
pub type VoltError = CogError;

/// Public project-name alias for [`CogResult`].
pub type VoltResult<T> = CogResult<T>;

/// Public project-name alias for [`CogRuntime`].
pub type VoltRuntime = CogRuntime;
