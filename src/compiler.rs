//! Versioned structured source, compilation, and execution for financial logic.
//!
//! The first COG language is deliberately a serde-backed structured format.
//! It avoids committing to decorative syntax before the typed source model,
//! diagnostics, immutable artifact, and runtime boundary are stable.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::canonical::CanonicalPortfolio;
use crate::contract::{ContractId, OptionContract, Position};
use crate::hash::canonical_hash;
use crate::runtime::{
    CogRuntime, ContractLifecycle, MarketObservation, PortfolioState, TransitionAction,
    TransitionResult,
};
use crate::search::{
    CandidateAction, CandidateActionEffect, CandidateGrammar, CandidateMarketContext,
    CompiledCandidateAction,
};
use crate::{CogError, CogResult};

/// Schema accepted by [`CogCompiler`] for structured COG source.
pub const COG_SOURCE_SCHEMA_VERSION: &str = "volt.cog_source.v1";
/// Schema emitted by [`CogCompiler`] for immutable compiled artifacts.
pub const COMPILED_COG_SCHEMA_VERSION: &str = "volt.compiled_cog.v1";
/// Schema used by a point-in-time [`CompiledActionSet`].
pub const COMPILED_ACTION_SET_SCHEMA_VERSION: &str = "volt.compiled_action_set.v1";
const COMPILED_RULE_SCHEMA_VERSION: &str = "volt.compiled_rule.v1";
const COMPILED_ACTION_SCHEMA_VERSION: &str = "volt.compiled_action_candidate.v1";

/// One initial exact position declared in structured source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CogPositionSource {
    /// Exact listed contract identifier.
    pub contract_id: ContractId,
    /// Signed contract quantity.
    pub quantity: i64,
}

/// Position-side predicate used by [`GuardExpr::HasLivePosition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionSide {
    /// Match any non-zero live position.
    Any,
    /// Match a positive live position.
    Long,
    /// Match a negative live position.
    Short,
}

/// Typed, composable predicate over portfolio and point-in-time market state.
///
/// Guards contain only deterministic business logic. They do not contain a
/// stochastic model, value estimate, or learned legality decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "guard", rename_all = "snake_case")]
pub enum GuardExpr {
    /// Predicate that is always true.
    Always,
    /// Every nested predicate must be true.
    All {
        /// Non-empty ordered child predicates.
        guards: Vec<GuardExpr>,
    },
    /// At least one nested predicate must be true.
    Any {
        /// Non-empty ordered child predicates.
        guards: Vec<GuardExpr>,
    },
    /// Logical negation.
    Not {
        /// Predicate whose result is negated.
        expr: Box<GuardExpr>,
    },
    /// Point-in-time spot must be at least this value.
    SpotAtLeast {
        /// Inclusive lower spot bound in price micros.
        spot_micros: i64,
    },
    /// Point-in-time spot must be at most this value.
    SpotAtMost {
        /// Inclusive upper spot bound in price micros.
        spot_micros: i64,
    },
    /// Number of live exact positions must not exceed this bound.
    LiveLegCountAtMost {
        /// Inclusive maximum number of live positions.
        count: u32,
    },
    /// Number of live exact positions must meet this bound.
    LiveLegCountAtLeast {
        /// Inclusive minimum number of live positions.
        count: u32,
    },
    /// Sum of absolute live quantities must not exceed this bound.
    GrossQuantityAtMost {
        /// Inclusive maximum gross quantity.
        quantity: u64,
    },
    /// Sum of absolute live quantities must meet this bound.
    GrossQuantityAtLeast {
        /// Inclusive minimum gross quantity.
        quantity: u64,
    },
    /// An exact contract must have a live position on the requested side.
    HasLivePosition {
        /// Exact listed contract identifier.
        contract_id: ContractId,
        /// Required signed side.
        side: PositionSide,
    },
    /// An exact contract's holder exercise opportunity must be open.
    ExerciseOpen {
        /// Exact listed contract identifier.
        contract_id: ContractId,
    },
}

impl GuardExpr {
    fn evaluate(
        &self,
        state: &PortfolioState,
        context: &CandidateMarketContext,
    ) -> CogResult<bool> {
        match self {
            Self::Always => Ok(true),
            Self::All { guards } => {
                for guard in guards {
                    if !guard.evaluate(state, context)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Self::Any { guards } => {
                for guard in guards {
                    if guard.evaluate(state, context)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Self::Not { expr } => Ok(!expr.evaluate(state, context)?),
            Self::SpotAtLeast { spot_micros } => Ok(context.reference_spot.0 >= *spot_micros),
            Self::SpotAtMost { spot_micros } => Ok(context.reference_spot.0 <= *spot_micros),
            Self::LiveLegCountAtMost { count } => Ok(live_leg_count(state)? <= u64::from(*count)),
            Self::LiveLegCountAtLeast { count } => Ok(live_leg_count(state)? >= u64::from(*count)),
            Self::GrossQuantityAtMost { quantity } => Ok(live_gross_quantity(state)? <= *quantity),
            Self::GrossQuantityAtLeast { quantity } => Ok(live_gross_quantity(state)? >= *quantity),
            Self::HasLivePosition { contract_id, side } => {
                let Some(position) = state.positions.get(contract_id) else {
                    return Ok(false);
                };
                if position.lifecycle != ContractLifecycle::Live {
                    return Ok(false);
                }
                Ok(match side {
                    PositionSide::Any => position.open_quantity != 0,
                    PositionSide::Long => position.open_quantity > 0,
                    PositionSide::Short => position.open_quantity < 0,
                })
            }
            Self::ExerciseOpen { contract_id } => {
                let Some(position) = state.positions.get(contract_id) else {
                    return Ok(false);
                };
                Ok(position.lifecycle == ContractLifecycle::Live
                    && position.open_quantity > 0
                    && position
                        .contract
                        .exercise
                        .permits_exercise(context.valuation_time))
            }
        }
    }
}

/// One source rule pairing a typed guard with a typed candidate action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CogRuleSource {
    /// Stable identifier retained in diagnostics and compiled output.
    pub rule_id: String,
    /// Higher values are considered before lower values.
    pub priority: i32,
    /// Deterministic business predicate.
    pub guard: GuardExpr,
    /// Candidate action proposed when the guard is true.
    pub action: CandidateAction,
}

/// Solver-independent declaration of the supervised fixture objective.
///
/// This describes the scenario functional but does not place a pricing or
/// probability model inside the compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioObjectiveSource {
    /// Stable objective name exposed to learning systems.
    pub objective_name: String,
    /// Terminal underlying spots in price micros.
    pub scenario_spots_micros: Vec<i64>,
    /// Scenario weights in millionths; the vector must sum to one million.
    pub probability_millionths: Vec<u32>,
    /// Downside RMS weight in basis points.
    pub downside_weight_bps: u32,
    /// Penalty per gross contract in money micros.
    pub gross_contract_penalty_micros: i64,
}

/// Versioned structured input accepted by the initial COG compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CogSource {
    /// Must equal [`COG_SOURCE_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Stable human-readable program identifier.
    pub program_id: String,
    /// Optional human-readable purpose.
    #[serde(default)]
    pub description: String,
    /// Exact point-in-time listed universe available to candidate rules.
    pub listed_universe: Vec<OptionContract>,
    /// Typed action grammar referenced by program rules.
    pub candidate_grammar: CandidateGrammar,
    /// Exact starting positions.
    #[serde(default)]
    pub initial_positions: Vec<CogPositionSource>,
    /// Guarded business rules.
    #[serde(default)]
    pub rules: Vec<CogRuleSource>,
    /// Optional declared supervised objective.
    #[serde(default)]
    pub objective: Option<ScenarioObjectiveSource>,
}

/// Diagnostic importance emitted during compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    /// Compilation cannot produce an artifact.
    Error,
    /// Compilation can continue, but source intent may be incomplete.
    Warning,
}

/// One stable, source-addressable compiler diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilerDiagnostic {
    /// Error or warning.
    pub severity: DiagnosticSeverity,
    /// Stable machine-readable diagnostic code.
    pub code: String,
    /// JSON-style path into structured source.
    pub path: String,
    /// Human-readable explanation.
    pub message: String,
}

/// Result of compiling source, including every collected diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilationReport {
    /// Immutable artifact, absent when any error diagnostic was emitted.
    pub artifact: Option<CompiledCog>,
    /// Ordered compiler diagnostics.
    pub diagnostics: Vec<CompilerDiagnostic>,
}

impl CompilationReport {
    /// Returns true when at least one error diagnostic is present.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    }

    /// Convert a successful report into its artifact.
    pub fn into_result(self) -> CogResult<CompiledCog> {
        self.artifact.ok_or_else(|| {
            CogError::InvalidDefinition(
                self.diagnostics
                    .iter()
                    .map(|diagnostic| {
                        format!(
                            "{} at {}: {}",
                            diagnostic.code, diagnostic.path, diagnostic.message
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })
    }
}

/// Canonical guarded rule retained in compiled Financial Logic IR.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledCogRule {
    /// Stable source rule identifier.
    pub rule_id: String,
    /// Higher values are considered first.
    pub priority: i32,
    /// Validated deterministic predicate.
    pub guard: GuardExpr,
    /// Validated candidate action.
    pub action: CandidateAction,
    /// Content hash of this canonical rule.
    pub rule_hash: String,
}

/// Immutable, content-addressed Financial Logic IR artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledCog {
    /// Must equal [`COMPILED_COG_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Hash of canonical structured source.
    pub source_hash: String,
    /// Hash of this compiled artifact body.
    pub artifact_hash: String,
    /// Stable source program identifier.
    pub program_id: String,
    /// Human-readable source purpose.
    pub description: String,
    /// Exact listed universe, sorted by contract identifier.
    pub listed_universe: Vec<OptionContract>,
    /// Validated candidate grammar.
    pub candidate_grammar: CandidateGrammar,
    /// Canonical exact starting portfolio.
    pub initial_portfolio: CanonicalPortfolio,
    /// Canonical rules sorted by descending priority and identifier.
    pub rules: Vec<CompiledCogRule>,
    /// Optional declared scenario objective.
    pub objective: Option<ScenarioObjectiveSource>,
}

impl CompiledCog {
    /// Validate schema, hashes, exact contracts, grammar, and canonical rules.
    pub fn validate(&self) -> CogResult<()> {
        if self.schema_version != COMPILED_COG_SCHEMA_VERSION {
            return Err(CogError::InvalidDefinition(format!(
                "unsupported compiled COG schema {}",
                self.schema_version
            )));
        }
        self.candidate_grammar.validate()?;
        let universe = universe_map(&self.listed_universe)?;
        for position in &self.initial_portfolio.positions {
            let listed = universe
                .get(&position.contract.contract_id)
                .ok_or_else(|| {
                    CogError::InvalidDefinition(format!(
                        "compiled initial contract {} is outside the universe",
                        position.contract.contract_id.0
                    ))
                })?;
            if *listed != &position.contract {
                return Err(CogError::InvalidDefinition(format!(
                    "compiled initial contract {} conflicts with the universe",
                    position.contract.contract_id.0
                )));
            }
        }
        for rule in &self.rules {
            let expected =
                compiled_rule_hash(&rule.rule_id, rule.priority, &rule.guard, &rule.action)?;
            if expected != rule.rule_hash {
                return Err(CogError::InvalidDefinition(format!(
                    "compiled rule {} hash does not match its body",
                    rule.rule_id
                )));
            }
        }
        let expected = artifact_hash(self)?;
        if expected != self.artifact_hash {
            return Err(CogError::InvalidDefinition(
                "compiled COG artifact hash does not match its body".to_owned(),
            ));
        }
        Ok(())
    }

    /// Construct an invariant-checked initial runtime state.
    pub fn initial_state(&self) -> CogResult<PortfolioState> {
        self.validate()?;
        PortfolioState::from_positions(
            self.initial_portfolio
                .positions
                .iter()
                .map(|position| Position::new(position.contract.clone(), position.quantity))
                .collect::<CogResult<Vec<_>>>()?,
        )
    }

    /// Evaluate every source rule and compile a stable point-in-time action set.
    ///
    /// Rejected rule slots remain present with `legal = false`, making the
    /// exact legal mask available to GNN and later RL consumers.
    pub fn action_set(
        &self,
        state: &PortfolioState,
        context: &CandidateMarketContext,
    ) -> CogResult<CompiledActionSet> {
        self.validate()?;
        state.check_invariants()?;
        context.validate()?;
        self.validate_state_contracts(state)?;
        let state_hash = state.state_hash()?;
        let portfolio = live_portfolio(state)?;
        let mut candidates = Vec::new();
        for rule in &self.rules {
            if !rule.guard.evaluate(state, context)? {
                candidates.push(CompiledActionCandidate::new(
                    &self.artifact_hash,
                    rule,
                    None,
                    false,
                    Some("guard evaluated false".to_owned()),
                    0,
                )?);
                continue;
            }
            match self.candidate_grammar.compile_action(
                &rule.action,
                &portfolio,
                &self.listed_universe,
                context,
            ) {
                Ok(compiled) => {
                    for (branch_index, action) in compiled.into_iter().enumerate() {
                        candidates.push(CompiledActionCandidate::new(
                            &self.artifact_hash,
                            rule,
                            Some(action),
                            true,
                            None,
                            u32::try_from(branch_index).map_err(|_| {
                                CogError::NumericOverflow(
                                    "compiled candidate branch index".to_owned(),
                                )
                            })?,
                        )?);
                    }
                }
                Err(error) => candidates.push(CompiledActionCandidate::new(
                    &self.artifact_hash,
                    rule,
                    None,
                    false,
                    Some(error.to_string()),
                    0,
                )?),
            }
        }
        let legal_action_mask = candidates
            .iter()
            .map(|candidate| candidate.legal)
            .collect::<Vec<_>>();
        let body = CompiledActionSetIdentity {
            compiled_cog_hash: &self.artifact_hash,
            state_hash: &state_hash,
            context,
            candidates: &candidates,
            legal_action_mask: &legal_action_mask,
        };
        let action_set_hash = canonical_hash(COMPILED_ACTION_SET_SCHEMA_VERSION, &body)?;
        let action_set = CompiledActionSet {
            schema_version: COMPILED_ACTION_SET_SCHEMA_VERSION.to_owned(),
            action_set_hash,
            compiled_cog_hash: self.artifact_hash.clone(),
            state_hash,
            context: context.clone(),
            candidates,
            legal_action_mask,
        };
        action_set.validate()?;
        Ok(action_set)
    }

    /// Execute one indexed legal action using existing candidate and lifecycle
    /// semantics.
    ///
    /// Add/remove/resize actions return a new canonical design portfolio.
    /// Wait and exercise actions execute through [`CogRuntime`] and therefore
    /// produce a journaled runtime transition.
    pub fn execute_action(
        &self,
        state: &PortfolioState,
        action_set: &CompiledActionSet,
        action_index: usize,
        observation: &MarketObservation,
    ) -> CogResult<CompiledStepResult> {
        self.validate()?;
        state.check_invariants()?;
        action_set.validate()?;
        if action_set.compiled_cog_hash != self.artifact_hash {
            return Err(CogError::InvalidAction(
                "action set belongs to a different compiled COG".to_owned(),
            ));
        }
        if action_set.state_hash != state.state_hash()? {
            return Err(CogError::InvalidAction(
                "action set was compiled for a different runtime state".to_owned(),
            ));
        }
        let candidate = action_set.candidates.get(action_index).ok_or_else(|| {
            CogError::InvalidAction(format!("action index {action_index} is outside the set"))
        })?;
        if !candidate.legal {
            return Err(CogError::InvalidAction(format!(
                "candidate {} is not legal: {}",
                candidate.action_hash,
                candidate
                    .rejection_reason
                    .as_deref()
                    .unwrap_or("compiler rejected the action")
            )));
        }
        let compiled = candidate.compiled_action.as_ref().ok_or_else(|| {
            CogError::InvariantViolation("legal candidate has no compiled action".to_owned())
        })?;
        if matches!(compiled, CompiledCandidateAction::Wait) {
            let transition = CogRuntime.transition(state, &TransitionAction::Hold, observation)?;
            return Ok(CompiledStepResult::Runtime {
                rule_id: candidate.rule_id.clone(),
                action_hash: candidate.action_hash.clone(),
                transition: Box::new(transition),
            });
        }
        let portfolio = live_portfolio(state)?;
        match compiled.apply(&portfolio)? {
            CandidateActionEffect::Portfolio(portfolio) => Ok(CompiledStepResult::Design {
                rule_id: candidate.rule_id.clone(),
                action_hash: candidate.action_hash.clone(),
                portfolio,
            }),
            CandidateActionEffect::Runtime(action) => {
                let transition = CogRuntime.transition(state, &action, observation)?;
                Ok(CompiledStepResult::Runtime {
                    rule_id: candidate.rule_id.clone(),
                    action_hash: candidate.action_hash.clone(),
                    transition: Box::new(transition),
                })
            }
        }
    }

    fn validate_state_contracts(&self, state: &PortfolioState) -> CogResult<()> {
        let universe = universe_map(&self.listed_universe)?;
        for position in state.positions.values() {
            let listed = universe
                .get(&position.contract.contract_id)
                .ok_or_else(|| {
                    CogError::InvalidAction(format!(
                        "runtime contract {} is outside compiled universe",
                        position.contract.contract_id.0
                    ))
                })?;
            if *listed != &position.contract {
                return Err(CogError::InvalidAction(format!(
                    "runtime contract {} conflicts with compiled universe",
                    position.contract.contract_id.0
                )));
            }
        }
        Ok(())
    }
}

/// One compiler-owned action slot, whether dynamically legal or rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledActionCandidate {
    /// Stable source rule identifier.
    pub rule_id: String,
    /// Source-rule priority.
    pub priority: i32,
    /// Branch index when one bucket resolves to several exact contracts.
    pub branch_index: u32,
    /// Original typed source action.
    pub source_action: CandidateAction,
    /// Exact action after listed-universe resolution, absent when rejected.
    pub compiled_action: Option<CompiledCandidateAction>,
    /// Compiler-owned legality bit.
    pub legal: bool,
    /// Stable explanation when `legal` is false.
    pub rejection_reason: Option<String>,
    /// Content hash of this action slot.
    pub action_hash: String,
}

impl CompiledActionCandidate {
    fn new(
        compiled_cog_hash: &str,
        rule: &CompiledCogRule,
        compiled_action: Option<CompiledCandidateAction>,
        legal: bool,
        rejection_reason: Option<String>,
        branch_index: u32,
    ) -> CogResult<Self> {
        let identity = CompiledActionIdentity {
            compiled_cog_hash,
            rule_id: &rule.rule_id,
            priority: rule.priority,
            branch_index,
            source_action: &rule.action,
            compiled_action: &compiled_action,
            legal,
            rejection_reason: &rejection_reason,
        };
        let action_hash = canonical_hash(COMPILED_ACTION_SCHEMA_VERSION, &identity)?;
        Ok(Self {
            rule_id: rule.rule_id.clone(),
            priority: rule.priority,
            branch_index,
            source_action: rule.action.clone(),
            compiled_action,
            legal,
            rejection_reason,
            action_hash,
        })
    }
}

/// Point-in-time compiler output aligned with an explicit legality mask.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledActionSet {
    /// Must equal [`COMPILED_ACTION_SET_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Content hash of context, candidates, and mask.
    pub action_set_hash: String,
    /// Compiled program that produced this set.
    pub compiled_cog_hash: String,
    /// Runtime state for which this set was produced.
    pub state_hash: String,
    /// Point-in-time market coordinates used by guards and bucket resolution.
    pub context: CandidateMarketContext,
    /// Stable action slots ordered by compiled rule and branch.
    pub candidates: Vec<CompiledActionCandidate>,
    /// Boolean mask aligned one-to-one with `candidates`.
    pub legal_action_mask: Vec<bool>,
}

impl CompiledActionSet {
    /// Validate schema, candidate hashes, mask alignment, context, and set hash.
    pub fn validate(&self) -> CogResult<()> {
        if self.schema_version != COMPILED_ACTION_SET_SCHEMA_VERSION {
            return Err(CogError::InvalidDefinition(format!(
                "unsupported compiled action-set schema {}",
                self.schema_version
            )));
        }
        self.context.validate()?;
        if self.candidates.len() != self.legal_action_mask.len() {
            return Err(CogError::InvalidDefinition(
                "compiled action candidates and legal mask are not aligned".to_owned(),
            ));
        }
        for (candidate, mask) in self.candidates.iter().zip(&self.legal_action_mask) {
            if candidate.legal != *mask {
                return Err(CogError::InvalidDefinition(
                    "compiled action legal bit disagrees with the mask".to_owned(),
                ));
            }
            if candidate.legal
                && (candidate.compiled_action.is_none() || candidate.rejection_reason.is_some())
            {
                return Err(CogError::InvalidDefinition(
                    "legal compiled action must have an exact action and no rejection".to_owned(),
                ));
            }
            if !candidate.legal && candidate.rejection_reason.is_none() {
                return Err(CogError::InvalidDefinition(
                    "rejected compiled action must explain its rejection".to_owned(),
                ));
            }
            let identity = CompiledActionIdentity {
                compiled_cog_hash: &self.compiled_cog_hash,
                rule_id: &candidate.rule_id,
                priority: candidate.priority,
                branch_index: candidate.branch_index,
                source_action: &candidate.source_action,
                compiled_action: &candidate.compiled_action,
                legal: candidate.legal,
                rejection_reason: &candidate.rejection_reason,
            };
            let expected = canonical_hash(COMPILED_ACTION_SCHEMA_VERSION, &identity)?;
            if expected != candidate.action_hash {
                return Err(CogError::InvalidDefinition(format!(
                    "compiled candidate {} hash does not match its body",
                    candidate.rule_id
                )));
            }
        }
        let body = CompiledActionSetIdentity {
            compiled_cog_hash: &self.compiled_cog_hash,
            state_hash: &self.state_hash,
            context: &self.context,
            candidates: &self.candidates,
            legal_action_mask: &self.legal_action_mask,
        };
        let expected = canonical_hash(COMPILED_ACTION_SET_SCHEMA_VERSION, &body)?;
        if expected != self.action_set_hash {
            return Err(CogError::InvalidDefinition(
                "compiled action-set hash does not match its body".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Result of executing one compiler-approved candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompiledStepResult {
    /// Static options-design transition producing another canonical portfolio.
    Design {
        /// Source rule responsible for the transition.
        rule_id: String,
        /// Exact compiled action hash.
        action_hash: String,
        /// Resulting canonical design.
        portfolio: CanonicalPortfolio,
    },
    /// Lifecycle transition executed and journaled by [`CogRuntime`].
    Runtime {
        /// Source rule responsible for the transition.
        rule_id: String,
        /// Exact compiled action hash.
        action_hash: String,
        /// Invariant-checked next state and journal.
        transition: Box<TransitionResult>,
    },
}

/// Deterministic structured-source compiler.
#[derive(Debug, Default, Clone, Copy)]
pub struct CogCompiler;

impl CogCompiler {
    /// Parse UTF-8 JSON and compile it into immutable Financial Logic IR.
    pub fn compile_json(&self, source_json: &str) -> CompilationReport {
        match serde_json::from_str::<CogSource>(source_json) {
            Ok(source) => self.compile(&source),
            Err(error) => CompilationReport {
                artifact: None,
                diagnostics: vec![CompilerDiagnostic {
                    severity: DiagnosticSeverity::Error,
                    code: "E0001_JSON_PARSE".to_owned(),
                    path: "$".to_owned(),
                    message: format!(
                        "{} at line {}, column {}",
                        error,
                        error.line(),
                        error.column()
                    ),
                }],
            },
        }
    }

    /// Validate, normalize, and compile structured source.
    pub fn compile(&self, source: &CogSource) -> CompilationReport {
        let mut diagnostics = Vec::new();
        validate_source(source, &mut diagnostics);
        if diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
        {
            return CompilationReport {
                artifact: None,
                diagnostics,
            };
        }

        let source_hash = match canonical_hash(COG_SOURCE_SCHEMA_VERSION, source) {
            Ok(hash) => hash,
            Err(error) => {
                diagnostics.push(error_diagnostic(
                    "E0900_SOURCE_HASH",
                    "$",
                    error.to_string(),
                ));
                return CompilationReport {
                    artifact: None,
                    diagnostics,
                };
            }
        };
        let mut listed_universe = source.listed_universe.clone();
        listed_universe.sort_by(|left, right| left.contract_id.cmp(&right.contract_id));
        let universe = match universe_map(&listed_universe) {
            Ok(universe) => universe,
            Err(error) => {
                diagnostics.push(error_diagnostic(
                    "E0901_UNIVERSE",
                    "$.listed_universe",
                    error.to_string(),
                ));
                return CompilationReport {
                    artifact: None,
                    diagnostics,
                };
            }
        };
        let positions = source
            .initial_positions
            .iter()
            .map(|position| {
                let contract = universe.get(&position.contract_id).ok_or_else(|| {
                    CogError::InvalidDefinition(format!(
                        "unknown initial contract {}",
                        position.contract_id.0
                    ))
                })?;
                Position::new((*contract).clone(), position.quantity)
            })
            .collect::<CogResult<Vec<_>>>();
        let initial_portfolio = match positions.and_then(CanonicalPortfolio::from_positions) {
            Ok(portfolio) => portfolio,
            Err(error) => {
                diagnostics.push(error_diagnostic(
                    "E0902_INITIAL_PORTFOLIO",
                    "$.initial_positions",
                    error.to_string(),
                ));
                return CompilationReport {
                    artifact: None,
                    diagnostics,
                };
            }
        };
        let mut rules = Vec::with_capacity(source.rules.len());
        for rule in &source.rules {
            let rule_hash =
                match compiled_rule_hash(&rule.rule_id, rule.priority, &rule.guard, &rule.action) {
                    Ok(hash) => hash,
                    Err(error) => {
                        diagnostics.push(error_diagnostic(
                            "E0903_RULE_HASH",
                            "$.rules",
                            error.to_string(),
                        ));
                        return CompilationReport {
                            artifact: None,
                            diagnostics,
                        };
                    }
                };
            rules.push(CompiledCogRule {
                rule_id: rule.rule_id.clone(),
                priority: rule.priority,
                guard: rule.guard.clone(),
                action: rule.action.clone(),
                rule_hash,
            });
        }
        rules.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.rule_id.cmp(&right.rule_id))
        });
        let mut artifact = CompiledCog {
            schema_version: COMPILED_COG_SCHEMA_VERSION.to_owned(),
            source_hash,
            artifact_hash: String::new(),
            program_id: source.program_id.clone(),
            description: source.description.clone(),
            listed_universe,
            candidate_grammar: source.candidate_grammar.clone(),
            initial_portfolio,
            rules,
            objective: source.objective.clone(),
        };
        match artifact_hash(&artifact) {
            Ok(hash) => artifact.artifact_hash = hash,
            Err(error) => {
                diagnostics.push(error_diagnostic(
                    "E0904_ARTIFACT_HASH",
                    "$",
                    error.to_string(),
                ));
                return CompilationReport {
                    artifact: None,
                    diagnostics,
                };
            }
        }
        if let Err(error) = artifact.validate() {
            diagnostics.push(error_diagnostic(
                "E0905_ARTIFACT_VALIDATE",
                "$",
                error.to_string(),
            ));
            return CompilationReport {
                artifact: None,
                diagnostics,
            };
        }
        CompilationReport {
            artifact: Some(artifact),
            diagnostics,
        }
    }
}

#[derive(Serialize)]
struct CompiledRuleIdentity<'a> {
    rule_id: &'a str,
    priority: i32,
    guard: &'a GuardExpr,
    action: &'a CandidateAction,
}

fn compiled_rule_hash(
    rule_id: &str,
    priority: i32,
    guard: &GuardExpr,
    action: &CandidateAction,
) -> CogResult<String> {
    canonical_hash(
        COMPILED_RULE_SCHEMA_VERSION,
        &CompiledRuleIdentity {
            rule_id,
            priority,
            guard,
            action,
        },
    )
}

#[derive(Serialize)]
struct ArtifactIdentity<'a> {
    schema_version: &'a str,
    source_hash: &'a str,
    program_id: &'a str,
    description: &'a str,
    listed_universe: &'a [OptionContract],
    candidate_grammar: &'a CandidateGrammar,
    initial_portfolio: &'a CanonicalPortfolio,
    rules: &'a [CompiledCogRule],
    objective: &'a Option<ScenarioObjectiveSource>,
}

fn artifact_hash(artifact: &CompiledCog) -> CogResult<String> {
    canonical_hash(
        COMPILED_COG_SCHEMA_VERSION,
        &ArtifactIdentity {
            schema_version: &artifact.schema_version,
            source_hash: &artifact.source_hash,
            program_id: &artifact.program_id,
            description: &artifact.description,
            listed_universe: &artifact.listed_universe,
            candidate_grammar: &artifact.candidate_grammar,
            initial_portfolio: &artifact.initial_portfolio,
            rules: &artifact.rules,
            objective: &artifact.objective,
        },
    )
}

#[derive(Serialize)]
struct CompiledActionIdentity<'a> {
    compiled_cog_hash: &'a str,
    rule_id: &'a str,
    priority: i32,
    branch_index: u32,
    source_action: &'a CandidateAction,
    compiled_action: &'a Option<CompiledCandidateAction>,
    legal: bool,
    rejection_reason: &'a Option<String>,
}

#[derive(Serialize)]
struct CompiledActionSetIdentity<'a> {
    compiled_cog_hash: &'a str,
    state_hash: &'a str,
    context: &'a CandidateMarketContext,
    candidates: &'a [CompiledActionCandidate],
    legal_action_mask: &'a [bool],
}

fn live_leg_count(state: &PortfolioState) -> CogResult<u64> {
    u64::try_from(
        state
            .positions
            .values()
            .filter(|position| position.lifecycle == ContractLifecycle::Live)
            .count(),
    )
    .map_err(|_| CogError::NumericOverflow("live leg count".to_owned()))
}

fn live_gross_quantity(state: &PortfolioState) -> CogResult<u64> {
    state
        .positions
        .values()
        .filter(|position| position.lifecycle == ContractLifecycle::Live)
        .try_fold(0_u64, |total, position| {
            total
                .checked_add(position.open_quantity.unsigned_abs())
                .ok_or_else(|| CogError::NumericOverflow("live gross quantity".to_owned()))
        })
}

fn live_portfolio(state: &PortfolioState) -> CogResult<CanonicalPortfolio> {
    CanonicalPortfolio::from_positions(
        state
            .positions
            .values()
            .filter(|position| position.lifecycle == ContractLifecycle::Live)
            .map(|position| Position::new(position.contract.clone(), position.open_quantity))
            .collect::<CogResult<Vec<_>>>()?,
    )
}

fn universe_map(
    listed_universe: &[OptionContract],
) -> CogResult<BTreeMap<ContractId, &OptionContract>> {
    let mut universe = BTreeMap::new();
    for contract in listed_universe {
        contract.validate()?;
        if universe
            .insert(contract.contract_id.clone(), contract)
            .is_some()
        {
            return Err(CogError::InvalidDefinition(format!(
                "listed universe repeats contract {}",
                contract.contract_id.0
            )));
        }
    }
    Ok(universe)
}

fn validate_source(source: &CogSource, diagnostics: &mut Vec<CompilerDiagnostic>) {
    if source.schema_version != COG_SOURCE_SCHEMA_VERSION {
        diagnostics.push(error_diagnostic(
            "E0100_SCHEMA",
            "$.schema_version",
            format!(
                "expected {}, received {}",
                COG_SOURCE_SCHEMA_VERSION, source.schema_version
            ),
        ));
    }
    if source.program_id.trim().is_empty() {
        diagnostics.push(error_diagnostic(
            "E0101_PROGRAM_ID",
            "$.program_id",
            "program_id must not be empty",
        ));
    }
    if source.listed_universe.is_empty() {
        diagnostics.push(error_diagnostic(
            "E0200_EMPTY_UNIVERSE",
            "$.listed_universe",
            "listed_universe must not be empty",
        ));
    }
    let mut universe = BTreeMap::new();
    for (index, contract) in source.listed_universe.iter().enumerate() {
        let path = format!("$.listed_universe[{index}]");
        if let Err(error) = contract.validate() {
            diagnostics.push(error_diagnostic("E0201_CONTRACT", &path, error.to_string()));
        }
        if universe
            .insert(contract.contract_id.clone(), contract)
            .is_some()
        {
            diagnostics.push(error_diagnostic(
                "E0202_DUPLICATE_CONTRACT",
                format!("{path}.contract_id"),
                format!("contract {} is repeated", contract.contract_id.0),
            ));
        }
    }
    if let Err(error) = source.candidate_grammar.validate() {
        diagnostics.push(error_diagnostic(
            "E0300_GRAMMAR",
            "$.candidate_grammar",
            error.to_string(),
        ));
    }
    for (index, position) in source.initial_positions.iter().enumerate() {
        let path = format!("$.initial_positions[{index}]");
        if position.quantity == 0 {
            diagnostics.push(error_diagnostic(
                "E0400_ZERO_POSITION",
                format!("{path}.quantity"),
                "initial position quantity must be non-zero",
            ));
        }
        if !universe.contains_key(&position.contract_id) {
            diagnostics.push(error_diagnostic(
                "E0401_UNKNOWN_INITIAL_CONTRACT",
                format!("{path}.contract_id"),
                format!(
                    "contract {} is not in listed_universe",
                    position.contract_id.0
                ),
            ));
        }
    }
    if source.rules.is_empty() {
        diagnostics.push(warning_diagnostic(
            "W0500_NO_RULES",
            "$.rules",
            "program has no guarded business rules",
        ));
    }
    let mut rule_ids = BTreeSet::new();
    for (index, rule) in source.rules.iter().enumerate() {
        let path = format!("$.rules[{index}]");
        if rule.rule_id.trim().is_empty() {
            diagnostics.push(error_diagnostic(
                "E0501_RULE_ID",
                format!("{path}.rule_id"),
                "rule_id must not be empty",
            ));
        } else if !rule_ids.insert(rule.rule_id.as_str()) {
            diagnostics.push(error_diagnostic(
                "E0502_DUPLICATE_RULE",
                format!("{path}.rule_id"),
                format!("rule {} is repeated", rule.rule_id),
            ));
        }
        validate_guard(
            &rule.guard,
            &format!("{path}.guard"),
            &universe,
            diagnostics,
        );
        validate_action(
            &rule.action,
            &format!("{path}.action"),
            source,
            &universe,
            diagnostics,
        );
    }
    if let Some(objective) = &source.objective {
        validate_objective(objective, diagnostics);
    }
}

fn validate_guard(
    guard: &GuardExpr,
    path: &str,
    universe: &BTreeMap<ContractId, &OptionContract>,
    diagnostics: &mut Vec<CompilerDiagnostic>,
) {
    match guard {
        GuardExpr::All { guards } | GuardExpr::Any { guards } => {
            if guards.is_empty() {
                diagnostics.push(error_diagnostic(
                    "E0600_EMPTY_GUARD_LIST",
                    path,
                    "all/any guard list must not be empty",
                ));
            }
            for (index, child) in guards.iter().enumerate() {
                validate_guard(
                    child,
                    &format!("{path}.guards[{index}]"),
                    universe,
                    diagnostics,
                );
            }
        }
        GuardExpr::Not { expr } => {
            validate_guard(expr, &format!("{path}.expr"), universe, diagnostics);
        }
        GuardExpr::SpotAtLeast { spot_micros } | GuardExpr::SpotAtMost { spot_micros } => {
            if *spot_micros < 0 {
                diagnostics.push(error_diagnostic(
                    "E0601_NEGATIVE_SPOT",
                    format!("{path}.spot_micros"),
                    "spot guard must be non-negative",
                ));
            }
        }
        GuardExpr::HasLivePosition { contract_id, .. }
        | GuardExpr::ExerciseOpen { contract_id } => {
            if !universe.contains_key(contract_id) {
                diagnostics.push(error_diagnostic(
                    "E0602_UNKNOWN_GUARD_CONTRACT",
                    format!("{path}.contract_id"),
                    format!("contract {} is not in listed_universe", contract_id.0),
                ));
            }
        }
        GuardExpr::Always
        | GuardExpr::LiveLegCountAtMost { .. }
        | GuardExpr::LiveLegCountAtLeast { .. }
        | GuardExpr::GrossQuantityAtMost { .. }
        | GuardExpr::GrossQuantityAtLeast { .. } => {}
    }
}

fn validate_action(
    action: &CandidateAction,
    path: &str,
    source: &CogSource,
    universe: &BTreeMap<ContractId, &OptionContract>,
    diagnostics: &mut Vec<CompilerDiagnostic>,
) {
    match action {
        CandidateAction::AddLeg {
            expiry_bucket,
            strike_or_delta_bucket,
            quantity,
            ..
        } => {
            if !source
                .candidate_grammar
                .expiry_buckets
                .iter()
                .any(|bucket| bucket.bucket_id == *expiry_bucket)
            {
                diagnostics.push(error_diagnostic(
                    "E0700_UNKNOWN_EXPIRY_BUCKET",
                    format!("{path}.expiry_bucket"),
                    format!("unknown expiry bucket {expiry_bucket}"),
                ));
            }
            if !source
                .candidate_grammar
                .strike_or_delta_buckets
                .iter()
                .any(|bucket| bucket.bucket_id() == strike_or_delta_bucket)
            {
                diagnostics.push(error_diagnostic(
                    "E0701_UNKNOWN_STRIKE_BUCKET",
                    format!("{path}.strike_or_delta_bucket"),
                    format!("unknown strike/delta bucket {strike_or_delta_bucket}"),
                ));
            }
            if !source
                .candidate_grammar
                .allowed_add_quantities
                .contains(quantity)
            {
                diagnostics.push(error_diagnostic(
                    "E0702_ADD_QUANTITY",
                    format!("{path}.quantity"),
                    format!("quantity {quantity} is outside candidate grammar"),
                ));
            }
        }
        CandidateAction::RemoveLeg { contract } => {
            validate_action_contract(contract, path, universe, diagnostics);
        }
        CandidateAction::ResizeLeg {
            contract,
            delta_quantity,
        } => {
            validate_action_contract(contract, path, universe, diagnostics);
            if !source
                .candidate_grammar
                .allowed_resize_quantities
                .contains(delta_quantity)
            {
                diagnostics.push(error_diagnostic(
                    "E0703_RESIZE_QUANTITY",
                    format!("{path}.delta_quantity"),
                    format!("quantity {delta_quantity} is outside candidate grammar"),
                ));
            }
        }
        CandidateAction::Exercise { contract, quantity } => {
            validate_action_contract(contract, path, universe, diagnostics);
            if *quantity <= 0 {
                diagnostics.push(error_diagnostic(
                    "E0704_EXERCISE_QUANTITY",
                    format!("{path}.quantity"),
                    "exercise quantity must be positive",
                ));
            }
        }
        CandidateAction::Wait => {}
    }
}

fn validate_action_contract(
    contract: &ContractId,
    path: &str,
    universe: &BTreeMap<ContractId, &OptionContract>,
    diagnostics: &mut Vec<CompilerDiagnostic>,
) {
    if !universe.contains_key(contract) {
        diagnostics.push(error_diagnostic(
            "E0705_UNKNOWN_ACTION_CONTRACT",
            format!("{path}.contract"),
            format!("contract {} is not in listed_universe", contract.0),
        ));
    }
}

fn validate_objective(
    objective: &ScenarioObjectiveSource,
    diagnostics: &mut Vec<CompilerDiagnostic>,
) {
    let path = "$.objective";
    if objective.objective_name.trim().is_empty() {
        diagnostics.push(error_diagnostic(
            "E0800_OBJECTIVE_NAME",
            format!("{path}.objective_name"),
            "objective_name must not be empty",
        ));
    }
    if objective.scenario_spots_micros.is_empty()
        || objective.scenario_spots_micros.len() != objective.probability_millionths.len()
    {
        diagnostics.push(error_diagnostic(
            "E0801_OBJECTIVE_SCENARIOS",
            path,
            "scenario spots and probabilities must be non-empty and aligned",
        ));
    }
    if objective.scenario_spots_micros.iter().any(|spot| *spot < 0) {
        diagnostics.push(error_diagnostic(
            "E0802_OBJECTIVE_SPOT",
            format!("{path}.scenario_spots_micros"),
            "scenario spots must be non-negative",
        ));
    }
    let total_probability = objective
        .probability_millionths
        .iter()
        .map(|probability| u64::from(*probability))
        .sum::<u64>();
    if total_probability != 1_000_000 {
        diagnostics.push(error_diagnostic(
            "E0803_OBJECTIVE_PROBABILITY",
            format!("{path}.probability_millionths"),
            format!("probabilities sum to {total_probability}, expected 1000000"),
        ));
    }
    if objective.downside_weight_bps > 100_000 {
        diagnostics.push(error_diagnostic(
            "E0804_DOWNSIDE_WEIGHT",
            format!("{path}.downside_weight_bps"),
            "downside weight must not exceed 100000 bps",
        ));
    }
    if objective.gross_contract_penalty_micros < 0 {
        diagnostics.push(error_diagnostic(
            "E0805_GROSS_PENALTY",
            format!("{path}.gross_contract_penalty_micros"),
            "gross contract penalty must be non-negative",
        ));
    }
}

fn error_diagnostic(
    code: impl Into<String>,
    path: impl Into<String>,
    message: impl Into<String>,
) -> CompilerDiagnostic {
    CompilerDiagnostic {
        severity: DiagnosticSeverity::Error,
        code: code.into(),
        path: path.into(),
        message: message.into(),
    }
}

fn warning_diagnostic(
    code: impl Into<String>,
    path: impl Into<String>,
    message: impl Into<String>,
) -> CompilerDiagnostic {
    CompilerDiagnostic {
        severity: DiagnosticSeverity::Warning,
        code: code.into(),
        path: path.into(),
        message: message.into(),
    }
}
