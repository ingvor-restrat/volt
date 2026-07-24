//! Typed, point-in-time candidate actions for bounded portfolio search.
//!
//! This module compiles a small action vocabulary against an explicitly
//! supplied listed-contract universe. It decides structural legality and
//! exercise authority only. Liquidity, capital, margin, ranking, dominance,
//! and clustering remain policy layers outside the semantic crate.
//! Exercise compilation checks static long quantity and schedule opportunity;
//! [`crate::CogRuntime`] remains authoritative for live lifecycle state.
//!
//! An [`AddLeg`](CandidateAction::AddLeg) bucket may resolve to several listed
//! contracts. The returned vector preserves that branch instead of silently
//! selecting a contract. Callers are responsible for retaining or rejecting
//! every realization under a declared search policy.

#![warn(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::canonical::{CanonicalPortfolio, CanonicalPosition};
use crate::contract::{
    ContractId, OptionContract, OptionRight, Position, PriceMicros, TimestampNs,
};
use crate::hash::canonical_hash;
use crate::runtime::TransitionAction;
use crate::{CogError, CogResult};

/// Schema used to serialize [`CandidateGrammar`].
pub const CANDIDATE_GRAMMAR_SCHEMA_VERSION: &str = "volt.candidate_grammar.v1";
/// Domain separator used when hashing a [`CandidateAction`].
pub const CANDIDATE_ACTION_SCHEMA_VERSION: &str = "volt.candidate_action.v1";

/// Inclusive time-to-expiry interval used by `AddLeg`.
///
/// Bounds are nanoseconds from the point-in-time valuation clock. A bucket
/// never infers calendar or trading-day conventions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpiryBucket {
    /// Stable identifier referenced by candidate actions.
    pub bucket_id: String,
    /// Inclusive lower time-to-expiry bound in nanoseconds.
    pub minimum_time_to_expiry_ns: i64,
    /// Inclusive upper time-to-expiry bound in nanoseconds.
    pub maximum_time_to_expiry_ns: i64,
}

impl ExpiryBucket {
    fn validate(&self) -> CogResult<()> {
        require_identifier(&self.bucket_id, "expiry bucket_id")?;
        if self.minimum_time_to_expiry_ns < 0
            || self.maximum_time_to_expiry_ns < self.minimum_time_to_expiry_ns
        {
            return Err(CogError::InvalidDefinition(format!(
                "expiry bucket {} has invalid time-to-expiry bounds",
                self.bucket_id
            )));
        }
        Ok(())
    }

    fn contains(&self, contract: &OptionContract, valuation_time: TimestampNs) -> bool {
        contract
            .expiry()
            .0
            .checked_sub(valuation_time.0)
            .is_some_and(|time_to_expiry| {
                self.minimum_time_to_expiry_ns <= time_to_expiry
                    && time_to_expiry <= self.maximum_time_to_expiry_ns
            })
    }
}

/// Coordinate system and inclusive interval used to select a listed strike.
///
/// Delta values are inputs in [`CandidateMarketContext`]; this module does not
/// calculate or validate an option-pricing model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "coordinate", rename_all = "snake_case")]
pub enum StrikeOrDeltaBucket {
    /// Match the contract's exact strike price interval.
    StrikeRange {
        /// Stable identifier referenced by candidate actions.
        bucket_id: String,
        /// Inclusive lower strike in currency micros per underlying unit.
        minimum_strike_micros: i64,
        /// Inclusive upper strike in currency micros per underlying unit.
        maximum_strike_micros: i64,
    },
    /// Match strike divided by the point-in-time reference spot.
    ///
    /// The fixed-point ratio is truncated toward zero to integral basis
    /// points, making boundary behavior deterministic.
    MoneynessBps {
        /// Stable identifier referenced by candidate actions.
        bucket_id: String,
        /// Inclusive lower strike/spot ratio, where 10,000 is at-the-money.
        minimum_moneyness_bps: i32,
        /// Inclusive upper strike/spot ratio, where 10,000 is at-the-money.
        maximum_moneyness_bps: i32,
    },
    /// Match an externally supplied absolute option delta.
    AbsoluteDeltaBps {
        /// Stable identifier referenced by candidate actions.
        bucket_id: String,
        /// Inclusive lower absolute delta, where 10,000 is one delta.
        minimum_absolute_delta_bps: u16,
        /// Inclusive upper absolute delta, where 10,000 is one delta.
        maximum_absolute_delta_bps: u16,
    },
}

impl StrikeOrDeltaBucket {
    /// Return the stable identifier shared by all coordinate variants.
    pub fn bucket_id(&self) -> &str {
        match self {
            Self::StrikeRange { bucket_id, .. }
            | Self::MoneynessBps { bucket_id, .. }
            | Self::AbsoluteDeltaBps { bucket_id, .. } => bucket_id,
        }
    }

    fn validate(&self) -> CogResult<()> {
        require_identifier(self.bucket_id(), "strike/delta bucket_id")?;
        match self {
            Self::StrikeRange {
                minimum_strike_micros,
                maximum_strike_micros,
                ..
            } if *minimum_strike_micros <= 0 || maximum_strike_micros < minimum_strike_micros => {
                Err(CogError::InvalidDefinition(format!(
                    "strike bucket {} has invalid bounds",
                    self.bucket_id()
                )))
            }
            Self::MoneynessBps {
                minimum_moneyness_bps,
                maximum_moneyness_bps,
                ..
            } if *minimum_moneyness_bps <= 0 || maximum_moneyness_bps < minimum_moneyness_bps => {
                Err(CogError::InvalidDefinition(format!(
                    "moneyness bucket {} has invalid bounds",
                    self.bucket_id()
                )))
            }
            Self::AbsoluteDeltaBps {
                minimum_absolute_delta_bps,
                maximum_absolute_delta_bps,
                ..
            } if maximum_absolute_delta_bps < minimum_absolute_delta_bps
                || *maximum_absolute_delta_bps > 10_000 =>
            {
                Err(CogError::InvalidDefinition(format!(
                    "delta bucket {} has invalid bounds",
                    self.bucket_id()
                )))
            }
            _ => Ok(()),
        }
    }

    fn contains(
        &self,
        contract: &OptionContract,
        context: &CandidateMarketContext,
    ) -> CogResult<bool> {
        match self {
            Self::StrikeRange {
                minimum_strike_micros,
                maximum_strike_micros,
                ..
            } => Ok(*minimum_strike_micros <= contract.strike.0
                && contract.strike.0 <= *maximum_strike_micros),
            Self::MoneynessBps {
                minimum_moneyness_bps,
                maximum_moneyness_bps,
                ..
            } => {
                let moneyness_bps = i128::from(contract.strike.0)
                    .checked_mul(10_000)
                    .and_then(|value| value.checked_div(i128::from(context.reference_spot.0)))
                    .ok_or_else(|| {
                        CogError::NumericOverflow("candidate moneyness calculation".to_owned())
                    })?;
                Ok(i128::from(*minimum_moneyness_bps) <= moneyness_bps
                    && moneyness_bps <= i128::from(*maximum_moneyness_bps))
            }
            Self::AbsoluteDeltaBps {
                minimum_absolute_delta_bps,
                maximum_absolute_delta_bps,
                ..
            } => {
                let delta_bps = context
                    .absolute_delta_bps
                    .get(&contract.contract_id)
                    .ok_or_else(|| {
                        CogError::InvalidDefinition(format!(
                            "delta bucket requires a point-in-time delta for {}",
                            contract.contract_id.0
                        ))
                    })?;
                Ok(minimum_absolute_delta_bps <= delta_bps
                    && delta_bps <= maximum_absolute_delta_bps)
            }
        }
    }
}

/// Point-in-time market coordinates needed to resolve grammar buckets.
///
/// The context is intentionally observation-only. It cannot change contract
/// identity, exercise schedules, settlement, or multiplier semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateMarketContext {
    /// Decision clock as Unix epoch nanoseconds.
    pub valuation_time: TimestampNs,
    /// Positive underlying spot in currency micros per underlying unit.
    pub reference_spot: PriceMicros,
    /// Absolute delta in basis points, where 10_000 represents one unit.
    pub absolute_delta_bps: BTreeMap<ContractId, u16>,
}

impl CandidateMarketContext {
    /// Check time, spot, and externally supplied delta bounds.
    pub fn validate(&self) -> CogResult<()> {
        if self.valuation_time.0 < 0 || self.reference_spot.0 <= 0 {
            return Err(CogError::InvalidDefinition(
                "candidate context requires non-negative time and positive spot".to_owned(),
            ));
        }
        if self
            .absolute_delta_bps
            .values()
            .any(|delta| *delta > 10_000)
        {
            return Err(CogError::InvalidDefinition(
                "candidate absolute delta must be at most 10000 bps".to_owned(),
            ));
        }
        if self
            .absolute_delta_bps
            .keys()
            .any(|contract_id| contract_id.0.trim().is_empty())
        {
            return Err(CogError::InvalidDefinition(
                "candidate delta contract ids must not be empty".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Versioned policy vocabulary for typed option-candidate actions.
///
/// Quantity sets contain signed contract deltas and may not contain zero.
/// Portfolio-size, liquidity, and capital limits are deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateGrammar {
    /// Must equal [`CANDIDATE_GRAMMAR_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Declared time-to-expiry buckets addressable by `AddLeg`.
    pub expiry_buckets: Vec<ExpiryBucket>,
    /// Declared strike, moneyness, or delta buckets addressable by `AddLeg`.
    pub strike_or_delta_buckets: Vec<StrikeOrDeltaBucket>,
    /// Signed quantities admitted for `AddLeg`.
    pub allowed_add_quantities: BTreeSet<i64>,
    /// Signed quantity deltas admitted for `ResizeLeg`.
    pub allowed_resize_quantities: BTreeSet<i64>,
}

impl CandidateGrammar {
    /// Construct and validate a versioned candidate grammar.
    pub fn new(
        expiry_buckets: Vec<ExpiryBucket>,
        strike_or_delta_buckets: Vec<StrikeOrDeltaBucket>,
        allowed_add_quantities: BTreeSet<i64>,
        allowed_resize_quantities: BTreeSet<i64>,
    ) -> CogResult<Self> {
        let grammar = Self {
            schema_version: CANDIDATE_GRAMMAR_SCHEMA_VERSION.to_owned(),
            expiry_buckets,
            strike_or_delta_buckets,
            allowed_add_quantities,
            allowed_resize_quantities,
        };
        grammar.validate()?;
        Ok(grammar)
    }

    /// Validate schema, bucket identifiers and bounds, and quantity sets.
    pub fn validate(&self) -> CogResult<()> {
        if self.schema_version != CANDIDATE_GRAMMAR_SCHEMA_VERSION {
            return Err(CogError::InvalidDefinition(format!(
                "unsupported candidate grammar schema {}",
                self.schema_version
            )));
        }
        if self.expiry_buckets.is_empty()
            || self.strike_or_delta_buckets.is_empty()
            || self.allowed_add_quantities.is_empty()
            || self.allowed_resize_quantities.is_empty()
        {
            return Err(CogError::InvalidDefinition(
                "candidate grammar buckets and quantity sets must not be empty".to_owned(),
            ));
        }
        let mut ids = BTreeSet::new();
        for bucket in &self.expiry_buckets {
            bucket.validate()?;
            if !ids.insert(bucket.bucket_id.as_str()) {
                return Err(CogError::InvalidDefinition(format!(
                    "duplicate candidate bucket_id {}",
                    bucket.bucket_id
                )));
            }
        }
        for bucket in &self.strike_or_delta_buckets {
            bucket.validate()?;
            if !ids.insert(bucket.bucket_id()) {
                return Err(CogError::InvalidDefinition(format!(
                    "duplicate candidate bucket_id {}",
                    bucket.bucket_id()
                )));
            }
        }
        if self
            .allowed_add_quantities
            .iter()
            .chain(&self.allowed_resize_quantities)
            .any(|quantity| *quantity == 0)
        {
            return Err(CogError::InvalidDefinition(
                "candidate grammar quantity deltas must be non-zero".to_owned(),
            ));
        }
        Ok(())
    }

    /// Compile one typed grammar action against a point-in-time listed
    /// universe and the current canonical portfolio. `AddLeg` can yield more
    /// than one realization when a declared bucket contains multiple listed
    /// contracts; callers must retain that branching explicitly.
    pub fn compile_action(
        &self,
        action: &CandidateAction,
        portfolio: &CanonicalPortfolio,
        listed_universe: &[OptionContract],
        context: &CandidateMarketContext,
    ) -> CogResult<Vec<CompiledCandidateAction>> {
        self.validate()?;
        context.validate()?;
        let mut listed_ids = BTreeSet::new();
        for contract in listed_universe {
            contract.validate()?;
            if !listed_ids.insert(&contract.contract_id) {
                return Err(CogError::InvalidDefinition(format!(
                    "listed candidate universe repeats contract {}",
                    contract.contract_id.0
                )));
            }
        }
        match action {
            CandidateAction::AddLeg {
                right,
                expiry_bucket,
                strike_or_delta_bucket,
                quantity,
            } => {
                if !self.allowed_add_quantities.contains(quantity) {
                    return Err(CogError::InvalidAction(format!(
                        "AddLeg quantity {quantity} is outside the grammar"
                    )));
                }
                let expiry_bucket = self.expiry_bucket(expiry_bucket)?;
                let strike_or_delta_bucket = self.strike_or_delta_bucket(strike_or_delta_bucket)?;
                let mut ordered = listed_universe.iter().collect::<Vec<_>>();
                ordered.sort_by(|left, right| left.contract_id.cmp(&right.contract_id));
                let mut compiled = Vec::new();
                for contract in ordered {
                    if contract.right == *right
                        && expiry_bucket.contains(contract, context.valuation_time)
                        && strike_or_delta_bucket.contains(contract, context)?
                    {
                        compiled.push(CompiledCandidateAction::AddLeg {
                            contract: contract.clone(),
                            quantity: *quantity,
                        });
                    }
                }
                if compiled.is_empty() {
                    return Err(CogError::InvalidAction(
                        "AddLeg matches no point-in-time listed contract".to_owned(),
                    ));
                }
                Ok(compiled)
            }
            CandidateAction::RemoveLeg { contract } => {
                require_position(portfolio, contract)?;
                Ok(vec![CompiledCandidateAction::RemoveLeg {
                    contract: contract.clone(),
                }])
            }
            CandidateAction::ResizeLeg {
                contract,
                delta_quantity,
            } => {
                if !self.allowed_resize_quantities.contains(delta_quantity) {
                    return Err(CogError::InvalidAction(format!(
                        "ResizeLeg delta {delta_quantity} is outside the grammar"
                    )));
                }
                let position = require_position(portfolio, contract)?;
                let resized = position
                    .quantity
                    .checked_add(*delta_quantity)
                    .ok_or_else(|| {
                        CogError::NumericOverflow("candidate resize quantity".to_owned())
                    })?;
                if resized == 0 {
                    return Err(CogError::InvalidAction(
                        "ResizeLeg may not remove a leg; use RemoveLeg".to_owned(),
                    ));
                }
                Ok(vec![CompiledCandidateAction::ResizeLeg {
                    contract: contract.clone(),
                    delta_quantity: *delta_quantity,
                }])
            }
            CandidateAction::Wait => Ok(vec![CompiledCandidateAction::Wait]),
            CandidateAction::Exercise { contract, quantity } => {
                if *quantity <= 0 {
                    return Err(CogError::InvalidAction(
                        "Exercise quantity must be positive".to_owned(),
                    ));
                }
                let position = require_position(portfolio, contract)?;
                if position.quantity < *quantity {
                    return Err(CogError::InvalidAction(format!(
                        "Exercise requires {quantity} long contracts but {} is open",
                        position.quantity
                    )));
                }
                if !position
                    .contract
                    .exercise
                    .permits_exercise(context.valuation_time)
                {
                    return Err(CogError::InvalidAction(
                        "Exercise is outside the contract opportunity set".to_owned(),
                    ));
                }
                Ok(vec![CompiledCandidateAction::Exercise {
                    contract: contract.clone(),
                    quantity: *quantity,
                }])
            }
        }
    }

    fn expiry_bucket(&self, bucket_id: &str) -> CogResult<&ExpiryBucket> {
        self.expiry_buckets
            .iter()
            .find(|bucket| bucket.bucket_id == bucket_id)
            .ok_or_else(|| CogError::InvalidAction(format!("unknown expiry bucket {bucket_id}")))
    }

    fn strike_or_delta_bucket(&self, bucket_id: &str) -> CogResult<&StrikeOrDeltaBucket> {
        self.strike_or_delta_buckets
            .iter()
            .find(|bucket| bucket.bucket_id() == bucket_id)
            .ok_or_else(|| {
                CogError::InvalidAction(format!("unknown strike/delta bucket {bucket_id}"))
            })
    }
}

/// Unresolved action vocabulary proposed by a search policy or model.
///
/// Named strategies are intentionally not part of this grammar. `RemoveLeg`
/// and `ResizeLeg` require exact listed-contract identities, while `AddLeg`
/// resolves declared point-in-time buckets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum CandidateAction {
    /// Add signed quantity in every matching listed contract branch.
    AddLeg {
        /// Call or put right required of a matching contract.
        right: OptionRight,
        /// Identifier of a declared [`ExpiryBucket`].
        expiry_bucket: String,
        /// Identifier of a declared [`StrikeOrDeltaBucket`].
        strike_or_delta_bucket: String,
        /// Signed number of contracts to add.
        quantity: i64,
    },
    /// Remove one exact open contract position, regardless of its quantity.
    RemoveLeg {
        /// Exact open contract to remove.
        contract: ContractId,
    },
    /// Change one exact open contract position without removing it.
    ResizeLeg {
        /// Exact open contract to resize.
        contract: ContractId,
        /// Signed quantity change. A change to zero must use `RemoveLeg`.
        delta_quantity: i64,
    },
    /// Preserve the current candidate portfolio.
    Wait,
    /// Exercise a positive quantity of an exact long position.
    Exercise {
        /// Exact long contract whose right is exercised.
        contract: ContractId,
        /// Positive number of contracts to exercise.
        quantity: i64,
    },
}

impl CandidateAction {
    /// Hash this unresolved action with a versioned domain separator.
    pub fn semantic_hash(&self) -> CogResult<String> {
        canonical_hash(CANDIDATE_ACTION_SCHEMA_VERSION, self)
    }
}

/// Exact realization emitted by [`CandidateGrammar::compile_action`].
///
/// Compiled actions contain no bucket ambiguity. Applying an exercise action
/// yields the existing runtime action rather than mutating a static portfolio.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "compiled_action", rename_all = "snake_case")]
pub enum CompiledCandidateAction {
    /// Add an exact listed contract.
    AddLeg {
        /// Fully typed listed-contract definition.
        contract: OptionContract,
        /// Signed number of contracts to add.
        quantity: i64,
    },
    /// Remove one exact open contract position.
    RemoveLeg {
        /// Exact open contract to remove.
        contract: ContractId,
    },
    /// Resize one exact open contract position.
    ResizeLeg {
        /// Exact open contract to resize.
        contract: ContractId,
        /// Signed quantity change.
        delta_quantity: i64,
    },
    /// Preserve the candidate portfolio.
    Wait,
    /// Delegate an authorized holder exercise to the runtime.
    Exercise {
        /// Exact contract to exercise.
        contract: ContractId,
        /// Positive exercise quantity.
        quantity: i64,
    },
}

/// Result of applying one exact compiled candidate action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateActionEffect {
    /// New canonical static portfolio after a portfolio mutation or wait.
    Portfolio(CanonicalPortfolio),
    /// Dynamic action that must execute through [`crate::CogRuntime`].
    Runtime(TransitionAction),
}

impl CompiledCandidateAction {
    /// Apply this exact action to a canonical portfolio.
    ///
    /// Exercise is returned as a runtime action. This method does not simulate
    /// exercise emissions or mutate lifecycle state.
    pub fn apply(&self, portfolio: &CanonicalPortfolio) -> CogResult<CandidateActionEffect> {
        match self {
            Self::AddLeg { contract, quantity } => {
                let mut positions = to_positions(portfolio)?;
                positions.push(Position::new(contract.clone(), *quantity)?);
                Ok(CandidateActionEffect::Portfolio(
                    CanonicalPortfolio::from_positions(positions)?,
                ))
            }
            Self::RemoveLeg { contract } => {
                require_position(portfolio, contract)?;
                let positions = portfolio
                    .positions
                    .iter()
                    .filter(|position| position.contract.contract_id != *contract)
                    .map(position_from_canonical)
                    .collect::<CogResult<Vec<_>>>()?;
                Ok(CandidateActionEffect::Portfolio(
                    CanonicalPortfolio::from_positions(positions)?,
                ))
            }
            Self::ResizeLeg {
                contract,
                delta_quantity,
            } => {
                let current = require_position(portfolio, contract)?;
                let resized = current
                    .quantity
                    .checked_add(*delta_quantity)
                    .ok_or_else(|| {
                        CogError::NumericOverflow("candidate resize quantity".to_owned())
                    })?;
                if resized == 0 {
                    return Err(CogError::InvalidAction(
                        "ResizeLeg may not remove a leg; use RemoveLeg".to_owned(),
                    ));
                }
                let positions = portfolio
                    .positions
                    .iter()
                    .map(|position| {
                        Position::new(
                            position.contract.clone(),
                            if position.contract.contract_id == *contract {
                                resized
                            } else {
                                position.quantity
                            },
                        )
                    })
                    .collect::<CogResult<Vec<_>>>()?;
                Ok(CandidateActionEffect::Portfolio(
                    CanonicalPortfolio::from_positions(positions)?,
                ))
            }
            Self::Wait => Ok(CandidateActionEffect::Portfolio(portfolio.clone())),
            Self::Exercise { contract, quantity } => {
                Ok(CandidateActionEffect::Runtime(TransitionAction::Exercise {
                    contract_id: contract.clone(),
                    quantity: *quantity,
                }))
            }
        }
    }
}

fn require_position<'a>(
    portfolio: &'a CanonicalPortfolio,
    contract: &ContractId,
) -> CogResult<&'a CanonicalPosition> {
    portfolio
        .positions
        .iter()
        .find(|position| position.contract.contract_id == *contract)
        .ok_or_else(|| {
            CogError::InvalidAction(format!(
                "candidate portfolio has no open contract {}",
                contract.0
            ))
        })
}

fn to_positions(portfolio: &CanonicalPortfolio) -> CogResult<Vec<Position>> {
    portfolio
        .positions
        .iter()
        .map(position_from_canonical)
        .collect()
}

fn position_from_canonical(position: &CanonicalPosition) -> CogResult<Position> {
    Position::new(position.contract.clone(), position.quantity)
}

fn require_identifier(value: &str, name: &str) -> CogResult<()> {
    if value.trim().is_empty() {
        return Err(CogError::InvalidDefinition(format!(
            "{name} must not be empty"
        )));
    }
    Ok(())
}
