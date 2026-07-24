//! Listed option identity, exercise, settlement, and signed positions.

use serde::{Deserialize, Serialize};

use crate::{CogError, CogResult};

/// Schema identifier serialized with [`OptionContract`].
pub const CONTRACT_SCHEMA_VERSION: &str = "volt.contract.v1";

/// Stable identity of one exact listed contract.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContractId(pub String);

/// Stable identity of an underlying asset.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UnderlyingId(pub String);

/// Currency code used by prices and cash ledgers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Currency(pub String);

/// Unix-epoch timestamp in nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TimestampNs(pub i64);

/// Currency amount or price in millionths of one currency unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PriceMicros(pub i64);

/// Direction of the option holder's terminal exercise payoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionRight {
    /// Right to buy the underlying at the strike.
    Call,
    /// Right to sell the underlying at the strike.
    Put,
}

/// Form in which exercise or assignment is discharged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Settlement {
    /// Pay intrinsic value into the cash ledger.
    Cash,
    /// Exchange strike cash and underlying units.
    Physical,
}

/// Exercise opportunities are explicit data rather than implicit option-style
/// behavior. American intervals are inclusive at both ends.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "style", rename_all = "snake_case")]
pub enum ExerciseSchedule {
    /// Exercise is permitted only at expiry.
    European {
        /// Sole exercise opportunity.
        expiry: TimestampNs,
    },
    /// Exercise is permitted at a finite ordered set of instants.
    Bermudan {
        /// Strictly increasing, nonempty exercise opportunities.
        opportunities: Vec<TimestampNs>,
    },
    /// Exercise is permitted throughout an inclusive interval.
    American {
        /// First permitted exercise instant.
        first_exercise: TimestampNs,
        /// Last permitted exercise instant.
        expiry: TimestampNs,
    },
}

impl ExerciseSchedule {
    /// Check nonnegative time bounds and schedule ordering.
    pub fn validate(&self) -> CogResult<()> {
        match self {
            Self::European { expiry } => require_nonnegative_time(*expiry, "european expiry"),
            Self::American {
                first_exercise,
                expiry,
            } => {
                require_nonnegative_time(*first_exercise, "american first exercise")?;
                require_nonnegative_time(*expiry, "american expiry")?;
                if first_exercise > expiry {
                    return Err(CogError::InvalidDefinition(
                        "american first exercise is after expiry".to_owned(),
                    ));
                }
                Ok(())
            }
            Self::Bermudan { opportunities } => {
                if opportunities.is_empty() {
                    return Err(CogError::InvalidDefinition(
                        "bermudan schedule has no opportunities".to_owned(),
                    ));
                }
                for (index, time) in opportunities.iter().enumerate() {
                    require_nonnegative_time(*time, "bermudan opportunity")?;
                    if index > 0 && opportunities[index - 1] >= *time {
                        return Err(CogError::InvalidDefinition(
                            "bermudan opportunities must be strictly increasing".to_owned(),
                        ));
                    }
                }
                Ok(())
            }
        }
    }

    /// Return the final exercise opportunity.
    pub fn expiry(&self) -> TimestampNs {
        match self {
            Self::European { expiry } | Self::American { expiry, .. } => *expiry,
            Self::Bermudan { opportunities } => *opportunities
                .last()
                .expect("validated Bermudan schedule is non-empty"),
        }
    }

    /// Return whether `time` belongs to the declared opportunity set.
    pub fn permits_exercise(&self, time: TimestampNs) -> bool {
        match self {
            Self::European { expiry } => time == *expiry,
            Self::American {
                first_exercise,
                expiry,
            } => *first_exercise <= time && time <= *expiry,
            Self::Bermudan { opportunities } => opportunities.binary_search(&time).is_ok(),
        }
    }
}

/// Exact, versioned definition of one listed option contract.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OptionContract {
    /// Must equal [`CONTRACT_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Stable exact-contract identity.
    pub contract_id: ContractId,
    /// Asset delivered or referenced by the option.
    pub underlying: UnderlyingId,
    /// Currency of strike, premium, and cash settlement.
    pub currency: Currency,
    /// Call or put right.
    pub right: OptionRight,
    /// Positive strike in currency micro-units per underlying unit.
    pub strike: PriceMicros,
    /// Positive underlying units represented by one contract.
    pub multiplier: u32,
    /// Explicit exercise opportunity set.
    pub exercise: ExerciseSchedule,
    /// Cash or physical settlement convention.
    pub settlement: Settlement,
}

impl OptionContract {
    /// Construct and validate an exact contract definition.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        contract_id: impl Into<String>,
        underlying: impl Into<String>,
        currency: impl Into<String>,
        right: OptionRight,
        strike: PriceMicros,
        multiplier: u32,
        exercise: ExerciseSchedule,
        settlement: Settlement,
    ) -> CogResult<Self> {
        let contract = Self {
            schema_version: CONTRACT_SCHEMA_VERSION.to_owned(),
            contract_id: ContractId(contract_id.into()),
            underlying: UnderlyingId(underlying.into()),
            currency: Currency(currency.into()),
            right,
            strike,
            multiplier,
            exercise,
            settlement,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Check schema, identifiers, strike, multiplier, and exercise schedule.
    pub fn validate(&self) -> CogResult<()> {
        if self.schema_version != CONTRACT_SCHEMA_VERSION {
            return Err(CogError::InvalidDefinition(format!(
                "unsupported contract schema {}",
                self.schema_version
            )));
        }
        require_text(&self.contract_id.0, "contract_id")?;
        require_text(&self.underlying.0, "underlying")?;
        require_text(&self.currency.0, "currency")?;
        if self.strike.0 <= 0 {
            return Err(CogError::InvalidDefinition(
                "strike must be positive".to_owned(),
            ));
        }
        if self.multiplier == 0 {
            return Err(CogError::InvalidDefinition(
                "multiplier must be positive".to_owned(),
            ));
        }
        self.exercise.validate()
    }

    /// Return the final exercise opportunity.
    pub fn expiry(&self) -> TimestampNs {
        self.exercise.expiry()
    }

    /// Return nonnegative intrinsic value per underlying unit.
    pub fn intrinsic_per_unit(&self, spot: PriceMicros) -> i64 {
        match self.right {
            OptionRight::Call => spot.0.saturating_sub(self.strike.0).max(0),
            OptionRight::Put => self.strike.0.saturating_sub(spot.0).max(0),
        }
    }
}

/// Signed quantity of one exact option contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    /// Exact listed contract.
    pub contract: OptionContract,
    /// Nonzero contracts; positive is long and negative is short.
    pub quantity: i64,
}

impl Position {
    /// Validate the contract and require a nonzero signed quantity.
    pub fn new(contract: OptionContract, quantity: i64) -> CogResult<Self> {
        contract.validate()?;
        if quantity == 0 {
            return Err(CogError::InvalidDefinition(
                "position quantity must be non-zero".to_owned(),
            ));
        }
        Ok(Self { contract, quantity })
    }
}

fn require_text(value: &str, name: &str) -> CogResult<()> {
    if value.trim().is_empty() {
        return Err(CogError::InvalidDefinition(format!(
            "{name} must not be empty"
        )));
    }
    Ok(())
}

fn require_nonnegative_time(value: TimestampNs, name: &str) -> CogResult<()> {
    if value.0 < 0 {
        return Err(CogError::InvalidDefinition(format!(
            "{name} must be non-negative"
        )));
    }
    Ok(())
}
