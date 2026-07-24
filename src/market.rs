//! Versioned market observations with explicit price authority.

use serde::{Deserialize, Serialize};

use crate::contract::{ContractId, PriceMicros, TimestampNs};
use crate::hash::canonical_hash;
use crate::{CogError, CogResult};

/// Schema identifier serialized with [`OptionDataSourceManifest`].
pub const MARKET_SOURCE_SCHEMA_VERSION: &str = "volt.market_source.v1";
/// Schema identifier serialized with [`OptionMarketObservation`].
pub const MARKET_OBSERVATION_SCHEMA_VERSION: &str = "volt.market_observation.v1";

/// Whether observations came from recorded evidence or a generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationOrigin {
    /// Previously recorded market evidence.
    Historical,
    /// Programmatically generated fixture or scenario evidence.
    Generated,
}

/// Strongest price claim a source is allowed to make.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriceAuthority {
    /// Trade and aggregate prices can be used as model inputs or explicit
    /// proxy marks, but not represented as executable bid/ask prices.
    ProxyOnly,
    /// The source contains top-of-book observations. Individual observations
    /// still determine whether both usable sides are present.
    TopOfBook,
}

/// Identity, provenance, and assumptions for a stream of observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionDataSourceManifest {
    /// Must equal [`MARKET_SOURCE_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Stable identifier repeated by every observation in the stream.
    pub source_id: String,
    /// Historical or generated provenance.
    pub origin: ObservationOrigin,
    /// Human-readable dataset identity.
    pub dataset: String,
    /// Immutable source-data or generator version.
    pub data_version: String,
    /// Maximum execution claim permitted for prices in this source.
    pub price_authority: PriceAuthority,
    /// Explicit qualifications a consumer should retain with results.
    pub assumptions: Vec<String>,
}

impl OptionDataSourceManifest {
    /// Construct and validate a market-source manifest.
    pub fn new(
        source_id: impl Into<String>,
        origin: ObservationOrigin,
        dataset: impl Into<String>,
        data_version: impl Into<String>,
        price_authority: PriceAuthority,
        assumptions: Vec<String>,
    ) -> CogResult<Self> {
        let manifest = Self {
            schema_version: MARKET_SOURCE_SCHEMA_VERSION.to_owned(),
            source_id: source_id.into(),
            origin,
            dataset: dataset.into(),
            data_version: data_version.into(),
            price_authority,
            assumptions,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Check schema identity, required text, and assumption entries.
    pub fn validate(&self) -> CogResult<()> {
        if self.schema_version != MARKET_SOURCE_SCHEMA_VERSION {
            return Err(CogError::InvalidDefinition(format!(
                "unsupported market source schema {}",
                self.schema_version
            )));
        }
        for (name, value) in [
            ("source_id", self.source_id.as_str()),
            ("dataset", self.dataset.as_str()),
            ("data_version", self.data_version.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CogError::InvalidDefinition(format!(
                    "market source {name} must not be empty"
                )));
            }
        }
        if self
            .assumptions
            .iter()
            .any(|assumption| assumption.trim().is_empty())
        {
            return Err(CogError::InvalidDefinition(
                "market source assumptions must not contain empty values".to_owned(),
            ));
        }
        Ok(())
    }

    /// Hash the manifest with its schema domain.
    pub fn semantic_hash(&self) -> CogResult<String> {
        canonical_hash(MARKET_SOURCE_SCHEMA_VERSION, self)
    }
}

/// Aggregation interval of an option bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BarInterval {
    /// One trading or calendar day as defined by the source.
    Day,
    /// One minute as defined by the source.
    Minute,
}

/// Aggregate OHLCV observation for one option contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionBar {
    /// Declared aggregation interval.
    pub interval: BarInterval,
    /// Opening price in currency micro-units.
    pub open: PriceMicros,
    /// Highest price in the interval.
    pub high: PriceMicros,
    /// Lowest price in the interval.
    pub low: PriceMicros,
    /// Closing price in the interval.
    pub close: PriceMicros,
    /// Source-reported contract volume.
    pub volume: u64,
    /// Source-reported transaction count.
    pub transactions: u64,
}

impl OptionBar {
    fn validate(&self) -> CogResult<()> {
        let prices = [self.open.0, self.high.0, self.low.0, self.close.0];
        if prices.iter().any(|price| *price < 0) {
            return Err(CogError::InvalidDefinition(
                "option bar prices must be non-negative".to_owned(),
            ));
        }
        if self.high.0 < self.low.0
            || self.high.0 < self.open.0
            || self.high.0 < self.close.0
            || self.low.0 > self.open.0
            || self.low.0 > self.close.0
        {
            return Err(CogError::InvalidDefinition(
                "option bar OHLC values are inconsistent".to_owned(),
            ));
        }
        Ok(())
    }
}

/// One observed option trade.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionTrade {
    /// Positive execution price in currency micro-units.
    pub price: PriceMicros,
    /// Positive executed contract quantity.
    pub size: u64,
    /// Optional source-specific venue code.
    pub exchange: Option<u16>,
    /// Source-specific trade condition codes.
    pub conditions: Vec<u32>,
    /// Source-specific correction indicator.
    pub correction: u16,
}

impl OptionTrade {
    fn validate(&self) -> CogResult<()> {
        if self.price.0 <= 0 || self.size == 0 {
            return Err(CogError::InvalidDefinition(
                "option trade price and size must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

/// One side of a top-of-book quote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookLevel {
    /// Nonnegative quoted price in currency micro-units.
    pub price: PriceMicros,
    /// Quoted contract quantity; zero makes the level unusable.
    pub size: u64,
    /// Optional source-specific venue code.
    pub exchange: Option<u16>,
}

impl BookLevel {
    fn validate(&self) -> CogResult<()> {
        if self.price.0 < 0 {
            return Err(CogError::InvalidDefinition(
                "option quote price must be non-negative".to_owned(),
            ));
        }
        Ok(())
    }

    fn is_usable(&self) -> bool {
        self.price.0 > 0 && self.size > 0
    }
}

/// Optional bid and ask observations for one contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionQuote {
    /// Best observed bid, when present.
    pub bid: Option<BookLevel>,
    /// Best observed ask, when present.
    pub ask: Option<BookLevel>,
}

impl OptionQuote {
    fn validate(&self) -> CogResult<()> {
        if self.bid.is_none() && self.ask.is_none() {
            return Err(CogError::InvalidDefinition(
                "option quote requires at least one side".to_owned(),
            ));
        }
        if let Some(bid) = &self.bid {
            bid.validate()?;
        }
        if let Some(ask) = &self.ask {
            ask.validate()?;
        }
        if let (Some(bid), Some(ask)) = (&self.bid, &self.ask) {
            if bid.is_usable() && ask.is_usable() && bid.price > ask.price {
                return Err(CogError::InvalidDefinition(
                    "usable option quote is crossed".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// Typed market payload carried by an [`OptionMarketObservation`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionObservationPayload {
    /// Aggregate bar, suitable only for proxy marking.
    Bar(OptionBar),
    /// Last trade, suitable only for proxy marking.
    Trade(OptionTrade),
    /// Top-of-book quote, potentially suitable for executable sides.
    Quote(OptionQuote),
}

impl OptionObservationPayload {
    fn validate(&self) -> CogResult<()> {
        match self {
            Self::Bar(bar) => bar.validate(),
            Self::Trade(trade) => trade.validate(),
            Self::Quote(quote) => quote.validate(),
        }
    }

    fn sort_rank(&self) -> u8 {
        match self {
            Self::Quote(_) => 0,
            Self::Trade(_) => 1,
            Self::Bar(_) => 2,
        }
    }
}

/// One versioned, contract-resolved observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionMarketObservation {
    /// Must equal [`MARKET_OBSERVATION_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Identifier of the matching source manifest.
    pub source_id: String,
    /// Source-local sequence used for stable ordering at equal timestamps.
    pub source_sequence: u64,
    /// Resolved exact listed contract.
    pub contract_id: ContractId,
    /// Original source symbol retained as evidence.
    pub vendor_symbol: String,
    /// Source event time in Unix-epoch nanoseconds.
    pub event_time: TimestampNs,
    /// Bar, trade, or quote content.
    pub payload: OptionObservationPayload,
}

impl OptionMarketObservation {
    /// Construct and validate one resolved observation.
    pub fn new(
        source_id: impl Into<String>,
        source_sequence: u64,
        contract_id: ContractId,
        vendor_symbol: impl Into<String>,
        event_time: TimestampNs,
        payload: OptionObservationPayload,
    ) -> CogResult<Self> {
        let observation = Self {
            schema_version: MARKET_OBSERVATION_SCHEMA_VERSION.to_owned(),
            source_id: source_id.into(),
            source_sequence,
            contract_id,
            vendor_symbol: vendor_symbol.into(),
            event_time,
            payload,
        };
        observation.validate()?;
        Ok(observation)
    }

    /// Check schema, identifiers, time, and payload consistency.
    pub fn validate(&self) -> CogResult<()> {
        if self.schema_version != MARKET_OBSERVATION_SCHEMA_VERSION {
            return Err(CogError::InvalidDefinition(format!(
                "unsupported market observation schema {}",
                self.schema_version
            )));
        }
        if self.source_id.trim().is_empty()
            || self.contract_id.0.trim().is_empty()
            || self.vendor_symbol.trim().is_empty()
        {
            return Err(CogError::InvalidDefinition(
                "market observation identifiers must not be empty".to_owned(),
            ));
        }
        if self.event_time.0 < 0 {
            return Err(CogError::InvalidDefinition(
                "market observation event time must be non-negative".to_owned(),
            ));
        }
        self.payload.validate()
    }

    /// Derive a clearly labeled proxy mark when the payload supports one.
    pub fn proxy_mark(&self) -> Option<OptionMark> {
        match &self.payload {
            OptionObservationPayload::Bar(bar) if bar.close.0 > 0 => Some(OptionMark {
                price: bar.close,
                basis: MarkBasis::AggregateClose,
            }),
            OptionObservationPayload::Trade(trade) => Some(OptionMark {
                price: trade.price,
                basis: MarkBasis::LastTrade,
            }),
            OptionObservationPayload::Quote(quote) => {
                let bid = quote.bid.as_ref().filter(|level| level.is_usable())?;
                let ask = quote.ask.as_ref().filter(|level| level.is_usable())?;
                let total = i128::from(bid.price.0) + i128::from(ask.price.0);
                let midpoint = i64::try_from(total / 2).ok()?;
                Some(OptionMark {
                    price: PriceMicros(midpoint),
                    basis: MarkBasis::QuoteMidpoint,
                })
            }
            OptionObservationPayload::Bar(_) => None,
        }
    }

    /// Return the quote side required to execute a buy or sell.
    ///
    /// Bars, trades, and unusable quote sides return `None`.
    pub fn executable_price(&self, side: OrderSide) -> Option<PriceMicros> {
        let OptionObservationPayload::Quote(quote) = &self.payload else {
            return None;
        };
        match side {
            OrderSide::Buy => quote
                .ask
                .as_ref()
                .filter(|level| level.is_usable())
                .map(|level| level.price),
            OrderSide::Sell => quote
                .bid
                .as_ref()
                .filter(|level| level.is_usable())
                .map(|level| level.price),
        }
    }
}

/// Evidence used to derive a non-executable option mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarkBasis {
    /// Closing value of an aggregate bar.
    AggregateClose,
    /// Most recent observed trade.
    LastTrade,
    /// Arithmetic midpoint of usable bid and ask.
    QuoteMidpoint,
}

/// Proxy price paired with the evidence basis that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionMark {
    /// Derived price in currency micro-units.
    pub price: PriceMicros,
    /// Observation type used to derive the price.
    pub basis: MarkBasis,
}

/// Economic side of a requested option transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderSide {
    /// Acquire contracts and therefore require an ask.
    Buy,
    /// Dispose of contracts and therefore require a bid.
    Sell,
}

/// Common streaming boundary for generated and historical observations.
///
/// Contract resolution is deliberately separate: observations use a
/// `contract_id` that must have been joined to a point-in-time reference
/// snapshot before simulation.
pub trait OptionMarketDataSource {
    /// Return source identity and price-authority metadata.
    fn manifest(&self) -> &OptionDataSourceManifest;
    /// Return the next ordered observation, or `None` at end of stream.
    fn next_observation(&mut self) -> CogResult<Option<OptionMarketObservation>>;
}

/// Repeatable in-memory implementation shared by generated and historical data.
#[derive(Debug, Clone)]
pub struct InMemoryOptionMarketDataSource {
    manifest: OptionDataSourceManifest,
    observations: Vec<OptionMarketObservation>,
    next_index: usize,
}

impl InMemoryOptionMarketDataSource {
    /// Validate, source-check, and deterministically order observations.
    pub fn new(
        manifest: OptionDataSourceManifest,
        mut observations: Vec<OptionMarketObservation>,
    ) -> CogResult<Self> {
        manifest.validate()?;
        for observation in &observations {
            observation.validate()?;
            if observation.source_id != manifest.source_id {
                return Err(CogError::InvalidDefinition(format!(
                    "observation source {} does not match manifest {}",
                    observation.source_id, manifest.source_id
                )));
            }
        }
        observations.sort_by(|left, right| {
            (
                left.event_time,
                left.source_sequence,
                &left.contract_id,
                left.payload.sort_rank(),
            )
                .cmp(&(
                    right.event_time,
                    right.source_sequence,
                    &right.contract_id,
                    right.payload.sort_rank(),
                ))
        });
        Ok(Self {
            manifest,
            observations,
            next_index: 0,
        })
    }

    /// Rewind the stream to its first observation.
    pub fn reset(&mut self) {
        self.next_index = 0;
    }

    /// Return the number of observations in the stream.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Return whether the stream contains no observations.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}

impl OptionMarketDataSource for InMemoryOptionMarketDataSource {
    fn manifest(&self) -> &OptionDataSourceManifest {
        &self.manifest
    }

    fn next_observation(&mut self) -> CogResult<Option<OptionMarketObservation>> {
        let Some(observation) = self.observations.get(self.next_index) else {
            return Ok(None);
        };
        self.next_index += 1;
        Ok(Some(observation.clone()))
    }
}
