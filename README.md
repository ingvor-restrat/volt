# volt
Volt is a typed, composable representation and runtime for options portfolios. An options position is
simultaneously a set of listed contracts, a payoff shape, a collection of volatility and path risks, and a
state that evolves through exercise, assignment, expiry, and trading. Volt keeps those meanings together
in one canonical portfolio graph, preserving relationships that flat tables and disconnected services lose.

From that authoritative representation, Volt compiles consistent payoff, graph, tensor, simulation, and
evidence views with shared financial semantics and provenance. Typed contracts and lifecycle transitions
exclude invalid states, canonicalization removes equivalent constructions, and deterministic journals make
results replayable and auditable. This gives strategy-discovery, optimization, and learning systems a
stable foundation without allowing them to redefine what a portfolio means.
