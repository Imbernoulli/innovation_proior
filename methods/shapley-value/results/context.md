## Research question

A coalition of agents can create value together, and the value of a coalition need not be the
sum of the values of its members acting alone. The problem is to assign credit to each agent
from the total value of the grand coalition in a way that is principled rather than ad hoc.

The setting is a transferable-utility cooperative game. There is a finite player set `N`, and
a characteristic function `v` assigns a real number `v(S)` to every coalition `S subseteq N`,
with `v(emptyset) = 0`. The number `v(S)` is the worth that coalition `S` can secure. The
desired object is a payoff vector `phi(v) in R^N`, one number per player, interpreted as each
player's credit for the collective worth `v(N)`.

A satisfactory solution has to satisfy four demands at once. It should allocate all available
value, treat players with identical roles identically, give no credit to a player who changes
no coalition's worth, and behave linearly when two independent payoff descriptions are added.
The challenge is that a player's contribution depends on which collaborators are already
present, so there is no single obvious marginal contribution to use.

## Background

The characteristic-function form abstracts away bargaining protocol and records only what
each coalition can achieve. This makes the credit-assignment question purely structural:
given all coalition worths, what payoff vector is justified by the game itself?

A player's contribution depends on context: if `i` joins `S`, the incremental contribution is

```text
v(S union {i}) - v(S),
```

where `S` does not contain `i`, and different `S` can give different increments.

The natural fairness constraints are these. **Efficiency** requires
`sum_i phi_i(v) = v(N)`, so no value is lost or manufactured. **Symmetry** requires equal
payoffs for players whose substitutions leave every coalition worth unchanged.
**Dummy** requires `phi_i(v) = v({i})` when player `i` adds exactly `v({i})` to every
coalition, and in particular zero when `i` adds nothing. **Additivity** requires
`phi(v + w) = phi(v) + phi(w)`, so the value assigned to a sum of games is the sum of the
values assigned to the games.

These axioms are strong but not computationally exotic. They are statements about the map
from characteristic functions to payoff vectors, and they apply before any bargaining,
auction, or algorithmic implementation is chosen.

## Baselines

**Equal split.** Divide `v(N)` equally among all players. This satisfies efficiency and treats
all names the same regardless of role.

**Standalone-value split.** Pay each player according to `v({i})`, possibly normalized to sum
to `v(N)`. This uses individual productivity as the basis for credit.

**Leave-one-out contribution.** Pay player `i` according to `v(N) - v(N \ {i})`. This bases
credit on the player's marginal contribution to the grand coalition specifically.

**Chosen-order marginal accounting.** Pick an order of arrival and give each player the value
added when they arrive. The marginal increments telescope to `v(N)`, making this rule efficient
for any chosen order.

## Evaluation settings

The main checks are axiomatic and algebraic rather than empirical. A candidate payoff rule
should be tested on small finite games where all coalition values can be enumerated: pure
dummy players, symmetric players, games with complementarity, games with substitutability,
and sums of games.

The strongest tests are exact identities: total payoff equals `v(N)`; symmetric players get
equal payoff; dummy players get only their standalone contribution; and applying the rule to
`v + w` gives the vector sum of applying it to `v` and to `w`. A complete solution should also
show uniqueness, not merely propose one rule that happens to pass examples.

## Code framework

For this theory artifact, the scaffold is a finite characteristic-function representation
and a neutral slot for a payoff rule.

```text
Inputs:
  finite player set N
  characteristic function v(S) for every coalition S subseteq N
  v(emptyset) = 0

Payoff-rule skeleton:
  for each player i in N:
    inspect how coalition worths change when i is included
    assign a real payoff phi_i(v)

Required checks:
  sum_i phi_i(v) = v(N)
  equivalent players receive equal payoff
  players who add no extra value receive no extra credit
  phi(v + w) = phi(v) + phi(w)
```
