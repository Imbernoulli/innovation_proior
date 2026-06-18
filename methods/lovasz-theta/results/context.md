## Channel Confusion

A zero-error channel can be represented by a graph whose vertices are signal values and whose edges mark pairs
that might be confused. A one-use code is a set of vertices with no confusable pair, so its size is governed by
the independence number `alpha(G)`.

This turns a communication question into a graph question, but only for one channel use. The real difficulty is
what happens when a message is allowed to use many positions.

## Product Growth

For length `k` words, the graph is the kth strong power `G^k`: two words are confusable when each coordinate is
equal or confusable. The asymptotic zero-error value is
`Theta(G) = lim_{k -> infinity} alpha(G^k)^(1/k)`.

The definition is compact, but the sequence it asks for is hostile. It combines rapidly growing graph products
with independence numbers, which are already difficult in one graph.

## Pentagon Barrier

The five-cycle is the first small case where one use gives too little information. A single use permits only two
unconfusable values, but a two-use code has five words. That proves the lower bound `Theta(C5) >= sqrt(5)`.

The lower bound is not the main obstacle. The hard part is proving that no longer block code can beat it.

## Upper-Bound Gap

Known clique-cover and fractional-packing bounds give general upper estimates, but they leave a gap for the
pentagon. A one-shot upper bound is not enough unless it also behaves well under the same product that builds
longer words.

The desired certificate must be checkable without enumerating all independent sets in all powers. Otherwise it
only restates the original difficulty.

## Design Pressure

The pre-method constraints are therefore sharp: keep the independent-set interpretation, produce an upper bound,
make it compatible with strong products, and avoid an exhaustive search over graph powers.

Any successful construction has to convert the discrete limit into an object whose algebra already knows how to
respect products.
