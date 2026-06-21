## Channel Confusion

A zero-error channel can be represented by a graph whose vertices are signal values and whose edges mark pairs
that might be confused. A one-use code is a set of vertices with no confusable pair, so its size is governed by
the independence number `alpha(G)`.

This turns a communication question into a graph question, but only for one channel use.

## Product Growth

For length `k` words, the graph is the kth strong power `G^k`: two words are confusable when each coordinate is
equal or confusable. The asymptotic zero-error value is
`Theta(G) = lim_{k -> infinity} alpha(G^k)^(1/k)`.

## Pentagon

The five-cycle is the first small case where one use gives too little information. A single use permits only two
unconfusable values, but a two-use code has five words. That proves the lower bound `Theta(C5) >= sqrt(5)`.

## Upper Bounds

Known clique-cover and fractional-packing bounds give general upper estimates for the zero-error capacity.
The question is how to obtain an upper bound on `Theta(G)` that is compatible with the strong-product structure.
