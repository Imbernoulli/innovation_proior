# The Probabilistic Method

The method is to prove existence by putting a probability distribution on candidate objects and showing the desired outcome has positive probability. In the Ramsey lower-bound case, color the edges of `K_n` red/blue uniformly at random. A fixed `k`-set is monochromatic with probability `2^(1-binom(k,2))`, so by the union bound

`Pr[some monochromatic K_k] <= binom(n,k)2^(1-binom(k,2))`.

If this is below `1`, then some coloring has no monochromatic `K_k`, hence `R(k,k) > n`. Erdos's 1947 proof states the same idea as a count over all labeled graphs: fewer than half contain a `K_k`, fewer than half contain an independent `k`-set by complementation, so at least one graph avoids both. This gives `f(k,k) > 2^(k/2)` for `k >= 3`.

The distinctive move is not a recombination of earlier Ramsey machinery. Ramsey and Erdos-Szekeres prove that order is forced in every sufficiently large object; explicit lower-bound attempts try to build one exceptional object. Erdos instead reasons over the entire ensemble of graphs and proves the union of all bad cases has less than full measure. The witness is certified statistically, not constructed.

Source anchors: Erdos 1947 for the original count; Ramsey 1930 for the finite threshold problem; Erdos-Szekeres 1935 for the binomial upper bound; Alon-Spencer and Zhao for the modern positive-probability and alteration formulations. No direct Erdos first-person discovery account was found in the retrieved sources.
