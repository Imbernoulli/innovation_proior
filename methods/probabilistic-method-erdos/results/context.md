## Quantitative Ramsey Question

For each `k`, let `f(k,k)` be the least number of vertices forcing every graph to contain either a `K_k` or an independent set of size `k`. Ramsey's finite theorem guarantees that this number exists. It does not by itself explain its size. The quantitative problem splits into two directions: upper bounds show when order is unavoidable, while lower bounds require proving that a large graph can avoid both kinds of ordered `k`-set.

## Existing Upper-Bound Machinery

The available pre-1947 quantitative tool is the Erdos-Szekeres inductive proof. Fix one vertex in a two-coloring, split the remaining vertices by the color of the incident edge, and recurse. This gives `f(k,l) <= f(k-1,l)+f(k,l-1)` and hence `f(k,l) <= binom(k+l-2,k-1)`, so in particular `f(k,k) < 4^(k-1)`. The argument explains how large structures force order, but it moves in the upper-bound direction.

## Lower-Bound Obstruction

To prove `f(k,k) > n`, one must certify that some `n`-vertex graph has no `K_k` and no independent `k`-set. A direct route asks for an explicit graph and then verifies every forbidden substructure is absent. Small examples exist for this verification style.

## Counting Ingredients

There are `2^binom(n,2)` labeled graphs on `n` vertices. For a fixed `k`-set, requiring it to span a clique forces `binom(k,2)` edges and leaves all other edges free, so exactly `2^(binom(n,2)-binom(k,2))` graphs contain that particular clique. The complement map is a bijection from graphs to graphs, and it turns cliques in the complement into independent sets in the original graph.

## Evaluation Yardstick

The result is judged by how large an `n` can be certified while still avoiding all forbidden `k`-sets, measured asymptotically against the upper scale `binom(2k-2,k-1) < 4^(k-1)`.
