# Answer

The theorem is:

`NSPACE(s(n)) = co-NSPACE(s(n))` for every `s(n) >= log n`.

The key method is inductive counting, or in Szelepcsenyi's phrase, forced enumeration. For a directed graph, let `R_i` be the vertices reachable from `s` in at most `i` steps and let `r_i = |R_i|`. If `r_i` is known, nondeterminism can certify that a vertex is not in `R_i` by enumerating the reachable vertices in a fixed order, with valid path witnesses and no repeated labels, and accepting only when the count is exactly `r_i` and the target never appears.

Then compute `r_{i+1}` from `r_i`: scan every vertex `v`; prove `v in R_{i+1}` by a path to `v` or to a predecessor of `v`, and prove `v notin R_{i+1}` by the same ordered counted enumeration of `R_i`, showing that no listed vertex is `v` or points to `v`. Increment the next count exactly once for each vertex certified reachable.

After computing `r_{n-1}`, use the nonmembership test for `t`. This accepts exactly when no path from `s` to `t` exists. Since directed reachability is NL-complete, `NL = coNL`.

For general space `s(n)`, replace graph vertices by configurations of the nondeterministic machine and iterate the counts until they stabilize. A configuration uses `O(s(n))` bits, and counters up to the number of configurations fit in `O(s(n))` space. The same ordered counted enumeration accepts exactly when no accepting configuration is reachable.

This bypasses the apparent existential/universal asymmetry: nondeterminism still guesses only positive path witnesses, but the exact count forces those witnesses to form a complete enumeration. It also bypasses Savitch's deterministic midpoint recursion, which proves a squared-space simulation rather than same-space complement closure.
