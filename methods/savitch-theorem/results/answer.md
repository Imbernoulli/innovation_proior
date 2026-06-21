# Savitch-Style Space Simulation

For every space-constructible `S(n) >= log n`,

`NSPACE(S(n)) subseteq DSPACE(S(n)^2)`.

Let `M` be a nondeterministic machine using `S(n)` space on input `x`. Its configurations form an implicit directed graph `G_{M,x}`. A configuration has `O(S(n))` bits, so `G_{M,x}` has at most `N = 2^{O(S(n))}` vertices. Acceptance is reachability from the start configuration to an accepting configuration.

Use bounded reachability:

```text
REACH(u, v, i):
    if i = 0:
        return (u = v) or (u -> v is one legal transition)
    for each configuration z:
        if REACH(u, z, i - 1) and REACH(z, v, i - 1):
            return true
    return false
```

The recurrence is correct because any path of length at most `2^i` has a middle configuration `z` splitting it into two paths of length at most `2^{i-1}`, and two such paths concatenate.

Start with `i = ceil(log N)`, since repeated configurations can be removed from any accepting walk. Each recursion frame stores only `u`, `v`, the current `z`, and counters, using `O(S(n))` space. The recursion depth is `O(log N) = O(S(n))`, and recursive workspace is reused between calls. Total deterministic space is therefore `O(S(n)^2)`.

The proof's point is not exhaustive DFS. It turns nondeterministic branching into implicit configuration-graph reachability and replaces a long accepting branch by recursively guessed middle configurations. The price is potentially enormous time; the gain is square space. This yields `PSPACE = NPSPACE` and `NL subseteq DSPACE(log^2 n)`.

