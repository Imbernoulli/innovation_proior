## Space-Bounded Nondeterminism

Space complexity measures how many work-tape cells a machine uses, not how many steps it takes. A nondeterministic machine using logarithmic space can still run for polynomially many steps on a finite configuration graph, because each configuration has only logarithmically many bits.

For a general bound `s(n) >= log n`, a configuration of an `s(n)`-space machine can be written in `O(s(n))` space. This lets questions about machines be rephrased as questions about reachability among configurations.

## Reachability as a Complete Problem

Directed reachability asks whether a path leads from a source vertex to a target vertex. It has the right shape for nondeterminism: guess the next vertex repeatedly and verify each edge locally.

The same problem captures all nondeterministic logspace computations. Given a logspace machine and an input, form the graph whose vertices are configurations and whose edges are legal one-step moves. Acceptance becomes reachability from the start configuration to an accepting configuration.

## Complements Look Different

The complement asks for absence of a path. The usual existential witness has disappeared: a path proves reachability, but a missing path seems to demand checking every possible path, every branch, or every reachable configuration.

This is the core asymmetry. Nondeterminism naturally supplies a local positive witness. It does not naturally supply a local witness that an entire finite search space has been exhausted.

## The Deterministic Detour

There is a deterministic way to simulate space-bounded nondeterminism by recursive reachability. To decide whether one configuration reaches another within a bounded number of steps, try every possible midpoint and recurse on the two halves.

This midpoint recursion is powerful, but it spends a squared space bound. It shows that nondeterministic space is not wildly stronger than deterministic space, yet it does not answer whether complementation can be done in the original nondeterministic space.

## The Same-Space Demand

A same-space complement procedure cannot store the reachable set. In a graph with `n` vertices, even the set of vertices reachable from a source can require `n` bits, far beyond logarithmic space.

Any proof at this scale must keep only small objects: vertex names, counters, step bounds, and one local certificate at a time. The open pressure is how such small objects could certify that no reachable accepting configuration has been missed.
