## Space-Bounded Nondeterminism

Space complexity measures how many work-tape cells a machine uses, not how many steps it takes. A nondeterministic machine using logarithmic space can still run for polynomially many steps on a finite configuration graph, because each configuration has only logarithmically many bits.

For a general bound `s(n) >= log n`, a configuration of an `s(n)`-space machine can be written in `O(s(n))` space. This lets questions about machines be rephrased as questions about reachability among configurations.

## Reachability as a Complete Problem

Directed reachability asks whether a path leads from a source vertex to a target vertex. It has the right shape for nondeterminism: guess the next vertex repeatedly and verify each edge locally.

The same problem captures all nondeterministic logspace computations. Given a logspace machine and an input, form the graph whose vertices are configurations and whose edges are legal one-step moves. Acceptance becomes reachability from the start configuration to an accepting configuration.

## Complements

The complement asks for absence of a path. A path proves reachability by exhibiting an existential witness. The complement instead concerns the entire finite search space of branches and reachable configurations.

Nondeterminism naturally supplies a local positive witness: a single guessed-and-verified path. The complement question is about the structure of the whole reachable set.

## The Deterministic Detour

There is a deterministic way to simulate space-bounded nondeterminism by recursive reachability. To decide whether one configuration reaches another within a bounded number of steps, try every possible midpoint and recurse on the two halves.

This midpoint recursion (Savitch) uses a squared space bound. It places nondeterministic space within deterministic space of squared size, and is stated for the simulation question rather than for complementation.

## The Same-Space Setting

A complement procedure that stays within the original space bound works with small objects only: vertex names, counters, step bounds, and one local certificate at a time. In a graph with `n` vertices, the set of vertices reachable from a source can require `n` bits, while a logarithmic-space procedure has only `O(log n)` cells.

The question is how to characterize non-reachability for a directed graph, and more generally for the configuration graph of an `s(n)`-space nondeterministic machine, using objects of size `O(s(n))`.
