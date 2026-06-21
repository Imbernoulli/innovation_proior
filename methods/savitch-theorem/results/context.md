# Context

## Space-Bounded Nondeterminism

A nondeterministic work-tape machine can use `S(n)` cells and accept when at least one computation branch reaches an accepting state. Along a single branch it stores only the current local situation: finite control, tape contents within the space bound, head positions, and the input-head position. The difficulty appears only when a deterministic machine tries to cover all possible branches without knowing which branch, if any, accepts.

The comparison should be in work space, not in time. Exponential time is an acceptable price if the simulator still uses little memory. The question is whether removing nondeterministic choices necessarily forces the deterministic simulator to store an exponentially large search object.

## Configuration Objects

For a fixed machine on an input of length `n`, a configuration is a finite encoding of everything needed to continue the computation. If `S(n) >= log n`, one configuration has `O(S(n))` bits: the bounded work-tape contents use `O(S(n))`, work-head positions are smaller, and the input-head position fits in `O(log n)`.

The set of all legal configurations is finite. There may be exponentially many of them in `S(n)`, but each individual configuration is small. A legal one-step transition can be checked locally from the machine description, the input symbol, and the two candidate configurations.

## Graph View

The computation can be represented as a directed graph whose vertices are configurations and whose edges are legal machine steps. The start configuration is a source, and accepting configurations are targets. A nondeterministic accepting computation exists exactly when some accepting vertex is reachable from the source.

This graph is normally implicit. Building it would waste the advantage that configurations are small. The useful access pattern is to enumerate candidate configuration encodings one at a time and test local adjacency, while never materializing the full graph.

## Failed Deterministic Simulations

A breadth-first simulation stores a frontier of possible configurations, which can be exponentially large. A depth-first simulation stores a long path and still needs a way to avoid cycling. An explicit visited set stores the graph's global history. A transitive-closure table stores reachability facts for many pairs of configurations.

All of these baselines spend space on global search state. They solve the wrong storage problem: they try to remember too much of the computation instead of exploiting the small representation of one configuration.

## Target Frame

The desired simulator must decide whether a target configuration is reachable from the start configuration in the implicit finite graph. It may revisit subproblems and spend very large time. It must count only reusable work space and should express its bound as a function of the original space `S(n)`.

The available ingredients are a small encoding for one configuration, a way to enumerate configurations, a local adjacency test, and a finite bound on useful path length obtained by deleting repeated configurations from any accepting walk.

