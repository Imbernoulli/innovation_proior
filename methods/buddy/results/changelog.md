# Changelog — buddy

## 2026-08-18 — svfix(D_candidate): ground the sketch-propagation recurrence, self-loop fix
`results/reasoning.md`'s decisive step (the min/max sketch-propagation recurrence that replaces
per-edge subgraph construction with per-node message passing) stated the neighborhood decomposition
as N_d(u) = ∪_{v ∈ N(u)} N_{d−1}(v) — an *open*-neighborhood recurrence — as if it were exact. It
isn't: N_d(u) is defined earlier in the same file as the *closed* ball (distance ≤ d, which includes
u itself at d = 0), and the correct decomposition requires the closed neighborhood N(u) ∪ {u}, i.e.
the min/max propagation must run on the graph with a self-loop added at every node. Without the
self-loop the recurrence silently drops each node's own identity from its own ball at every hop; this
is checkable on the smallest possible counterexample, two nodes joined by one edge, where the
open-neighbor recurrence estimates their 1-hop intersection as 0 when the true value is 2 (both nodes
lie in both closed 1-hop balls).

Found via the paper's official repo, github.com/melifluos/subgraph-sketching, issue #7 ("Is a node
its own neighbour?"): the paper's corresponding author, Ben Chamberlain, replies with this exact
reason ("you must add self loops... in the trivial case of a graph that contains only 2 connected
nodes, you will think that the cardinality of their 1-hop intersection is zero") and points at
`hashing.py`'s `add_self_loops(edge_index)` call before every round of sketch propagation. Local
copies: `notes/sources.md`, `refs/github-issue7-self-loops.txt`, `src/hashing.py`.

Rewrote the reasoning.md passage to run through the obvious-but-wrong open-neighbor recurrence,
the two-node counterexample that breaks it, and the self-loop fix, before continuing into the
existing (unchanged) k-hop propagation / count-table readout / clamping-caveat material.

`results/answer.md`'s and `results/train_answer.md`'s `ElphHashes.propagate` code was already
correct (`out = sketch.clone(); out.index_reduce_(..., include_self=True)` — the self-inclusion was
already implemented), so no code change was needed. `results/train_answer.md`'s prose repeated the
same bare open-neighbor formula as reasoning.md's original text; corrected it to match, for
consistency between the two write-ups.

No change to the landing (BUDDY/ELPH as the method, or the code).
