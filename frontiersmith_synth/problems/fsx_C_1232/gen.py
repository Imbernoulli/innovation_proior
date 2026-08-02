#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance of "Three Calls Deep" (api-fuzz-schedule).

Model: a small deterministic stateful API schema.

  * T resource TYPES, labelled 0..T-1 (labels are a random permutation of the
    schema's true topological role -- do NOT assume low label = few
    dependencies).
  * Each type t has a dependency list deps[t] (other type labels): to
    CREATE a resource of type t, at least one resource of EVERY type in
    deps[t] must already have been successfully created earlier in the
    sequence. deps[] form a DAG (every instance is acyclic by construction).
  * Each type t also has its own local state machine: S_t states
    0..S_t-1 (every fresh resource starts at state 0), and a transition
    table trans[t][s][c] -> next state, for opcodes c in 0..C-1. Every table
    in this generator is a MONOTONE CHAIN: state index == the minimum number
    of correct ops needed to reach it from state 0 (wrong opcodes never
    move you backward, only fail to advance), so "reach state k" and "this
    took >= k calls, in the right order" are the same fact.

Three flavours of type (never labelled as such in the input -- must be
inferred from deps[]/tables):
  * SCAFFOLD type (1 per instance): S_t=1 (no state beyond 0 -- ops on it
    are inert). It has no dependency itself and gates every DEEP type
    below. Zero direct reward, but mandatory to touch first.
  * SHALLOW types (4 per instance): no dependencies, S_t=2, and ANY opcode
    advances state 0 -> 1. Trivial to find with one blind op. Held at a
    constant count so the checker's dependency-oblivious baseline B (which
    always secures every zero-dependency type) stays a stable, sizeable
    anchor across the whole ladder.
  * DEEP types (2 per instance): depend on the single scaffold type,
    S_t=4, and need an EXACT length-3 opcode sequence (planted per type,
    independently random) to walk 0->1->2->3; any other opcode at a given
    state self-loops (no progress, no penalty).

Trap (matches the brief's example almost verbatim): endpoint-uniform random
sampling ignores deps[] entirely, so a good chunk of its CREATE attempts
target a type before its one prerequisite is up, and even once a deep
resource does get created, matching its exact 3-opcode sequence by chance
is a (1/C)^3 event per blind attempt. It reliably finds the 4 shallow
("depth-1") reward-types and essentially never fully clears a depth-3 deep
type within budget. This holds on every one of the 10 tests, worsening as
the opcode alphabet C grows (2 -> 4) and the budget is cut tighter
relative to a full clear.

Determinism: all randomness from random.Random(testId * 2654435761 + 1232).
"""
import sys
import random

# idx (0-based): chain_len, n_shallow, n_deep, C(#opcodes), budget, for
# testId 1..10.
#
# chain_len and n_shallow/n_deep are held CONSTANT across the whole ladder
# on purpose: n_shallow=4 keeps the checker's dependency-oblivious baseline
# B (which always secures every zero-dependency type) a stable, sizeable
# anchor, so that even a full clear of both depth-3 chains stays a bounded
# multiple of B instead of drifting off and saturating the score as the
# schema grows.  Difficulty instead grows via a wider opcode alphabet C
# (harder to stumble onto the right 3-call sequence by chance) and a
# shrinking, explicitly-tuned budget (forces real prioritization even for
# an exact planner).  Values were calibrated empirically against
# trivial/greedy/strong reference solutions to hit the acceptance band
# (strong-greedy >= 0.06, strong <= 0.92, greedy-trivial >= 0.03).
PARAMS = [
    (1, 4, 2, 2, 17),
    (2, 4, 2, 2, 16),
    (3, 4, 2, 2, 16),
    (4, 4, 2, 3, 16),
    (5, 4, 2, 3, 15),
    (6, 4, 2, 3, 15),
    (7, 4, 2, 3, 14),
    (8, 4, 2, 4, 15),
    (9, 4, 2, 4, 14),
    (10, 4, 2, 4, 13),   # largest opcode alphabet & tightest budget
]


def build_types(rng, chain_len, n_shallow, n_deep, C):
    """Return list of (deps_logical, S_t, table) in LOGICAL order:
    [scaffold_0..scaffold_{chain_len-1}, shallow_0.., deep_0..].
    deps_logical uses logical indices (0-based, this list's own order)."""
    types = []

    # scaffolds: linear chain, S=1, inert table (1 row, C self-loops to 0)
    for j in range(chain_len):
        deps = [j - 1] if j > 0 else []
        table = [[0] * C]
        types.append({"deps": deps, "S": 1, "table": table, "kind": "scaffold"})

    scaffold_last = chain_len - 1 if chain_len > 0 else None

    # shallow: no deps, S=2, any opcode 0->1, terminal self-loop
    for _ in range(n_shallow):
        table = [[1] * C, [1] * C]
        types.append({"deps": [], "S": 2, "table": table, "kind": "shallow"})

    # deep: depends on last scaffold, S=4, exact length-3 required sequence
    for _ in range(n_deep):
        req = [rng.randrange(C) for _ in range(3)]
        row0 = [(1 if c == req[0] else 0) for c in range(C)]
        row1 = [(2 if c == req[1] else 1) for c in range(C)]
        row2 = [(3 if c == req[2] else 2) for c in range(C)]
        row3 = [3] * C
        deps = [scaffold_last] if scaffold_last is not None else []
        types.append({"deps": deps, "S": 4, "table": [row0, row1, row2, row3],
                       "kind": "deep", "req": req})

    return types


def gen(test_id):
    rng = random.Random(test_id * 2654435761 + 1232)
    idx = (test_id - 1) % len(PARAMS)
    _tid, n_shallow, n_deep, C, budget = PARAMS[idx]
    chain_len = 1

    types = build_types(rng, chain_len, n_shallow, n_deep, C)
    T = len(types)

    # random relabelling: logical index i -> shuffled label perm[i]
    perm = list(range(T))
    rng.shuffle(perm)
    # inv[label] = logical index that owns this label
    inv = [0] * T
    for i, lab in enumerate(perm):
        inv[lab] = i

    # emit types in LABEL order (0..T-1); each type's deps translated to labels
    out_types = [None] * T
    for i, ty in enumerate(types):
        lab = perm[i]
        deps_labels = [perm[d] for d in ty["deps"]]
        out_types[lab] = (deps_labels, ty["S"], ty["table"])

    return T, C, budget, out_types


def main():
    test_id = int(sys.argv[1])
    T, C, budget, out_types = gen(test_id)
    lines = [f"{T} {C} {budget}"]
    for deps_labels, S, table in out_types:
        lines.append(f"{len(deps_labels)} " + " ".join(map(str, deps_labels)))
        lines.append(str(S))
        for row in table:
            lines.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
