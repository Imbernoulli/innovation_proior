#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0790 -- "The Quiet Chain: Weighted Constraints with a Hidden
Backbone" (family: backbone-guided-weighted-sat; format B, quality-metric; theme: maximize
satisfied constraint weight in a weighted CSP with a planted backbone hidden among free
variables).

THEME.  You are given `n` boolean variables (1-indexed 1..n) and a weighted list of OR
clauses (each clause a short list of signed integer literals; positive = variable true,
negative = variable false). You output ONE full assignment. The score is the total weight
of satisfied clauses, normalized against a static per-variable baseline and a generous
unreachable upper bound.

UNIT-PROPAGATION-BACKBONE-PROBE (mechanism 1, the innovation hook).  A small subset of the
variables form a hidden BACKBONE: their value is forced by a handful of very heavily
weighted clauses -- some are unit clauses (a single literal, pinning one variable directly),
the rest are 2-literal implication clauses that chain forced values from those "root" units
across the rest of the backbone (standard 2-SAT-style propagation: from `a=matches` and
clause `(-a_lit, b_lit)`, `b` is forced too). Running unit propagation to closure on ONLY the
heaviest clauses recovers the *entire* backbone assignment exactly -- and it is always the
uniquely weight-maximizing choice for those clauses, since they vastly outweigh everything
else touching those variables.

FOCUSED-FLIP-WALKSAT + TABU-RESTART (mechanisms 2 & 3).  Once the backbone is frozen, the
remaining FREE variables interact through many lighter, noisier clauses -- classic local-
search territory. A focused WalkSAT (flip only the free set, greedy break/make scoring with
occasional random noise to escape plateaus) combined with a short tabu list (don't
immediately undo a recent flip) and periodic restarts of the free assignment converges well.

THE TRAP.  Many of those lighter clauses are built to be satisfied more easily when a
backbone variable sits at the OPPOSITE of its forced value -- a constant, cheap pull away
from the backbone's true assignment. A generic WalkSAT that treats every variable the same
(no probing, no freezing) spends almost all of its flips chasing the huge population of
light clauses and only rarely touches the handful of heavy ones; on instances where that
light-clause pull is strong, it drifts the backbone away from its heavy-weight-optimal value
and never fully recovers -- losing a large, fixed chunk of weight that the propagation-first
approach picks up for free.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": int,
             "clauses": [[lit, lit, ...], ...],   # signed ints, 1-indexed variables
             "weights": [w, w, ...]}               # same length as clauses, w > 0
  stdout: ONE JSON object:
            {"assign": [a_1, ..., a_n]}   # each a_i in {0, 1} (int, not bool)
  VALID iff "assign" is a list of exactly n values, each literally 0 or 1 (no bool, no
  float, no NaN). Any violation, crash, timeout, or non-JSON -> 0.0 on that instance.

SCORING (deterministic; no wall-time).  Let `b = baseline(inst)` be the objective of a
static per-variable majority heuristic (for each variable, sum the weight of clauses where
it appears positively vs negatively and take the side with more weight -- computed by THIS
evaluator, independent of any candidate) and `hi(inst)` a generous unreachable upper bound
(the sum of ALL clause weights, times a small slack factor -- clause density guarantees this
is never fully attainable):
    r = clamp( 0.1 + 0.9 * (obj - b) / (hi - b), 0, 1 )
The final score is the mean of r over 10 fixed seeded instances.

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance. The objective and the
normalization are computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json
import isorun

GAIN = 1.03  # small slack on the unreachable upper bound


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


def _pick_distinct(rng, lo, hi, k):
    """k distinct ints in [lo, hi) via rejection sampling (deterministic)."""
    out = []
    seen = set()
    guard = 0
    while len(out) < k and guard < 10000:
        guard += 1
        v = rng(lo, hi - 1)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ----------------------------- instance construction -------------------------
HEAVY_WEIGHT = 40
LIGHT_LO, LIGHT_HI = 1, 4


def _lit_matches(var0, val):
    """Signed 1-indexed literal that is TRUE iff var0 (0-indexed) == val."""
    return (var0 + 1) if val == 1 else -(var0 + 1)


def _build_backbone(rng, n_root, chain_vars):
    """Return (pv, heavy_clauses) -- planted values for ALL backbone vars (roots + chain)
    and the heavy unit/implication clauses that force them via propagation."""
    n_back = n_root + len(chain_vars)
    pv = {}
    for v in range(n_back):
        pv[v] = rng(0, 1)

    heavy = []  # list of (clause, weight)
    heads = list(range(n_root))
    for h in heads:
        heavy.append(([_lit_matches(h, pv[h])], HEAVY_WEIGHT))

    perm = list(chain_vars)
    # Fisher-Yates shuffle with our own rng
    for i in range(len(perm) - 1, 0, -1):
        j = rng(0, i)
        perm[i], perm[j] = perm[j], perm[i]

    chains = [[] for _ in range(n_root)]
    for i, v in enumerate(perm):
        chains[i % n_root].append(v)

    for i, ch in enumerate(chains):
        prev = heads[i]
        for cur in ch:
            lit_prev = _lit_matches(prev, pv[prev])
            lit_cur = _lit_matches(cur, pv[cur])
            heavy.append(([-lit_prev, lit_cur], HEAVY_WEIGHT))
            prev = cur

    return pv, heavy


def _make_instance(name, seed, n_free, n_root, chain_len, decoys_per_back, noise_mult):
    rng = _rng(seed)
    n_back = n_root + chain_len
    n = n_back + n_free
    chain_vars = list(range(n_root, n_back))

    pv, heavy = _build_backbone(rng, n_root, chain_vars)

    clauses = []
    weights = []
    for c, w in heavy:
        clauses.append(c)
        weights.append(w)

    # decoy clauses: pull each backbone var toward its WRONG value, softened by
    # two random free-variable literals
    for b in range(n_back):
        for _ in range(decoys_per_back):
            wrong_lit = -_lit_matches(b, pv[b])
            fv = _pick_distinct(rng, n_back, n, 2)
            if len(fv) < 2:
                continue
            lits = [wrong_lit]
            for v in fv:
                sign = 1 if rng(0, 1) == 1 else -1
                lits.append(sign * (v + 1))
            clauses.append(lits)
            weights.append(rng(LIGHT_LO, LIGHT_HI))

    # pure free-variable noise clauses (the genuine local-search territory)
    n_noise = noise_mult * max(n_free, 1)
    for _ in range(n_noise):
        fv = _pick_distinct(rng, n_back, n, 3)
        if len(fv) < 3:
            continue
        lits = []
        for v in fv:
            sign = 1 if rng(0, 1) == 1 else -1
            lits.append(sign * (v + 1))
        clauses.append(lits)
        weights.append(rng(LIGHT_LO, LIGHT_HI))

    return {"name": name, "n": n, "clauses": clauses, "weights": weights,
            "n_back": n_back, "pv": pv}


def _build_instances():
    specs = [
        # (name, seed, n_free, n_root, chain_len, decoys_per_back, noise_mult)
        ("calm_small",   9101, 30, 2, 6,  1, 4),
        ("trap_small",   9102, 30, 2, 8,  6, 4),
        ("calm_medium",  9103, 50, 2, 12, 1, 4),
        ("trap_medium1", 9104, 55, 3, 13, 7, 4),
        ("trap_medium2", 9105, 60, 2, 16, 8, 4),
        ("calm_large",   9106, 70, 3, 17, 1, 4),
        ("trap_large1",  9107, 70, 3, 19, 8, 4),
        ("trap_large2",  9108, 75, 3, 21, 9, 4),
        ("held_out_wide", 9109, 90, 3, 13, 7, 4),
        ("held_out_deep", 9110, 60, 2, 28, 7, 4),
    ]
    return [_make_instance(*s) for s in specs]


# ----------------------------- scoring ---------------------------------------
def _clause_satisfied(clause, assign):
    for lit in clause:
        v = abs(lit) - 1
        if (lit > 0 and assign[v] == 1) or (lit < 0 and assign[v] == 0):
            return True
    return False


def objective(inst, assign):
    total = 0.0
    for c, w in zip(inst["clauses"], inst["weights"]):
        if _clause_satisfied(c, assign):
            total += w
    return total


def baseline(inst):
    n = inst["n"]
    pos = [0.0] * n
    neg = [0.0] * n
    for c, w in zip(inst["clauses"], inst["weights"]):
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos[v] += w
            else:
                neg[v] += w
    assign = [1 if pos[v] >= neg[v] else 0 for v in range(n)]
    return objective(inst, assign)


def _hi(inst):
    return GAIN * sum(inst["weights"])


def _public(inst):
    return {"name": inst["name"], "n": inst["n"],
            "clauses": [list(c) for c in inst["clauses"]],
            "weights": list(inst["weights"])}


def _valid_answer(inst, answer):
    if not isinstance(answer, dict):
        return None
    n = inst["n"]
    a = answer.get("assign")
    if not isinstance(a, list) or len(a) != n:
        return None
    out = []
    for x in a:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if x != 0 and x != 1:
            return None
        out.append(x)
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        b = baseline(inst)
        h = _hi(inst)
        denom = h - b
        if denom <= 1e-9:
            denom = 1e-9
        ans, st = isorun.run_candidate(cand, _public(inst), timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            assign = _valid_answer(inst, ans)
            if assign is None:
                vec.append(0.0)
                continue
            obj = objective(inst, assign)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (obj - b) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
