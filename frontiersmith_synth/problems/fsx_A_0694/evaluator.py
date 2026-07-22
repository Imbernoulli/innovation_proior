#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0694 -- "Apprentice's Montage: Interference-Aware Skill
Curriculum" (family: interference-aware-curriculum; format B, quality-metric).

THEME.  An apprentice must master K skills before the final trial.  You plan a
length-T practice montage: at each of the T time steps you drill ONE skill.
Drilling skill j does two things simultaneously:
  (1) DIMINISHING RETURNS on j itself: its proficiency p_j (in [0,1)) closes a
      fraction gain_j of the remaining gap to mastery:  p_j <- p_j + gain_j*(1-p_j).
      Early drills help a lot; a near-mastered skill barely moves -- over-drilling
      a saturated skill wastes montage time.
  (2) INTERFERENCE on every OTHER skill i: p_i <- p_i * interfere[j][i], where
      interfere[j][i] <= 1.  Skills in the same "reinforcement clique" barely decay
      each other (interfere close to 1); skills that are antagonists decay each
      other hard every time either is drilled.  This happens on EVERY step, not
      just when a skill is neglected -- so a skill's rivals keep eroding it even
      while it is not being trained.
The apprentice only passes if the WORST of the K skills is good, so the score is
the MAXIMIN final proficiency: min_i p_i(T).  A skill's fate is decided by two
things: how many times it was drilled (its post-drill peak, from diminishing
returns) and how much interference battered it AFTER its last drill -- so the very
last time you touch a skill is when you "lock in" it against everything that comes
after.  This makes montage planning a graph-partitioning problem in time: block
mutually-reinforcing skills together, keep antagonists apart, and save each skill's
final refresher for as late as its planted vulnerability demands.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "K": int,                     # number of skills
             "T": int,                     # montage length
             "p0": [K floats in [0,1)],    # starting proficiency
             "gain": [K floats in (0,1)],  # diminishing-returns rate per skill
             "interfere": [[K floats]]}    # KxK; interfere[j][i] in (0,1] is the
                                            # multiplier applied to skill i when
                                            # skill j is drilled (i != j); diagonal
                                            # entries are 1.0 and unused.
  stdout: ONE JSON object:
            {"sequence": [a_0, ..., a_{T-1}]}   # EXACTLY T ints, each in [0, K)
          -- the skill drilled at every time step, in order.

  A sequence is VALID iff it is a list of exactly T integers, each in [0, K).
  Repeats are required and expected (T > K always). Wrong length, an out-of-range
  or non-integer entry, a crash, a timeout, or non-JSON output -> that instance
  scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references,
all from the SAME simulation rule the candidate is scored under:
    y_base = maximin score of "grind only the initially-weakest skill for the
             whole montage" (repeat the single index argmin(p0) T times). This is
             the evaluator's own weak baseline construction.
    y_ideal= a loose, generally-unreachable upper reference: split the T slots as
             evenly as possible across the K skills (K disjoint dedicated blocks,
             no cross-skill decay at all -- i.e. pretend interference away) and
             take the maximin of THAT.  Real interference only makes life harder,
             so this stays a comfortable ceiling; strong solvers do not reach it.
    y_cand = maximin score of the candidate's own sequence, under the real
             simulation (including interference).
  normalized with an affine anchor (weak baseline -> 0.1, loose ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (y_cand - y_base) / max(1e-9, y_ideal - y_base), 0, 1 )
  Reproducing the weak "grind-the-weakest" baseline scores ~0.1; doing worse scores
  0 (clamped); spreading and sequencing drills well scores higher, capped at 1.0
  but never actually reaching it because y_ideal ignores interference entirely.
  Final score is the mean of r over all instances (varied K/T/clique-shape, some
  with tight budgets, some with cliques interleaved across skill indices).

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance above (which already
contains every number needed to plan well -- p0, gain, interfere are all public).
Every reference (baseline, ideal) and all validation happen in THIS parent
process, so a frame-walking / introspecting candidate learns nothing extra.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    box = [state]

    def nxt_float():
        box[0] = (box[0] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((box[0] >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return nxt_float


# ----------------------------- instance family -----------------------------
def _build_one(seed, K, T, n_cliques, contiguous, tight):
    """Deterministic instance: K skills, T-slot montage, `n_cliques` reinforcement
    cliques (contiguous index blocks or interleaved by index), planted antagonism
    across cliques.  `tight` shrinks the budget to stress the maximin water-fill."""
    nf = _rng(seed)
    p0 = [round(0.05 + 0.30 * nf(), 6) for _ in range(K)]
    gain = [round(0.15 + 0.45 * nf(), 6) for _ in range(K)]
    if contiguous:
        clique_of = [(i * n_cliques) // K for i in range(K)]
    else:
        clique_of = [i % n_cliques for i in range(K)]
    interfere = [[1.0] * K for _ in range(K)]
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            if clique_of[i] == clique_of[j]:
                interfere[j][i] = round(0.985 + 0.014 * nf(), 6)   # near-neutral
            else:
                interfere[j][i] = round(0.86 + 0.09 * nf(), 6)     # antagonist
    if tight:
        T = int(T * 0.78)
    return {"name": f"apprentice{seed}", "K": K, "T": T, "p0": p0, "gain": gain,
            "interfere": interfere}


def _build_instances():
    # (seed, K, T, n_cliques, contiguous_index_blocks, tight_budget)
    specs = [
        (101, 4, 18, 2, True,  False),
        (102, 5, 22, 2, True,  False),
        (103, 5, 22, 2, False, False),
        (104, 6, 26, 3, True,  False),
        (105, 6, 26, 3, False, False),
        (106, 6, 18, 3, False, True),
        (107, 7, 30, 3, True,  False),
        (108, 7, 30, 3, False, False),
        (211, 7, 22, 3, False, True),   # held-out: interleaved cliques + tight budget
        (212, 5, 20, 2, False, True),   # held-out: interleaved cliques + tight budget
    ]
    return [_build_one(*s) for s in specs]


# ----------------------------- simulation / references ---------------------
def _simulate(inst, seq):
    K = inst["K"]
    p = list(inst["p0"])
    gain = inst["gain"]; interfere = inst["interfere"]
    for j in seq:
        gj = gain[j]
        row = interfere[j]
        for i in range(K):
            if i == j:
                p[i] = p[i] + gj * (1.0 - p[i])
            else:
                p[i] = p[i] * row[i]
    return p


def _baseline(inst):
    j0 = min(range(inst["K"]), key=lambda i: inst["p0"][i])
    return min(_simulate(inst, [j0] * inst["T"]))


def _ideal(inst):
    K = inst["K"]; T = inst["T"]; p0 = inst["p0"]; gain = inst["gain"]
    base = T // K; rem = T - base * K
    vals = []
    for i in range(K):
        n = base + (1 if i < rem else 0)
        p = p0[i]
        for _ in range(n):
            p = p + gain[i] * (1.0 - p)
        vals.append(p)
    return min(vals)


def _validate(inst, answer):
    if not isinstance(answer, dict):
        return None
    seq = answer.get("sequence")
    if not isinstance(seq, list):
        return None
    T = inst["T"]; K = inst["K"]
    if len(seq) != T:
        return None
    out = []
    for x in seq:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if x < 0 or x >= K:
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
        y_base = _baseline(inst)
        y_ideal = _ideal(inst)
        denom = y_ideal - y_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "K": inst["K"], "T": inst["T"],
                  "p0": list(inst["p0"]), "gain": list(inst["gain"]),
                  "interfere": [list(row) for row in inst["interfere"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=8)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            seq = _validate(inst, ans)
        except Exception:
            seq = None
        if seq is None:
            vec.append(0.0)
            continue
        y_cand = min(_simulate(inst, seq))
        r = 0.1 + 0.9 * (y_cand - y_base) / denom
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
