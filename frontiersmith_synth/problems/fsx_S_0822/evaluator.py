#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0822 -- "The Gene You Can Lose"
(family: evodevo-grn-knockout; format B, quality-metric).

THEME.  A body plan grows along a linear axis of L positions.  Each position
reads its own location via a "positional morphogen": the binary digits of its
index p, x_0..x_{T-1} (x_t = bit t of p, LSB first), where L = 2**T.  The
candidate supplies a small GENE REGULATORY NETWORK (GRN): G genes, each with a
linear-threshold regulation rule that reads the positional morphogen AND the
other genes' current expression, iterated for a fixed number of developmental
rounds.  The genes' FINAL expression pattern (a length-G bit vector) is decoded
into a cell/tissue type via a lookup table the candidate also supplies.  This
produces one "phenotype": a type for every position along the body.

The phenotype is scored against a fixed TARGET body plan -- but not only in the
wild type (WT, full genotype).  The evaluator also re-develops the SAME body
under every possible SINGLE-GENE KNOCKOUT (that gene's expression forced to 0
in every round, every position) and scores the resulting phenotype too.  The
final objective is the MEAN match fraction over {WT} union {knockout(g) : all
g}.

INNOVATION HOOK.  A network that assigns exactly one gene to one positional
bit (the "obvious" minimal encoding: G=T, gene i mirrors bit i) reconstructs
the target EXACTLY in the wild type -- but losing any single gene erases that
bit for every position, silently re-mapping roughly half the body to a
DIFFERENT position's identity.  A network with built-in regulatory redundancy
(e.g. duplicating each positional channel onto two genes, a "paralog") can
survive the loss of any one gene without corrupting the position code at all,
because its sibling still reports the bit.  This is developmental canalization:
robustness is bought with redundant regulation, not with a tighter fit.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
      {"name": str, "L": L, "T": T, "K": K, "G_max": Gmax, "iters": I,
       "target": [t_0..t_{L-1}]}    # t_p in [0,K)
  stdout: ONE JSON object:
      {"G": G, "Win": GxT ints, "W": GxG ints, "bias": G ints,
       "decode": (2**G)-length list of ints in [0,K)}
  Win/W entries must lie in [-3,3]; bias entries in [-8,8]; 1<=G<=G_max.
  Any shape/range/type violation, a crash, a timeout, or non-JSON output
  makes that instance score 0.0.

DEVELOPMENT (deterministic; run in the PARENT, not by the candidate).
  For position p, x_t = (p >> t) & 1 for t in [0,T).  State s starts at all
  zeros.  For `iters` synchronous rounds:
      raw_i  = bias[i] + sum_t Win[i][t]*x[t] + sum_j W[i][j]*s[j]
      s'_i   = 0 if gene i is knocked out, else (1 if raw_i > 0 else 0)
  After `iters` rounds, code c = sum_i s_i * 2**i (gene 0 = LSB), and the
  position's type is decode[c].

SCORING (deterministic; no wall-time).  Let obj = mean, over the wild type and
every single-gene knockout, of the fraction of positions whose type matches
the target.  Let base = the frequency of the most common type in the target
(the score a network that ignores position entirely, and just always guesses
the mode, would get -- computed by the evaluator itself, never by the
candidate).  Normalize with a fixed-headroom affine anchor:
    r = clamp( 0.1 + 0.9 * (obj - base) / (IDEAL - base), 0, 1 )     IDEAL=1.2
Matching the mode-guess baseline scores ~0.1; a perfect, fully knockout-proof
network (obj=1.0) still lands near 0.85, leaving headroom above every
reference solution.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The target and
the internal mode-frequency baseline live only in this parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

IDEAL = 1.2


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- target generation ----------------------------
def _gen_target(seed, L, K):
    """Deterministic blocky 'body plan': a handful of contiguous segments,
    each independently assigned a random type in [0,K)."""
    ni = _rng(seed)
    max_segs = max(3, min(L, 2 * K))
    S = ni(3, max_segs)
    cuts = sorted(set(ni(1, L - 1) for _ in range(S - 1))) if L > 1 else []
    bounds = [0] + cuts + [L]
    target = []
    for i in range(len(bounds) - 1):
        t = ni(0, K - 1)
        target += [t] * (bounds[i + 1] - bounds[i])
    return target


def _build_instances():
    """Deterministic instance family. (seed, T, K)."""
    specs = [
        (101, 3, 4),
        (102, 3, 5),
        (103, 4, 5),
        (104, 4, 6),
        (105, 4, 4),
        (106, 5, 6),
        (107, 5, 8),
        (108, 4, 7),
        # harder / larger held-out instances
        (209, 5, 5),
        (210, 5, 9),
    ]
    out = []
    for seed, T, K in specs:
        L = 1 << T
        target = _gen_target(seed, L, K)
        Gmax = min(12, 2 * T + 2)
        out.append({"name": f"grn{seed}", "L": L, "T": T, "K": K,
                    "G_max": Gmax, "iters": 3, "target": target})
    return out


# ----------------------------- development -----------------------------
def _develop(ans, T, L, G, ko):
    """Return the phenotype (length-L list of types) with gene `ko` knocked
    out (ko=None -> wild type)."""
    Win = ans["Win"]; W = ans["W"]; bias = ans["bias"]; decode = ans["decode"]
    iters = ans["_iters"]
    ndec = len(decode)
    out = []
    for p in range(L):
        x = [(p >> t) & 1 for t in range(T)]
        s = [0] * G
        for _ in range(iters):
            new_s = [0] * G
            for i in range(G):
                if i == ko:
                    new_s[i] = 0
                    continue
                raw = bias[i]
                Wi = Win[i]
                for t in range(T):
                    raw += Wi[t] * x[t]
                Wr = W[i]
                for j in range(G):
                    raw += Wr[j] * s[j]
                new_s[i] = 1 if raw > 0 else 0
            s = new_s
        c = 0
        for i in range(G):
            if s[i]:
                c |= (1 << i)
        if c >= ndec:
            c = ndec - 1
        out.append(decode[c])
    return out


# ----------------------------- validation ----------------------------------
def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _validate(ans, T, K, Gmax):
    if not isinstance(ans, dict):
        return None
    G = ans.get("G")
    if not _is_int(G) or not (1 <= G <= Gmax):
        return None
    Win = ans.get("Win"); W = ans.get("W"); bias = ans.get("bias"); decode = ans.get("decode")
    if not (isinstance(Win, list) and len(Win) == G):
        return None
    for row in Win:
        if not (isinstance(row, list) and len(row) == T):
            return None
        for v in row:
            if not _is_int(v) or not (-3 <= v <= 3):
                return None
    if not (isinstance(W, list) and len(W) == G):
        return None
    for row in W:
        if not (isinstance(row, list) and len(row) == G):
            return None
        for v in row:
            if not _is_int(v) or not (-3 <= v <= 3):
                return None
    if not (isinstance(bias, list) and len(bias) == G):
        return None
    for v in bias:
        if not _is_int(v) or not (-8 <= v <= 8):
            return None
    ndec = 1 << G
    if not (isinstance(decode, list) and len(decode) == ndec):
        return None
    for v in decode:
        if not _is_int(v) or not (0 <= v < K):
            return None
    return {"G": G, "Win": Win, "W": W, "bias": bias, "decode": decode}


def score(inst, answer):
    T = inst["T"]; K = inst["K"]; L = inst["L"]; Gmax = inst["G_max"]
    ans = _validate(answer, T, K, Gmax)
    if ans is None:
        return False, 0.0
    ans["_iters"] = inst["iters"]
    G = ans["G"]
    target = inst["target"]

    def match(phenotype):
        n_ok = sum(1 for a, b in zip(phenotype, target) if a == b)
        return n_ok / L

    scores = [match(_develop(ans, T, L, G, None))]
    for g in range(G):
        scores.append(match(_develop(ans, T, L, G, g)))
    obj = sum(scores) / len(scores)
    return True, obj


def baseline(inst):
    target = inst["target"]; K = inst["K"]; L = inst["L"]
    counts = [0] * K
    for t in target:
        counts[t] += 1
    return max(counts) / L


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {"name": inst["name"], "L": inst["L"], "T": inst["T"],
                  "K": inst["K"], "G_max": inst["G_max"], "iters": inst["iters"],
                  "target": list(inst["target"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        denom = IDEAL - b
        if denom < 1e-9:
            denom = 1e-9
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
