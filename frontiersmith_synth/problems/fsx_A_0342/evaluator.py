#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0342 -- "Feeder Consolidation on a Distribution Substation"
(family: online-heuristic-simulator; format B, quality-metric; theme: power grid).

THEME.  A distribution substation must energize a queue of customer *load blocks*
(integer kVA demands) onto identical distribution transformers.  Each transformer
has two independent limits:

    * a thermal CAPACITY  C  (kVA)  -- the summed demand of the blocks it carries
      must never exceed C, or the transformer overheats;
    * a breaker-CHANNEL count K    -- the transformer's protection panel has only K
      feeder breakers, so it can carry AT MOST K distinct load blocks.

A load block must be energized WHOLE on a single transformer (you cannot split one
customer's demand across two transformers).  Energizing a transformer costs one
"unit" (a tap changer + a cooling loop).  The dispatcher wants to clear the whole
queue while energizing as FEW transformers as possible.

This is 1-D bin packing with an added cardinality constraint (a.k.a. K-item bin
packing), skinned onto a power grid: load blocks = items (integer sizes),
transformer thermal limit = bin capacity C, breaker count = per-bin item cap K,
"transformers energized" = bins used, which we MINIMIZE.  The dispatcher supplies
the packing (the priority with which blocks are steered onto transformers); the
simulator just checks feasibility and counts energized transformers.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int), "channels": K (int), "n": N (int),
             "demands": [d_0, ..., d_{N-1}]   # integer kVA, 1 <= d_i <= C}
  stdout: ONE JSON object:
            {"assign": [t_0, ..., t_{N-1}]}
          where t_i >= 0 is the transformer index that load block i is energized on.
          Transformer indices need not be contiguous; a transformer "exists" iff it
          carries >=1 block, and the number of DISTINCT non-empty transformers is
          the energized count.

  A dispatch is VALID iff `assign` is a list of exactly N non-negative integers,
  no transformer's summed demand exceeds C, AND no transformer carries more than K
  blocks.  Invalid output, wrong length, an overloaded transformer, an over-full
  breaker panel, a crash, a timeout, or non-JSON  ->  that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    q_lb   = max( ceil(sum(demands)/C), ceil(N/K) )   # dual L1 lower bound (ideal)
    q_base = transformers used by the internal NEXT-FIT dispatcher (weak baseline)
    q_cand = transformers used by the candidate dispatch
  and normalize with an affine anchor (weak baseline -> 0.1, L1 ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate matching next-fit scores ~0.1; a candidate reaching the (generally
  unreachable) dual L1 bound scores 1.0; doing worse than next-fit scores < 0.1.

  Because L1 is a LOOSE lower bound, even strong offline packers (first-fit- /
  best-fit-decreasing under the cardinality cap) stay strictly below 1.0 on most
  instances  ->  headroom for open-ended improvement.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(dual L1, next-fit baseline) are computed by THIS parent process, so a
frame-walking / introspecting candidate learns nothing useful.

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

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _build_demands(seed, n, C, dist):
    """Return a list of N integer kVA demands in [1, C]. Deterministic."""
    ni = _rng(seed)
    out = []
    for _ in range(n):
        if dist == "residential":                 # many small feeder demands
            d = ni(1, max(1, C // 4))
        elif dist == "mixed":                      # broad spread over the rating
            d = ni(1, C)
        elif dist == "commercial":                 # medium demands near half rating
            d = ni(max(1, C // 4), (3 * C) // 4)
        elif dist == "industrial":                 # mostly large, hard-to-pair loads
            d = ni(max(1, (2 * C) // 5), (17 * C) // 20)
        elif dist == "bimodal":                    # lots of tiny + a few heavy loads
            d = ni(1, max(1, C // 5)) if ni(0, 99) < 60 else ni((3 * C) // 5, (9 * C) // 10)
        else:
            d = ni(1, C)
        if d < 1:
            d = 1
        if d > C:
            d = C
        out.append(d)
    return out


def _build_instances():
    """Deterministic instance family. (seed, n, C, K, dist)."""
    specs = [
        (401, 24, 20, 5, "commercial"),
        (402, 28, 20, 5, "commercial"),
        (403, 30, 24, 6, "mixed"),
        (414, 26, 20, 4, "bimodal"),
        (405, 32, 24, 5, "residential"),
        (420, 30, 18, 4, "commercial"),
        (407, 32, 20, 6, "mixed"),
        (408, 28, 24, 5, "industrial"),
        # harder / larger held-out instances
        (511, 45, 22, 5, "commercial"),
        (510, 44, 20, 4, "bimodal"),
        (512, 48, 24, 6, "residential"),
        (513, 52, 20, 5, "industrial"),
    ]
    out = []
    for seed, n, C, K, dist in specs:
        demands = _build_demands(seed, n, C, dist)
        out.append({"name": f"sub{seed}", "capacity": C, "channels": K,
                    "n": n, "demands": demands, "dist": dist})
    return out


# ----------------------------- references ----------------------------------
def _l1(demands, C, K):
    """Dual L1 lower bound: the thermal bound AND the breaker-channel bound."""
    n = len(demands)
    thermal = -(-sum(demands) // C)       # ceil(sum / C)
    channel = -(-n // K)                  # ceil(n / K)
    return max(thermal, channel)


def _next_fit(demands, C, K):
    """Weak online dispatcher: keep loading the current transformer; open a new
    one the moment a block does not fit thermally OR the breaker panel is full.
    Never looks back."""
    bins = 1
    rem = C
    cnt = 0
    for d in demands:
        if d <= rem and cnt < K:
            rem -= d
            cnt += 1
        else:
            bins += 1
            rem = C - d
            cnt = 1
    return bins


# ----------------------------- validation ----------------------------------
def _energized(inst, answer):
    """Validate answer against the instance. Return energized-transformer count
    or None if infeasible / malformed."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    demands = inst["demands"]
    C = inst["capacity"]
    K = inst["channels"]
    N = inst["n"]
    if len(assign) != N:
        return None
    load = {}
    cnt = {}
    for i, t in enumerate(assign):
        if isinstance(t, bool) or not isinstance(t, int):
            return None
        if t < 0:
            return None
        load[t] = load.get(t, 0) + demands[i]
        cnt[t] = cnt.get(t, 0) + 1
        if load[t] > C:
            return None
        if cnt[t] > K:
            return None
    return len(load)          # number of distinct energized transformers


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        C = inst["capacity"]
        K = inst["channels"]
        demands = inst["demands"]
        q_lb = _l1(demands, C, K)
        q_base = _next_fit(demands, C, K)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": C, "channels": K,
                  "n": inst["n"], "demands": list(demands)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _energized(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_base - q_cand) / denom
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
