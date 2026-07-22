#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0841 -- "Hot-Key Repair: Bounded Migration under Power-of-Two
Choices" (family: skew-aware-shard-assignment; format B, quality-metric; theme: key-value
shard rebalancing).

THEME.  A key-value store has already placed `N` keys onto `S` shards (`shard0[i]` is
key `i`'s CURRENT shard).  Each key carries a real read/write **weight** `weight[i]`
(its load contribution).  Keys are listed in CREATION order (id 0 = oldest); a small
handful of the most RECENTLY created keys ("viral" content) carry most of the weight --
a classic Zipfian **key-skew** tail sitting on top of a large, boring, near-uniform
majority.  Because the initial placement never looked at weight, the few heavy keys can
be co-located on the same shard(s), overloading them relative to the rest.

BOUNDED-MIGRATION-REPAIR (mechanism 1).  You may REPAIR the placement, but migration is
expensive: at most `budget` keys total may end up on a shard different from `shard0[i]`
(a hard cap on the *count* of moved keys, independent of how much weight they carry).

POWER-OF-TWO-CHOICES (mechanism 2).  A moved key cannot go anywhere: for every key `i`
the instance precomputes exactly two alternative candidate shards `alt[i] = [a, b]`
(`a != b`, both `!= shard0[i]`) -- the system's two consistent-hash choices for that key.
If you move key `i`, its final shard must be `shard0[i]`, `alt[i][0]`, or `alt[i][1]`.

KEY-SKEW-DIAGNOSIS (mechanism 3, the innovation hook).  The budget is far smaller than
`N`, so it must be spent where it matters: on the few keys whose weight actually drives
the max-loaded shard, using their power-of-two choices to route them off the hot shard --
leaving the light majority exactly where it is.  A policy that spends its migration
budget on keys in an arbitrary (e.g. creation-id) order, rather than diagnosing which
keys are heavy, burns the whole budget relocating boring low-weight keys before it ever
reaches the viral ones sitting at the end of the id order.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "S": int, "N": int,
             "shard0": [s_0 ... s_{N-1}],      # current shard of key i, 0<=s_i<S
             "weight": [w_0 ... w_{N-1}],      # key weights, w_i > 0
             "alt":    [[a_i,b_i] ...],        # the two power-of-two alternative shards
             "budget": int}                    # max number of keys allowed to move
  stdout: ONE JSON object:
            {"assign": [f_0 ... f_{N-1}]}      # final shard of every key

  VALID iff `assign` is a list of exactly N integers (no bool/float/NaN/inf), each
  `f_i` in `{shard0[i], alt[i][0], alt[i][1]}`, and the COUNT of i with `f_i !=
  shard0[i]` is <= budget.  Any violation, crash, timeout, or non-JSON -> 0.0 on that
  instance.

SCORING (deterministic; no wall-time).  Per instance, load(assign, s) = sum of weight[i]
over keys assigned to shard s; M(assign) = max_s load(assign, s) (LOWER is better).
    M_base = M(shard0)                        -- do-nothing reference
    LB     = total_weight / S                 -- unreachable perfectly-balanced ideal
    r = clamp( 0.1 + 0.9 * (M_base - M(assign)) / (M_base - LB), 0, 1 )
  Doing nothing scores exactly 0.1. Because moves are capped in COUNT (not weight) and
  restricted to two precomputed candidates each, the ideal balanced bound LB is never
  reachable, so even an optimal repair keeps r < 1. The final score is the mean of r
  over 10 fixed seeded instances.

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance above (identical to
the internal instance -- there is no separate hidden half here, scoring only needs what
the candidate already reads).

CLI:  python3 evaluator.py <solution.py>
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


# ----------------------------- scoring primitives ---------------------------
def loads_of(inst, assign):
    S = inst["S"]
    load = [0.0] * S
    for i, s in enumerate(assign):
        load[s] += inst["weight"][i]
    return load


def makespan(inst, assign):
    return max(loads_of(inst, assign))


# ----------------------------- instance family -----------------------------
def _base_instance(seed, S, N, hot_shards, heavy_count_per_hot, heavy_lo, heavy_hi,
                    light_lo, light_hi, budget):
    """
    Build one instance.  Key ids 0..N-1 are CREATION order.  The last
    `len(hot_shards)*heavy_count_per_hot` ids are the heavy ("viral") keys, appended in
    id order round-robin across `hot_shards`; every one of them starts pinned to its hot
    shard (deterministic overload, not a probabilistic hash collision).  All earlier ids
    are light keys, whose current shard is drawn near-uniformly at random -- so a naive
    id-order pass sees plenty of ordinary-looking rebalancing opportunities among them
    long before it ever reaches the heavy tail at the end of the list.
    """
    ni = _rng(seed)
    heavy_ids = []
    for k in range(heavy_count_per_hot):
        for hs in hot_shards:
            heavy_ids.append(hs)          # placeholder; real id assigned below in order
    heavy_n = len(heavy_ids)
    light_n = N - heavy_n

    shard0 = [0] * N
    weight = [0.0] * N
    for i in range(light_n):
        shard0[i] = ni(0, S - 1)
        weight[i] = float(ni(light_lo, light_hi))
    for k in range(heavy_n):
        i = light_n + k
        shard0[i] = heavy_ids[k]
        weight[i] = float(ni(heavy_lo, heavy_hi))

    alt = []
    for i in range(N):
        a = ni(0, S - 1)
        while a == shard0[i]:
            a = ni(0, S - 1)
        b = ni(0, S - 1)
        while b == shard0[i] or b == a:
            b = ni(0, S - 1)
        alt.append([a, b])

    return {"S": S, "N": N, "shard0": shard0, "weight": weight, "alt": alt,
            "budget": budget}


def _skew_trap(seed, S, N, budget):
    """Trap: ONE hot shard collects several viral keys appended at the END of the id
    list. Budget is small and consumed by ordinary early-id light-key churn under an
    id-order policy, well before it ever reaches the viral tail."""
    return _base_instance(seed, S, N, hot_shards=[0], heavy_count_per_hot=5,
                           heavy_lo=55, heavy_hi=85, light_lo=1, light_hi=4,
                           budget=budget)


def _multi_hot_trap(seed, S, N, budget):
    """Held-out, harder trap: TWO independent hot shards each collect their own viral
    tail; the repair budget must be diagnosed and split correctly across both hotspots,
    not just the single most obvious one."""
    return _base_instance(seed, S, N, hot_shards=[0, S // 2], heavy_count_per_hot=3,
                           heavy_lo=50, heavy_hi=80, light_lo=1, light_hi=4,
                           budget=budget)


def _diffuse(seed, S, N, budget):
    """Greedy-friendly: no heavy tail, weights are all close together, so an id-order
    power-of-two pass rebalances about as well as a weight-diagnosed one -- the recipe
    genuinely works here, unlike on the skewed traps."""
    ni = _rng(seed)
    shard0 = [ni(0, S - 1) for _ in range(N)]
    weight = [float(ni(5, 15)) for _ in range(N)]
    alt = []
    for i in range(N):
        a = ni(0, S - 1)
        while a == shard0[i]:
            a = ni(0, S - 1)
        b = ni(0, S - 1)
        while b == shard0[i] or b == a:
            b = ni(0, S - 1)
        alt.append([a, b])
    return {"S": S, "N": N, "shard0": shard0, "weight": weight, "alt": alt,
            "budget": budget}


def _build_instances():
    out = []
    # (name, kind, seed, S, N, budget)
    specs = [
        ("skew1", "skew", 8401, 12, 260, 9),
        ("skew2", "skew", 8402, 14, 300, 10),
        ("skew3", "skew", 8403, 10, 220, 8),
        ("skew4", "skew", 8404, 16, 340, 11),
        ("diff1", "diff", 8411, 12, 240, 14),
        ("diff2", "diff", 8412, 14, 280, 16),
        ("diff3", "diff", 8413, 10, 200, 12),
        ("multi1", "multi", 8421, 16, 320, 13),
        ("multi2", "multi", 8422, 18, 360, 14),
        ("multi3", "multi", 8423, 14, 300, 12),
    ]
    for name, kind, seed, S, N, budget in specs:
        if kind == "skew":
            inst = _skew_trap(seed, S, N, budget)
        elif kind == "multi":
            inst = _multi_hot_trap(seed, S, N, budget)
        else:
            inst = _diffuse(seed, S, N, budget)
        inst["name"] = name
        out.append(inst)
    return out


# ----------------------------- validation ----------------------------------
def _valid_assign(inst, answer):
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    N = inst["N"]
    if not isinstance(assign, list) or len(assign) != N:
        return None
    S = inst["S"]
    shard0 = inst["shard0"]
    alt = inst["alt"]
    out = []
    moves = 0
    for i, v in enumerate(assign):
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 0 or v >= S:
            return None
        allowed = (shard0[i], alt[i][0], alt[i][1])
        if v not in allowed:
            return None
        if v != shard0[i]:
            moves += 1
        out.append(v)
    if moves > inst["budget"]:
        return None
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        m_base = makespan(inst, inst["shard0"])
        lb = sum(inst["weight"]) / inst["S"]
        public = {"name": inst["name"], "S": inst["S"], "N": inst["N"],
                  "shard0": list(inst["shard0"]), "weight": list(inst["weight"]),
                  "alt": [list(x) for x in inst["alt"]], "budget": inst["budget"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            assign = _valid_assign(inst, ans)
            if assign is None:
                vec.append(0.0)
                continue
            m_cand = makespan(inst, assign)
        except Exception:
            vec.append(0.0)
            continue
        denom = m_base - lb
        if denom <= 1e-9:
            r = 1.0 if m_cand <= lb + 1e-9 else 0.1
        else:
            r = 0.1 + 0.9 * (m_base - m_cand) / denom
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
