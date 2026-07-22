#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0673 -- "One Cache Config Against Five Workload Personalities"
(family: pinned-cache-trace-hedging; format B, quality-metric).

THEME.  A cache operator must ship ONE fixed cache configuration -- a small PINNED key
set (permanently resident, protected from eviction) plus a small set of POLICY WEIGHTS
that parametrize a generic online eviction-scoring rule -- before knowing which of five
very different access-pattern "personalities" (zipf / phase-shifting / scan-heavy /
bursty / adversarial-loop) will actually run against it in production. The same fixed
(pin, weights) answer is REPLAYED, online and causally, against all five traces, and the
instance is graded on its WORST (minimum) trace hit rate -- so a config that is brilliant
on four traces and catastrophic on the fifth scores as if it were catastrophic everywhere.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"instance_id": str, "capacity": C (int), "pin_budget": P (int),
             "universe_size": M (int),
             "traces": {"zipf": [...], "phase": [...], "scan": [...],
                        "bursty": [...], "loop": [...]}}   # each a list of int key ids
  stdout: ONE JSON object:
            {"pin": [k0, k1, ...],           # <= P distinct ints in [0, M)
             "w_lru": float, "w_mru": float, "w_lfu": float, "w_scan": float}
          "pin" keys are permanently resident for the ENTIRE replay of every trace (always
          a hit, never evicted, and do not count against the online-managed region -- the
          online-managed region has capacity C - len(pin)).  The four weights parametrize
          a single generic eviction rule (see SIMULATION below) applied identically, and
          purely causally (no lookahead), to all five traces.

  A valid answer has "pin" a list of at most P DISTINCT integers in [0, M), and all four
  weights finite numbers in [0, 8].  Any violation, a crash, a timeout, or non-JSON output
  makes that instance score 0.0.

SIMULATION (deterministic, causal, run independently per trace -- pin/weights are shared
across traces but resident-set/recency/frequency state is NOT).  At each access to key k:
  - if k is pinned: HIT.
  - elif k is resident in the C-len(pin)-slot managed region: HIT.
  - else: MISS. Compute scan_signal = fraction of the last 24 accesses that were each
    key's FIRST-EVER occurrence in this trace (near 1.0 during a long one-shot sweep of
    fresh keys, low during steady reuse). If w_scan>0, key k is first-ever, and
    scan_signal * w_scan >= 1.0, the miss is BYPASSED (not cached, no eviction) -- a
    scan-admission gate. Otherwise, if the managed region has a free slot, k is inserted;
    else the resident j maximizing
        w_lru * recnorm(j) + w_mru * (1 - recnorm(j)) - w_lfu * freqnorm(j)
    is evicted and replaced by k, where recnorm/freqnorm normalize (time-since-last-use)
    and (uses-so-far) by their current max among residents. w_lru=1 alone reproduces
    plain LRU; w_mru alone reproduces MRU (evict the newest); w_lfu penalizes evicting
    frequently-used residents.

SCORING.  Per instance: q_base = worst-trace hit rate of the reference (pin=[], pure
LRU) baseline; q_opt = worst-trace hit rate of an UNCONSTRAINED, offline-optimal (Belady)
cache of size C on each trace (a true, generally unreachable ceiling for ANY realizable
online scheme with total budget C); q_cand = worst-trace hit rate of the candidate's
(pin, weights). Normalize with a stretched affine anchor (baseline -> 0.1, 1.5x the
Belady gap -> 1.0, so even matching Belady on the binding trace leaves headroom):
    r = clamp(0.1 + 0.9 * (q_cand - q_base) / max(1e-9, 1.5 * (q_opt - q_base)), 0, 1)

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance. q_base/q_opt are computed by
THIS parent process, so a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
from collections import defaultdict
import isorun

# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# --------------------------- trace-family generators ------------------------
# A small CORE (ids 0..3) is woven into every one of the five traces (a planted
# small "intersection working set"); a small SCAN-secondary pair (ids 4..5) is
# woven only into the scan trace, as a trap-relevant target for spare pin budget.
CORE = [0, 1, 2, 3]
SCAN_SEC = [4, 5]
SCAN_WINDOW = 24


def _gen_zipf(rng, L, extra_lo, extra_hi, core_prob):
    extra = list(range(extra_lo, extra_hi))
    weights = [1.0 / r for r in range(1, len(extra) + 1)]
    tot = sum(weights)
    cum = []
    c = 0.0
    for w in weights:
        c += w
        cum.append(c)
    seq = []
    for _ in range(L):
        if rng(0, 999) < int(core_prob * 1000):
            seq.append(CORE[rng(0, len(CORE) - 1)])
            continue
        x = (rng(0, 999999) / 1000000.0) * tot
        idx = len(extra) - 1
        for i, cv in enumerate(cum):
            if x <= cv:
                idx = i
                break
        seq.append(extra[idx])
    return seq


def _gen_phase(rng, L, phase_ranges, core_prob, n_phases=3):
    seq = []
    plen = L // n_phases
    for p in range(n_phases):
        lo, hi = phase_ranges[p % len(phase_ranges)]
        hot = list(range(lo, hi))
        for _ in range(plen):
            if rng(0, 999) < int(core_prob * 1000):
                seq.append(CORE[rng(0, len(CORE) - 1)])
            else:
                seq.append(hot[rng(0, len(hot) - 1)])
    while len(seq) < L:
        seq.append(CORE[rng(0, len(CORE) - 1)])
    return seq


def _gen_scan(rng, L, scan_lo, scan_hi, core_prob, sec_prob):
    seq = []
    cursor = scan_lo
    for _ in range(L):
        r = rng(0, 999)
        if r < int(core_prob * 1000):
            seq.append(CORE[rng(0, len(CORE) - 1)])
        elif r < int((core_prob + sec_prob) * 1000):
            seq.append(SCAN_SEC[rng(0, len(SCAN_SEC) - 1)])
        else:
            seq.append(cursor)
            cursor += 1
            if cursor >= scan_hi:
                cursor = scan_lo
    return seq


def _gen_bursty(rng, L, pool_lo, pool_hi, core_prob, burst_len=16, subset_size=4):
    pool = list(range(pool_lo, pool_hi))
    seq = []
    cur_subset = None
    remaining = 0
    while len(seq) < L:
        if rng(0, 999) < int(core_prob * 1000):
            seq.append(CORE[rng(0, len(CORE) - 1)])
            continue
        if remaining <= 0:
            idxs = []
            while len(idxs) < subset_size:
                k = pool[rng(0, len(pool) - 1)]
                if k not in idxs:
                    idxs.append(k)
            cur_subset = idxs
            remaining = burst_len
        seq.append(cur_subset[rng(0, len(cur_subset) - 1)])
        remaining -= 1
    return seq[:L]


def _gen_loop(rng, L, loop_lo, loop_hi):
    # pure cyclic sweep -- period Q = |CORE| + (loop_hi - loop_lo), always kept
    # strictly larger than the cache capacity so plain-LRU (and any policy whose
    # managed region alone is smaller than Q) achieves ~0% hit rate on the
    # non-core portion: the classic worst case for recency-only eviction.
    cycle = list(CORE) + list(range(loop_lo, loop_hi))
    Q = len(cycle)
    return [cycle[i % Q] for i in range(L)]


def _build_traces(seed, L, scan_span, loop_extra):
    zipf = _gen_zipf(_rng(seed * 7 + 1), L, 6, 40, core_prob=0.10)
    phase = _gen_phase(_rng(seed * 7 + 2), L, [(6, 18), (18, 30), (30, 42)], core_prob=0.12)
    scan = _gen_scan(_rng(seed * 7 + 3), L, 42, 42 + scan_span, core_prob=0.08, sec_prob=0.20)
    bursty = _gen_bursty(_rng(seed * 7 + 4), L, 42, 60, core_prob=0.10)
    loop = _gen_loop(_rng(seed * 7 + 5), L, 60, 60 + loop_extra)
    universe = 60 + loop_extra
    universe = max(universe, 42 + scan_span, 60)
    return {"zipf": zipf, "phase": phase, "scan": scan, "bursty": bursty, "loop": loop}, universe


def _build_instances():
    """Deterministic instance family: (seed, L, capacity, pin_budget, scan_span, loop_extra)."""
    specs = [
        (2101, 460, 24, 6, 120, 30),
        (2102, 480, 24, 6, 130, 32),
        (2103, 500, 22, 6, 110, 28),
        (2104, 480, 26, 7, 140, 34),
        (2105, 460, 24, 6, 120, 30),
        (2106, 500, 20, 5, 100, 26),
        (2107, 480, 24, 6, 130, 32),
        (2108, 500, 22, 6, 120, 30),
        # harder / held-out (bigger scan sweep + bigger loop period, tighter pin)
        (2201, 560, 24, 5, 220, 40),
        (2202, 560, 20, 5, 240, 36),
    ]
    out = []
    for seed, L, C, P, scan_span, loop_extra in specs:
        traces, universe = _build_traces(seed, L, scan_span, loop_extra)
        out.append({"name": f"cache{seed}", "capacity": C, "pin_budget": P,
                    "universe_size": universe, "traces": traces})
    return out


# ----------------------------- online cache simulation ----------------------
def _simulate(trace, pin, w_lru, w_mru, w_lfu, w_scan, C, W=SCAN_WINDOW):
    pinset = set(pin)
    managed_cap = max(0, C - len(pinset))
    resident = set()
    last_access = {}
    freq = {}
    ever_seen = set()
    window = []
    hits = 0
    n = len(trace)
    for t, k in enumerate(trace):
        first_time = k not in ever_seen
        ever_seen.add(k)
        scan_signal = (sum(window) / len(window)) if window else 0.0
        if k in pinset or k in resident:
            hits += 1
            freq[k] = freq.get(k, 0) + 1
            last_access[k] = t
        else:
            admit = True
            if w_scan > 0 and first_time and (scan_signal * w_scan >= 1.0):
                admit = False
            if admit:
                if len(resident) < managed_cap:
                    resident.add(k)
                else:
                    maxrec = 0
                    maxfreq = 0
                    recs = {}
                    freqs = {}
                    for j in resident:
                        recs[j] = t - last_access.get(j, t - 1)
                        freqs[j] = freq.get(j, 0)
                        if recs[j] > maxrec:
                            maxrec = recs[j]
                        if freqs[j] > maxfreq:
                            maxfreq = freqs[j]
                    maxrec = maxrec or 1
                    maxfreq = maxfreq or 1
                    best_key = None
                    best_score = None
                    for j in sorted(resident):
                        recnorm = recs[j] / maxrec
                        freqnorm = freqs[j] / maxfreq
                        score = w_lru * recnorm + w_mru * (1 - recnorm) - w_lfu * freqnorm
                        if best_score is None or score > best_score:
                            best_score = score
                            best_key = j
                    resident.discard(best_key)
                    resident.add(k)
            freq[k] = freq.get(k, 0) + 1
            last_access[k] = t
        window.append(1 if first_time else 0)
        if len(window) > W:
            window.pop(0)
    return hits / n if n else 0.0


def _opt_hit_rate(trace, C):
    """Belady offline-optimal hit rate for an UNCONSTRAINED cache of size C (a
    true ceiling for any realizable online/pinned scheme with the same total budget)."""
    n = len(trace)
    positions = defaultdict(list)
    for i, k in enumerate(trace):
        positions[k].append(i)
    ptr = defaultdict(int)

    def next_use(k, t):
        lst = positions[k]
        p = ptr[k]
        while p < len(lst) and lst[p] <= t:
            p += 1
        ptr[k] = p
        return lst[p] if p < len(lst) else math.inf

    resident = set()
    hits = 0
    for t, k in enumerate(trace):
        if k in resident:
            hits += 1
            next_use(k, t)
            continue
        if len(resident) < C:
            resident.add(k)
        else:
            far_key = None
            far_val = -1
            for j in resident:
                nu = next_use(j, t)
                if nu > far_val:
                    far_val = nu
                    far_key = j
            resident.discard(far_key)
            resident.add(k)
        next_use(k, t)
    return hits / n if n else 0.0


# ----------------------------- answer validation -----------------------------
def _validate_answer(answer, pin_budget, universe_size):
    if not isinstance(answer, dict):
        return None
    pin = answer.get("pin")
    if not isinstance(pin, list) or len(pin) > pin_budget:
        return None
    seen = set()
    for x in pin:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if x < 0 or x >= universe_size:
            return None
        if x in seen:
            return None
        seen.add(x)
    weights = []
    for key in ("w_lru", "w_mru", "w_lfu", "w_scan"):
        v = answer.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if not math.isfinite(v) or v < 0.0 or v > 8.0:
            return None
        weights.append(v)
    return pin, weights


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
        P = inst["pin_budget"]
        M = inst["universe_size"]
        traces = inst["traces"]
        names = list(traces.keys())

        q_base = min(_simulate(traces[nm], [], 1.0, 0.0, 0.0, 0.0, C) for nm in names)
        q_opt = min(_opt_hit_rate(traces[nm], C) for nm in names)
        denom = 1.5 * (q_opt - q_base)
        if denom < 1e-9:
            denom = 1e-9

        public = {"instance_id": inst["name"], "capacity": C, "pin_budget": P,
                  "universe_size": M, "traces": {nm: list(traces[nm]) for nm in names}}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            parsed = _validate_answer(ans, P, M)
        except Exception:
            parsed = None
        if parsed is None:
            vec.append(0.0)
            continue
        pin, (w_lru, w_mru, w_lfu, w_scan) = parsed
        try:
            q_cand = min(_simulate(traces[nm], pin, w_lru, w_mru, w_lfu, w_scan, C) for nm in names)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (q_cand - q_base) / denom
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
