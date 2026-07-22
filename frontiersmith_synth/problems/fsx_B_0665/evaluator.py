#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0665 -- "One Cache Policy vs a Gauntlet of Access Rhythms".

Family: phase-shift-cache-policy. The candidate designs ONE online cache-eviction
POLICY, expressed as a small weight vector over a fixed, causally-computable
feature set (recency / frequency / self-estimated period / predicted-next-use /
regime flag). The evaluator then RUNS that policy (never the candidate's own code)
online, causally, over a hidden gauntlet of 10 instances; each instance bundles a
small SUITE of structurally different access rhythms (pure long cyclic scans,
several short loops overloading the cache together, phase-interleaved duplicate
periods, noisy scans, boundary period==cache_size cases). An instance's score is
the WORST hit rate across its own suite (worst-trace-hit-rate): a policy that only
handles one rhythm well is punished by whichever trace in the suite it handles
badly. No single fixed replacement rule (pure LRU, pure MRU, pure LFU) wins every
regime -- the ceiling requires an online per-line estimate of "when will this line
next be needed" that adapts automatically as the active rhythm changes.

Candidate contract (isolated, stdin/stdout JSON, see isorun):
  input:  {"cache_size": C}                      (that instance's cache size ONLY;
                                                    the traces themselves are HIDDEN)
  output: {"w0":.., "w1":.., "w2":.., "w3":.., "w4":.., "w5":..}   six finite floats

At every cache MISS with the cache full, the evaluator scores every RESIDENT line y
using the candidate's weights against features computed ONLY from access history up
to and including the current step t (fully causal -- no future lookahead):
  recency(y)        = t - last_seen[y]
  freq(y)           = # times y accessed so far (in the whole trace, not just resident)
  gap_est(y)        = last_seen[y] - prev_seen[y]      (y's own last observed
                       inter-access gap; SENTINEL=1e6 if y has been seen only once)
  predicted_wait(y) = gap_est(y) - recency(y)           (estimated steps until y's
                       next use, under a "repeats with its last gap" model)
  regime(y)         = 1.0 if gap_est(y) > cache_size else 0.0
  score(y) = w0 + w1*recency(y) + w2*freq(y) + w3*gap_est(y)
                + w4*predicted_wait(y) + w5*regime(y)
The line with the LARGEST score is evicted (ties -> lowest line id). Per-line
metadata (last_seen/prev_seen/freq) persists for the whole trace even while a line
is not resident (cheap bookkeeping vs. expensive data storage -- standard in
history-aware cache policies).

Each hidden instance bundles 3 traces built by PROBABILISTICALLY interleaving three
cyclic components of very different periods (short/medium/long) plus light noise --
not a single perfectly uniform scan (that special symmetry would let any static,
signal-free address subset act as an accidental near-optimal reservoir). The short
component punishes MRU-like eviction (discarding a line right before it recurs);
the long component punishes LRU-like eviction (sequential flooding can crash
recency-chasing policies to near-0% hit rate); only a per-line reuse-gap estimate
handles both.

Scoring per trace: hitrate = hits/len(trace). The evaluator also computes Belady's
exact offline-optimal hit rate on the SAME trace (hi, a true ceiling no causal
policy can beat), inflated by a fixed headroom margin (hi' = min(1, hi+0.15) --
even a perfect period estimate should not saturate the score). Per-trace ratio =
clamp(hitrate/hi', 0, 1). An instance's score = MIN ratio over its own 3-trace
suite (worst-trace-hit-rate: a policy that only handles one rhythm well is
punished by whichever it misses). Final Ratio = mean over the 10 hidden instances.
Malformed / infeasible / non-finite answers score 0 on that instance.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean over 10 instances, in [0,1]>
  Vector: [r_1, ..., r_10]
"""
import sys, json, math
from collections import defaultdict, deque
import isorun

SENTINEL = 1.0e6
NOISE_BASE = 5_000_00


# ----------------------------- deterministic RNG ---------------------------
class RNG:
    def __init__(self, seed):
        self.s = seed & ((1 << 64) - 1)

    def _step(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.s

    def uf(self):
        return (self._step() >> 11) / float(1 << 53)


# ----------------------------- trace construction ---------------------------
def make_mixed_trace(rng, length, components, noise_rate=0.0, noise_base=NOISE_BASE):
    """components: list of {'period':p,'base':b,'weight':w}. Each step, with prob
    noise_rate emit a fresh never-repeating 'noise' address; else pick a component
    by weight and advance its own cyclic counter (independent phase per component,
    natural phase drift from probabilistic interleaving)."""
    counters = [0] * len(components)
    weights = [c["weight"] for c in components]
    total_w = float(sum(weights))
    seq = []
    noise_ctr = 0
    for _ in range(length):
        if noise_rate > 0.0 and rng.uf() < noise_rate:
            seq.append(noise_base + noise_ctr)
            noise_ctr += 1
            continue
        r = rng.uf() * total_w
        acc = 0.0
        ci = len(components) - 1
        for k, w in enumerate(weights):
            acc += w
            if r <= acc:
                ci = k
                break
        c = components[ci]
        addr = c["base"] + (counters[ci] % c["period"])
        counters[ci] += 1
        seq.append(addr)
    return seq


# ----------------------------- instance definitions -------------------------
# Each instance bundles a small SUITE of 3 traces, all built by *probabilistically*
# interleaving 3 cyclic components of very different periods (a short fast-cycling
# component, a medium component, a long slow-scanning component) plus light noise.
# Deliberately NOT a single perfectly-uniform cyclic scan: a perfectly uniform scan
# lets ANY fixed static address subset act as a near-optimal reservoir (an accident
# of symmetry, not real skill), which would make even a signal-free policy look
# good. The irregular, probabilistically-interleaved mix breaks that symmetry, so a
# policy actually has to track access history to do well -- the short component
# punishes MRU-like eviction (it discards lines that are about to recur), the long
# component punishes LRU-like eviction (sequential-flooding: recency chases a
# moving target and can crash to ~0% hit rate), and only an estimate of each LINE'S
# OWN reuse gap handles both at once.
def _instance_specs():
    # (seed, cache_size, [ (p_a,w_a, p_b,w_b, p_c,w_c, noise, length), ... ])
    return [
        (9001, 16, [(3, 4, 48, 2, 8, 1, 0.03, 1300), (3, 3, 72, 1, 19, 1, 0.03, 1300), (3, 5, 96, 1, 6, 2, 0.03, 1400)]),
        (9002, 20, [(3, 4, 60, 2, 10, 1, 0.03, 1300), (4, 3, 90, 1, 24, 1, 0.03, 1300), (3, 5, 120, 1, 6, 2, 0.03, 1400)]),
        (9003, 16, [(3, 4, 48, 2, 8, 1, 0.03, 1300), (3, 3, 72, 1, 19, 1, 0.03, 1300), (3, 5, 96, 1, 6, 2, 0.03, 1400)]),
        (9004, 24, [(4, 4, 72, 2, 12, 1, 0.03, 1300), (4, 3, 108, 1, 28, 1, 0.03, 1300), (3, 5, 144, 1, 8, 2, 0.03, 1400)]),
        (9005, 20, [(3, 4, 60, 2, 10, 1, 0.03, 1300), (4, 3, 90, 1, 24, 1, 0.03, 1300), (3, 5, 120, 1, 6, 2, 0.03, 1400)]),
        (9006, 24, [(4, 4, 72, 2, 12, 1, 0.03, 1300), (4, 3, 108, 1, 28, 1, 0.03, 1300), (3, 5, 144, 1, 8, 2, 0.03, 1400)]),
        (9007, 8, [(3, 4, 24, 2, 6, 1, 0.03, 1300), (3, 3, 36, 1, 9, 1, 0.03, 1300), (3, 5, 48, 1, 6, 2, 0.03, 1400)]),
        (9008, 40, [(6, 4, 120, 2, 20, 1, 0.03, 1300), (8, 3, 180, 1, 48, 1, 0.03, 1300), (5, 5, 240, 1, 13, 2, 0.03, 1400)]),
        (9009, 48, [(8, 4, 144, 2, 24, 1, 0.03, 1300), (9, 3, 216, 1, 57, 1, 0.03, 1300), (6, 5, 288, 1, 16, 2, 0.03, 1400)]),
        (9010, 32, [(5, 4, 96, 2, 16, 1, 0.03, 1300), (6, 3, 144, 1, 38, 1, 0.03, 1300), (4, 5, 192, 1, 10, 2, 0.03, 1400)]),
    ]


def make_instances():
    out = []
    for seed, C, trace_defs in _instance_specs():
        rng = RNG(seed)
        traces = []
        for i, (pa, wa, pb, wb, pc, wc, noise_rate, length) in enumerate(trace_defs):
            comps = [{"period": pa, "base": 0, "weight": wa},
                     {"period": pb, "base": 2_000_00, "weight": wb},
                     {"period": pc, "base": 5_000_00, "weight": wc}]
            traces.append(make_mixed_trace(rng, length, comps, noise_rate,
                                            noise_base=NOISE_BASE + i * 100000))
        out.append({"public": {"cache_size": C}, "hidden": {"traces": traces, "cache_size": C}})
    return out


# ----------------------------- reference simulators --------------------------
def simulate_fifo(trace, C):
    resident = set(); order = deque(); hits = 0
    for x in trace:
        if x in resident:
            hits += 1
        else:
            if len(resident) >= C:
                while True:
                    victim = order.popleft()
                    if victim in resident:
                        resident.discard(victim); break
            resident.add(x); order.append(x)
    return hits / len(trace)


def simulate_belady(trace, C):
    n = len(trace)
    positions = defaultdict(list)
    for i, a in enumerate(trace):
        positions[a].append(i)
    occ_idx = defaultdict(int)
    INF = n + 10
    resident = set(); hits = 0
    for t, x in enumerate(trace):
        if x in resident:
            hits += 1
        else:
            if len(resident) >= C:
                victim = None; far = -1
                for y in sorted(resident):
                    lst = positions[y]; idx = occ_idx[y]
                    nu = lst[idx] if idx < len(lst) else INF
                    if nu > far:
                        far = nu; victim = y
                resident.discard(victim)
            resident.add(x)
        occ_idx[x] += 1
    return hits / n


def _tiebreak(y):
    """Deterministic pseudo-random nudge in [0,1), decoupled from address
    magnitude/temporal-assignment order, so a policy with all-zero weights (or
    any exact tie) degrades to unstructured eviction instead of accidentally
    inheriting a reservoir-like policy from address ordering."""
    h = (y * 2654435761 + 40503) & 0xFFFFFFFF
    h ^= (h >> 15)
    return (h & 0xFFFFFF) / float(0x1000000)


def simulate_policy(trace, C, w):
    w0, w1, w2, w3, w4, w5 = w
    last_seen = {}; prev_seen = {}; freq = {}
    resident = set(); hits = 0
    for t, x in enumerate(trace):
        freq[x] = freq.get(x, 0) + 1
        if x in resident:
            hits += 1
        else:
            if len(resident) >= C:
                best = None; best_sc = float("-inf")
                for y in sorted(resident):
                    rec = t - last_seen[y]
                    f = freq.get(y, 0)
                    if y in prev_seen:
                        gap = last_seen[y] - prev_seen[y]
                    else:
                        gap = SENTINEL
                    pred = gap - rec
                    reg = 1.0 if gap > C else 0.0
                    sc = (w0 + w1 * rec + w2 * f + w3 * gap + w4 * pred + w5 * reg
                          + 1e-6 * _tiebreak(y))
                    if sc > best_sc:
                        best_sc = sc; best = y
                resident.discard(best)
            resident.add(x)
        old = last_seen.get(x)
        if old is not None:
            prev_seen[x] = old
        last_seen[x] = t
    return hits / len(trace)


# ----------------------------- scoring ---------------------------------------
_REF_CACHE = {}
HI_MARGIN = 0.15  # headroom above Belady's true optimal (never reachable online);
                   # keeps a perfect period-estimate policy from saturating to 1.0


def _hi(idx, trace, C):
    key = (idx, id(trace))
    if key not in _REF_CACHE:
        _REF_CACHE[key] = min(1.0, simulate_belady(trace, C) + HI_MARGIN)
    return _REF_CACHE[key]


def _parse_weights(answer):
    if not isinstance(answer, dict):
        return None
    keys = ["w0", "w1", "w2", "w3", "w4", "w5"]
    w = []
    for k in keys:
        if k not in answer:
            return None
        v = answer[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if not math.isfinite(v) or abs(v) > 1000.0:
            return None
        w.append(v)
    return w


def score(inst, answer):
    """Validate + run the candidate's policy weights over this instance's hidden
    trace suite; return (ok, per-instance ratio already in [0,1])."""
    w = _parse_weights(answer)
    if w is None:
        return False, None
    hidden = inst["hidden"]
    C = hidden["cache_size"]; traces = hidden["traces"]
    ratios = []
    for i, trace in enumerate(traces):
        hi = _hi(i, trace, C)
        try:
            hitrate = simulate_policy(trace, C, w)
        except Exception:
            return False, None
        if not math.isfinite(hitrate):
            return False, None
        ratio = hitrate / max(hi, 1e-6)
        ratios.append(max(0.0, min(1.0, ratio)))
    if not ratios:
        return False, None
    # worst-trace-hit-rate: the instance is scored by the WORST of its own suite --
    # a policy that only handles one rhythm well is punished by whichever it misses.
    return True, min(ratios)


def baseline(inst):
    """Informational: this instance's worst-trace FIFO hit rate (a naive floor)."""
    hidden = inst["hidden"]
    C = hidden["cache_size"]; traces = hidden["traces"]
    return min(simulate_fifo(t, C) for t in traces)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None:
            vec.append(0.0); continue
        r = obj
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
