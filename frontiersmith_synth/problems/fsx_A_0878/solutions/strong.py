# TIER: strong
# Calibrated adaptive blend. The insight: neither ignoring the runner (LRU) nor
# blindly trusting them (hint-follow) is robust across a whole shift, because
# the runner's reliability is itself unknown and can change mid-shift.
#
# Two-stage calibration against the given calibration shift (same recipe as
# the real shift, independent draw):
#   1. Estimate a GLOBAL trust prior from data: reconstruct the calibration
#      shift's own true next-use gaps (a simple backward scan over data the
#      candidate legitimately owns) and correlate them with the runner's
#      forecasts. Strong positive correlation -> the runner is a good
#      predictor (trust ~1); near-zero or NEGATIVE correlation (a guessing or
#      backwards runner) -> trusting the raw forecast is not just useless but
#      actively harmful, so trust is clamped to ~0, i.e. fall back to LRU.
#      This aggregate statistic is far less noise-sensitive than picking
#      whichever single simulated policy happens to minimize misses on one
#      short calibration draw (which overfits to that draw's specific noise).
#   2. With that trust prior anchored, grid-search only the ADAPTATION knobs
#      (EWMA decay, phase-edge lookback window/threshold, whether to
#      re-anchor trust at a flagged edge) by replaying candidates on the
#      calibration shift, so the policy can still react to an in-shift regime
#      change instead of just running a static blend all the way through.
import sys, json, math
from collections import deque


def replay(trace, hints, C, pol):
    """Mirrors the judge's causal replay engine exactly (see evaluator.py's
    _replay docstring for why trust-tracking is mode-independent forecast
    verification rather than an eviction-outcome "regret" signal)."""
    mode = pol["mode"]
    if mode == "lru":
        trust, adaptive = 0.0, False
    elif mode == "hint":
        trust, adaptive = 1.0, False
    else:
        trust, adaptive = pol["init_trust"], True
    decay = pol.get("decay", 0.1)
    window = pol.get("window", 10)
    edge_threshold = pol.get("edge_threshold", 0.5)
    reset_on_edge = pol.get("reset_on_edge", False)
    init_trust = pol.get("init_trust", trust)

    cache = set()
    last_idx = {}
    hint_at = {}
    pending_verify = {}
    misses = 0
    recent = deque(maxlen=window if window >= 2 else 2)

    for t, page in enumerate(trace):
        h = hints[t]

        if page in pending_verify:
            t0, pred0 = pending_verify.pop(page)
            if adaptive:
                actual_gap = t - t0
                tol = max(3.0, 0.4 * actual_gap)
                if abs(actual_gap - pred0) <= tol:
                    trust = trust + decay * (1.0 - trust)
                else:
                    trust = trust - decay * trust

        if page in cache:
            hit = True
            last_idx[page] = t
            hint_at[page] = h
        else:
            hit = False
            misses += 1
            if len(cache) >= C:
                pages = list(cache)
                k = len(pages)
                if k == 1:
                    victim = pages[0]
                else:
                    rec_sorted = sorted(pages, key=lambda p: (last_idx[p], p))
                    rec_rank = {p: i for i, p in enumerate(rec_sorted)}
                    rec_norm = {p: (k - 1 - rec_rank[p]) / (k - 1) for p in pages}
                    remgap = {p: hint_at[p] - (t - last_idx[p]) for p in pages}
                    hint_sorted = sorted(pages, key=lambda p: (remgap[p], p))
                    hint_rank = {p: i for i, p in enumerate(hint_sorted)}
                    hint_norm = {p: hint_rank[p] / (k - 1) for p in pages}
                    combo = {p: trust * hint_norm[p] + (1.0 - trust) * rec_norm[p] for p in pages}
                    victim = max(pages, key=lambda p: (combo[p], -last_idx[p], -p))
                cache.discard(victim)
                del last_idx[victim]
                del hint_at[victim]
            cache.add(page)
            last_idx[page] = t
            hint_at[page] = h

        pending_verify[page] = (t, h)

        recent.append(0 if hit else 1)
        if adaptive and len(recent) == recent.maxlen:
            if (sum(recent) / len(recent)) >= edge_threshold and reset_on_edge:
                trust = init_trust
                recent.clear()

    return misses


def _true_gaps(trace, sentinel):
    n = len(trace)
    next_pos = {}
    gaps = [0] * n
    for t in range(n - 1, -1, -1):
        p = trace[t]
        gaps[t] = (next_pos[p] - t) if p in next_pos else sentinel
        next_pos[p] = t
    return gaps


def _pearson(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def main():
    inst = json.load(sys.stdin)
    C = inst["capacity"]
    calib = inst["calib"]
    page, hint = calib["page"], calib["hint"]

    # ---- stage 1: data-driven trust prior from forecast/truth correlation ----
    sentinel = (max(hint) + 1) if hint else 1
    true_gap = _true_gaps(page, sentinel)
    corr = _pearson(hint, true_gap)
    # dead-zone: a barely-positive correlation is noise, not signal -- only
    # commit trust once the correlation clears a real threshold, else a
    # near-random forecast still leaks harmful weight into the blend.
    trust0 = max(0.0, min(1.0, (corr - 0.15) / 0.85))

    # ---- stage 2: grid-search only the adaptation knobs, prior anchored ----
    candidates = [{"mode": "lru"}, {"mode": "hint"}]
    for dc in (0.1, 0.2, 0.35, 0.5):
        for wd in (6, 10, 16):
            for eth in (0.3, 0.5, 0.7):
                for roe in (True, False):
                    candidates.append({
                        "mode": "blend", "init_trust": trust0, "decay": dc,
                        "window": wd, "edge_threshold": eth, "reset_on_edge": roe,
                    })

    # A rigid, non-adaptive fixed mode has no fallback if reliability turns
    # out to differ on the real shift from this one calibration draw; only
    # let it win the backtest by a clear margin, not by noise-level luck.
    FIXED_MODE_HANDICAP = max(2, len(page) // 100)

    best_pol, best_score = None, None
    for pol in candidates:
        try:
            m = replay(page, hint, C, pol)
        except Exception:
            continue
        score = m + (FIXED_MODE_HANDICAP if pol["mode"] in ("lru", "hint") else 0)
        if best_score is None or score < best_score:
            best_score, best_pol = score, pol

    if best_pol is None:
        best_pol = {"mode": "lru"}
    print(json.dumps(best_pol))


main()
