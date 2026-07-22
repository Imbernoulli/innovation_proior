# TIER: strong
# INSIGHT (not "greedy + more probes"): under a tight probe budget a probe's worth is the
# INFORMATION it buys about the law's STRUCTURE -- its symmetry and its discontinuities --
# not the value it reads.  Three composed moves:
#   (1) IDENTIFY THE SYMMETRY.  The law obeys point symmetry f(2c-x)=2b-f(x) for hidden
#       (c,b).  From the probes, find the center c that makes mirrored pairs consistent:
#       for a candidate c, m_k = (y_k + f_hat(2c - x_k))/2 should be CONSTANT (=b) across k.
#       Pick the c minimizing the spread of {m_k}; b is their mean.  This spends a handful of
#       probes to pin a mechanism, not to regress a curve.
#   (2) LOCALIZE BREAKPOINTS.  Error concentrates at the discontinuities.  After a coarse
#       sweep, adaptively BISECT toward adjacent readings whose slope jumps -- placing later
#       probes exactly where the kink is -- so each breakpoint (and its mirror in the tail)
#       is pinned instead of smeared.
#   (3) REFLECT THE TAIL.  The grid extends past the queryable window into a tail that can
#       NEVER be probed.  Symmetry maps it into the interior we DID probe: predict the tail
#       as 2b - f_hat(2c - x).  This is the payoff the uniform-sweep-then-extrapolate recipe
#       cannot get.
import sys, json

inst = json.load(sys.stdin)
phase = inst.get("phase")
QR = float(inst["QR"]); GR = float(inst["GR"]); G = int(inst["G"])
Q = int(inst["Q"]); c_lo = float(inst["c_lo"]); c_hi = float(inst["c_hi"])

N0 = 18          # coarse round-0 sweep
PER_ROUND = 6    # refinement probes per later round
WMIN = 0.14      # do not bisect intervals narrower than this


def _pts(inst):
    hist = inst.get("history", [])
    pts = sorted((float(h[0]), float(h[1])) for h in hist if h[1] is not None)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return xs, ys


def _interp(xs, ys, x):
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return ys[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[lo]
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


def _kink_intervals(xs, ys):
    """Return interval midpoints ranked by how strong a slope-kink borders them."""
    n = len(xs)
    if n < 3:
        return []
    scored = []
    for i in range(n - 1):
        w = xs[i + 1] - xs[i]
        if w < WMIN:
            continue
        # kink magnitude at the two endpoints of this interval
        k = 0.0
        if i >= 1:
            sl = (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] != xs[i - 1] else 0.0
            sr = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) if xs[i + 1] != xs[i] else 0.0
            k = max(k, abs(sr - sl))
        if i + 2 < n:
            sl = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) if xs[i + 1] != xs[i] else 0.0
            sr = (ys[i + 2] - ys[i + 1]) / (xs[i + 2] - xs[i + 1]) if xs[i + 2] != xs[i + 1] else 0.0
            k = max(k, abs(sr - sl))
        scored.append((k * w, 0.5 * (xs[i] + xs[i + 1])))
    scored.sort(reverse=True)
    return [m for _, m in scored]


def _estimate_cb(xs, ys):
    if len(xs) < 4:
        return 0.5 * (c_lo + c_hi), (sum(ys) / len(ys) if ys else 0.0)
    xmin, xmax = xs[0], xs[-1]
    best_c, best_b, best_res = 0.5 * (c_lo + c_hi), 0.0, None
    steps = 90
    for s in range(steps + 1):
        c = c_lo + (c_hi - c_lo) * s / steps
        ms = []
        for k in range(len(xs)):
            xm = 2.0 * c - xs[k]
            if xmin + 1e-9 <= xm <= xmax - 1e-9:
                ms.append(0.5 * (ys[k] + _interp(xs, ys, xm)))
        if len(ms) < 4:
            continue
        mean = sum(ms) / len(ms)
        res = sum((m - mean) ** 2 for m in ms) / len(ms)
        if best_res is None or res < best_res:
            best_res, best_c, best_b = res, c, mean
    return best_c, best_b


if phase == "query":
    budget = int(inst["budget_left"])
    rnd = int(inst.get("round", 0))
    if rnd == 0:
        n = min(N0, budget)
        qs = [QR * i / (n - 1) for i in range(n)] if n > 1 else [QR * 0.5]
        print(json.dumps({"queries": qs}))
    else:
        xs, ys = _pts(inst)
        want = min(PER_ROUND, budget)
        mids = _kink_intervals(xs, ys)
        out = []
        for m in mids:
            if len(out) >= want:
                break
            # dedup against existing probes and already-chosen midpoints
            if any(abs(m - x) < WMIN * 0.5 for x in xs):
                continue
            if any(abs(m - o) < WMIN * 0.5 for o in out):
                continue
            out.append(m)
        print(json.dumps({"queries": out}))
else:
    xs, ys = _pts(inst)
    c, b = _estimate_cb(xs, ys)
    pred = []
    for j in range(G):
        x = GR * j / (G - 1)
        if x <= QR:
            pred.append(_interp(xs, ys, x))
        else:
            xm = 2.0 * c - x
            pred.append(2.0 * b - _interp(xs, ys, xm))
    print(json.dumps({"pred": pred}))
