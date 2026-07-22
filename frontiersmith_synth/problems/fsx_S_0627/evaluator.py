#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0627 -- "Twenty Questions Against a Sealed Physics Box:
Budgeted Probing of a Symmetric Piecewise Law" (family: probe-budget-mechanism-id;
format B, quality-metric; theme: identify a hidden law by choosing what to measure).

THEME.  A sealed box hides a scalar law f(x): x -> R that is PIECEWISE-LINEAR with a few
unknown breakpoints (discontinuities / slope changes) AND obeys a hidden POINT SYMMETRY:
there exist an unknown center c and value b = f(c) with
        f(2c - x) = 2*b - f(x)        (rotational symmetry about the point (c, b)).
You may PROBE the box: each probe names an x in the QUERYABLE window [0, QR] and returns a
NOISY reading y = f(x) + eps.  You get only Q probes total, spent across a few adaptive
ROUNDS (each round you see every reading so far and choose the next batch).  Then the box is
resealed and you must PREDICT f on a fixed dense grid over [0, GR] -- which EXTENDS BEYOND
the queryable window: the tail (QR, GR] can NEVER be probed directly.

WHY THE TWO MECHANISMS ARE FORCED (both shape the score).
  * active-experiment-budget -- Q is tight relative to the grid.  A probe's worth is the
    INFORMATION it buys, not the value it reads.  The symmetry means f on the whole domain
    is fixed by its values on [0, c]; a probe at x ALSO constrains the mirror point 2c - x.
    So probes spent identifying (c, b) and reconstructing [0, c] pay double: they let you
    REFLECT the un-probeable tail into a region you did probe.  Spend the budget resolving
    smooth interior instead and the tail is lost.
  * breakpoint-localization -- prediction error concentrates at the discontinuities.  Where
    breakpoints sit is unknown; pinning them needs probes PLACED at them, which needs
    ADAPTATION (bisect toward where adjacent readings jump).  A uniform sweep of the same Q
    resolves smooth stretches it did not need and leaves each breakpoint (and its mirror in
    the tail) smeared.

INNOVATION HOOK (what `strong` exploits).  Under a tight probe budget the value of a probe
is its information about DISCONTINUITIES and SYMMETRIES, not about function values --
experiment design beats regression.  `strong` (a) spends a few probes to IDENTIFY the
symmetry center c and offset b (by finding the c that makes probed pairs (x, 2c-x) consistent
with f(x)+f(2c-x)=const), (b) adaptively bisects toward breakpoints to localize them, and
(c) predicts the un-probeable tail by REFLECTION 2b - f(2c - x).  The TRAP instances put
steep, high-jump structure in the tail (the mirror of the interior), so the obvious recipe --
spread Q uniformly over [0, QR] and linearly extrapolate past QR -- lands far from strong on
those cases.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called ONCE PER ROUND).
  The evaluator drives an interactive loop by RE-INVOKING the candidate each round with the
  full public state.  The candidate is stateless across calls; it re-derives its plan from
  the history it is handed.
    stdin  : ONE JSON object.  Common fields:
        {"name","phase","QR","GR","G","c_lo","c_hi","Q",
         "budget_left","round","R","max_this_round",
         "history": [[x, y], ...]}       # every probe SO FAR, y=null if x was out of window
      phase == "query"   -> return {"queries": [x1, x2, ...]}   (each x a real number;
                            only x in [0,QR] is answered, others count against budget and
                            return null; at most min(budget_left, max_this_round) are honored)
      phase == "predict" -> return {"pred": [y_0, ..., y_{G-1}]}  (prediction at grid point
                            g_j = GR * j/(G-1) for j in 0..G-1)
  Any crash / timeout / non-JSON / wrong shape on ANY call -> 0.0 on that instance.

SCORING (deterministic; no wall-time).  Per instance, with truth f on the grid:
    err_ref = mean_j |f(g_j)|                       # error of the do-nothing "predict 0"
    err     = mean_j |pred_j - f(g_j)|
    quality = clip(1 - err/err_ref, 0, 1)           # 0 for predict-zero, ->1 for perfect
    r       = OFFSET + SPAN * quality                # OFFSET=0.10, SPAN=0.82 (cap 0.92)
  Predict-zero scores exactly OFFSET=0.10; noise + un-pinned breakpoints + tail-reflection
  slack keep even a strong strategy below the 0.92 cap, so headroom stays open above the
  reference.  Final score = mean of r over 10 fixed seeded instances (4 traps + 3 moderate
  + 3 gentle/held-out).

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it sees only the public state.  c, b, the breakpoints, the noise
tape, and f live only in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json
import isorun

# --------------------------- scoring / protocol constants ------------------------
OFFSET = 0.10
SPAN = 0.82          # r = OFFSET + SPAN*quality ; max attainable r = 0.92 (headroom)
QR = 12.0            # queryable window [0, QR]
GR = 20.0            # prediction grid [0, GR]  (tail (QR,GR] is un-probeable)
G = 400              # grid points
Q = 36               # probe budget
R = 5                # adaptive rounds
MAX_PER_ROUND = Q    # a round may spend up to the whole remaining budget (rounds optional)


# ------------------------------- deterministic RNG -------------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def u01():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return u01


def _uni(u01, lo, hi):
    return lo + (hi - lo) * u01()


def _gauss(u01):
    # sum of 6 uniforms -> approx normal, mean 0, std ~ sqrt(6/12)=0.707
    return (u01() + u01() + u01() + u01() + u01() + u01()) - 3.0


# ------------------------------- instance family ---------------------------------
def _build_one(seed, K, slope_rng, jump_rng, sigma, v0_rng, tail_steep):
    """Construct a hidden law.  The LEFT function fL is piecewise-linear on [0, c] with K
    interior breakpoints (slope changes + jumps); the whole law is fL on [0,c] and its point
    reflection about (c, b) on (c, GR].  `tail_steep` scales the interior slopes/jumps that,
    once reflected, populate the un-probeable tail -- large for traps."""
    u = _rng(seed)
    c = _uni(u, 10.2, 11.3)                     # center near the high end of the window
    # interior breakpoints strictly inside (0.6, c-0.6)
    ks = sorted(_uni(u, 0.8, c - 0.8) for _ in range(K))
    # keep them separated a bit
    for i in range(1, K):
        if ks[i] - ks[i - 1] < 0.6:
            ks[i] = ks[i - 1] + 0.6
    ks = [k for k in ks if k < c - 0.4]
    X = [0.0] + ks + [c]                        # knots; segments 0..len-2
    nseg = len(X) - 1
    slopes = []
    for i in range(nseg):
        s = _uni(u, -slope_rng, slope_rng) * tail_steep
        slopes.append(s)
    jumps = []                                  # one jump at each interior knot
    for i in range(nseg - 1):
        mag = _uni(u, jump_rng * 0.5, jump_rng) * tail_steep
        sgn = 1.0 if u() < 0.5 else -1.0
        jumps.append(sgn * mag)
    # segment start values
    v0 = _uni(u, -v0_rng, v0_rng)
    V = [0.0] * nseg
    V[0] = v0
    for i in range(nseg - 1):
        end = V[i] + slopes[i] * (X[i + 1] - X[i])
        V[i + 1] = end + jumps[i]
    b = V[nseg - 1] + slopes[nseg - 1] * (c - X[nseg - 1])
    # noise tape (indexed by count of answered probes)
    tape = [sigma * _gauss(u) for _ in range(Q + 8)]
    return {"c": c, "b": b, "X": X, "V": V, "slopes": slopes, "sigma": sigma, "tape": tape}


def _fL(inst, x):
    X, V, S = inst["X"], inst["V"], inst["slopes"]
    nseg = len(X) - 1
    # locate segment i with X[i] <= x <= X[i+1]
    i = 0
    while i < nseg - 1 and x >= X[i + 1]:
        i += 1
    return V[i] + S[i] * (x - X[i])


def _f(inst, x):
    c = inst["c"]
    if x <= c:
        return _fL(inst, x)
    xm = 2.0 * c - x                            # in [2c-GR, c) ; 2c>=GR so xm>=0
    return 2.0 * inst["b"] - _fL(inst, xm)


def _build_instances():
    specs = [
        # name,          seed,  K, slope, jump, sigma, v0, tail_steep
        ("trap1",        62701, 3, 2.2, 4.0, 0.22, 3.0, 1.35),
        ("trap2",        62702, 4, 2.5, 4.5, 0.24, 3.0, 1.40),
        ("trap3",        62703, 3, 2.0, 5.0, 0.22, 2.5, 1.45),
        ("trap4",        62704, 4, 2.4, 4.2, 0.26, 3.0, 1.30),
        ("mod1",         62711, 3, 1.8, 3.0, 0.22, 2.5, 1.00),
        ("mod2",         62712, 3, 1.6, 3.2, 0.24, 2.5, 1.05),
        ("mod3",         62713, 4, 1.7, 2.8, 0.22, 2.5, 1.00),
        ("gentle1",      62721, 2, 1.2, 1.6, 0.20, 2.0, 0.85),
        ("gentle2",      62722, 2, 1.0, 1.4, 0.20, 2.0, 0.80),
        ("gentle3",      62723, 3, 1.1, 1.5, 0.20, 2.0, 0.85),
    ]
    out = []
    for name, seed, K, sl, jp, sg, v0, ts in specs:
        inst = _build_one(seed, K, sl, jp, sg, v0, ts)
        inst["name"] = name
        out.append(inst)
    return out


def _grid_x(j):
    return GR * j / (G - 1)


def _err_ref(inst):
    s = 0.0
    for j in range(G):
        s += abs(_f(inst, _grid_x(j)))
    return s / G


# ------------------------------- interactive run ---------------------------------
def _public_query(inst, phase, history, budget_left, rnd):
    return {"name": inst["name"], "phase": phase, "QR": QR, "GR": GR, "G": G,
            "c_lo": 10.0, "c_hi": 11.5, "Q": Q,
            "budget_left": budget_left, "round": rnd, "R": R,
            "max_this_round": MAX_PER_ROUND,
            "history": [[h[0], h[1]] for h in history]}


def _run_instance(cand, inst):
    history = []            # list of [x, y]  (y None if x out of window)
    budget = Q
    tape_i = 0
    tape = inst["tape"]
    # ---- adaptive query rounds ----
    for rnd in range(R):
        if budget <= 0:
            break
        pub = _public_query(inst, "query", history, budget, rnd)
        ans, st = isorun.run_candidate(cand, pub, timeout=20)
        if st != "OK" or not isinstance(ans, dict):
            return 0.0
        qs = ans.get("queries", [])
        if not isinstance(qs, list):
            return 0.0
        allow = min(budget, MAX_PER_ROUND)
        n_taken = 0
        for xq in qs:
            if n_taken >= allow or budget <= 0:
                break
            if isinstance(xq, bool) or not isinstance(xq, (int, float)):
                return 0.0
            xf = float(xq)
            if xf != xf or xf in (float("inf"), float("-inf")):
                return 0.0
            budget -= 1
            n_taken += 1
            if 0.0 <= xf <= QR:
                y = _f(inst, xf) + (tape[tape_i] if tape_i < len(tape) else 0.0)
                tape_i += 1
                history.append([xf, y])
            else:
                history.append([xf, None])
    # ---- prediction phase ----
    pub = _public_query(inst, "predict", history, budget, R)
    ans, st = isorun.run_candidate(cand, pub, timeout=20)
    if st != "OK" or not isinstance(ans, dict):
        return 0.0
    pred = ans.get("pred")
    if not isinstance(pred, list) or len(pred) != G:
        return 0.0
    err = 0.0
    for j in range(G):
        p = pred[j]
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            return 0.0
        pf = float(p)
        if pf != pf or pf in (float("inf"), float("-inf")):
            return 0.0
        err += abs(pf - _f(inst, _grid_x(j)))
    err /= G
    eref = _err_ref(inst)
    if eref <= 1e-9:
        eref = 1e-9
    quality = 1.0 - err / eref
    if quality < 0.0:
        quality = 0.0
    elif quality > 1.0:
        quality = 1.0
    r = OFFSET + SPAN * quality
    if r < 0.0:
        r = 0.0
    elif r > 1.0:
        r = 1.0
    return r


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = _build_instances()
    vec = []
    for inst in insts:
        try:
            r = _run_instance(cand, inst)
        except Exception:
            r = 0.0
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        vec.append(r)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
