# TIER: strong
"""
Envelope (hinge) regression: reformulate the observed voltage as the lower
envelope of two latent log-log power-law surfaces, then recover both.

Because log() is monotonic, log(min(V1,V2)) == min(log V1, log V2) exactly,
so the whole problem can be solved in LOG SPACE with ordinary linear
algebra: each channel k is a log-log LINEAR surface log(V_k) = a_k + p_k*ld
+ q_k*lt (ld=log d, lt=log T). We do not know the assignment of training
points to channels, so we run a hard-EM / hinge-regression loop: fit a
log-log linear surface to each of two point groups, then reassign every
point to whichever surface currently predicts the SMALLER value (the
"responsible" channel for a weakest-link system), repeat to convergence.
Several structurally different initial splits (by d, by T, by the diagonal
d*T, by the anti-diagonal d/T, and random) are tried and the run with the
lowest total envelope residual on the training data is kept -- this is the
genuine insight: a single least-squares fit can never discover the second
channel, but alternating "fit two surfaces / reassign to the weaker one"
converges onto BOTH channels' own exponents, which is exactly what is
needed to saturate correctly on the held-out corners far outside the
sampled window.

After recovering both channels, a single softness constant k is
grid-searched against the TRAINING data (comparing the smoothed
weakest-link blend to the observed values) so the emitted law reproduces
the physically-required continuous crossover, not a hard kink.
"""
import sys
import math
import random


def solve3(M, V):
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    D = det3(M)
    if abs(D) < 1e-30:
        return None
    out = []
    for col in range(3):
        Mi = [row[:] for row in M]
        for r in range(3):
            Mi[r][col] = V[r]
        out.append(det3(Mi) / D)
    return out


def fit_loglinear(pts):
    """pts: list of (ld, lt, lv). Fit lv = a + p*ld + q*lt. Returns (a,p,q) or None."""
    n = len(pts)
    if n < 6:
        return None
    S0 = n
    Sld = Slt = Sldld = Sltlt = Sldlt = 0.0
    Tv = Tld = Tlt = 0.0
    for (ld, lt, lv) in pts:
        Sld += ld; Slt += lt
        Sldld += ld * ld; Sltlt += lt * lt; Sldlt += ld * lt
        Tv += lv; Tld += ld * lv; Tlt += lt * lv
    M = [[S0, Sld, Slt], [Sld, Sldld, Sldlt], [Slt, Sldlt, Sltlt]]
    Vv = [Tv, Tld, Tlt]
    sol = solve3(M, Vv)
    if sol is None:
        return None
    return tuple(sol)


def predict(fit, ld, lt):
    a, p, q = fit
    return a + p * ld + q * lt


def envelope_em(pts, init_labels, max_iters=30, min_group=8):
    labels = list(init_labels)
    n = len(pts)
    prev = None
    fitA = fitB = None
    for _ in range(max_iters):
        groupA = [pts[i] for i in range(n) if labels[i] == 0]
        groupB = [pts[i] for i in range(n) if labels[i] == 1]
        if len(groupA) < min_group or len(groupB) < min_group:
            return None
        fitA = fit_loglinear(groupA)
        fitB = fit_loglinear(groupB)
        if fitA is None or fitB is None:
            return None
        new_labels = []
        for (ld, lt, lv) in pts:
            predA = predict(fitA, ld, lt)
            predB = predict(fitB, ld, lt)
            new_labels.append(0 if predA <= predB else 1)
        if new_labels == labels:
            break
        labels = new_labels
        if labels == prev:
            break
        prev = labels
    # final loss = sum of squared residual of min(predA,predB) vs observed lv
    loss = 0.0
    for (ld, lt, lv) in pts:
        predA = predict(fitA, ld, lt)
        predB = predict(fitB, ld, lt)
        m = predA if predA <= predB else predB
        loss += (m - lv) ** 2
    return fitA, fitB, loss


def _split_by_key(pts, key, frac):
    n = len(pts)
    order = sorted(range(n), key=key)
    cut = max(1, min(n - 1, int(round(n * frac))))
    lab = [0] * n
    for r, i in enumerate(order):
        lab[i] = 0 if r < cut else 1
    return lab


def initial_splits(pts):
    n = len(pts)
    splits = []
    keys = {
        "d": lambda i: pts[i][0],
        "T": lambda i: pts[i][1],
        "diag": lambda i: pts[i][0] + pts[i][1],
        "adiag": lambda i: pts[i][0] - pts[i][1],
    }
    # several imbalanced as well as balanced cut fractions per axis, since
    # the true crossover need not sit at the training-window median
    for key in keys.values():
        for frac in (0.5, 0.3, 0.7, 0.2, 0.8):
            splits.append(_split_by_key(pts, key, frac))
    # residual-of-single-global-fit split: sort by how far above/below the
    # single power-law line each point sits, then cut at several fractions
    # -- directly targets the opposite-edge curvature signature
    single = fit_loglinear(pts)
    if single is not None:
        resid_key = lambda i: pts[i][2] - predict(single, pts[i][0], pts[i][1])
        for frac in (0.5, 0.3, 0.7):
            splits.append(_split_by_key(pts, resid_key, frac))
    # a few random splits
    for seed in (9001, 9002, 9003, 9004):
        rng = random.Random(seed)
        lab = [rng.randint(0, 1) for _ in range(n)]
        if sum(lab) < 6 or sum(lab) > n - 6:
            lab = [(i % 2) for i in range(n)]
            rng.shuffle(lab)
        splits.append(lab)
    return splits


def best_envelope(pts):
    best = None
    for init in initial_splits(pts):
        res = envelope_em(pts, init, min_group=6)
        if res is None:
            continue
        fitA, fitB, loss = res
        if best is None or loss < best[2]:
            best = res
    return best


def grid_search_k(rows, fitA, fitB):
    """rows: list of (d,T,v). Grid-search softness k against TRAIN data."""
    ks = []
    k = 0.02
    while k <= 0.22:
        ks.append(k)
        k *= 1.25
    best_k, best_loss = ks[0], float("inf")
    for k in ks:
        loss = 0.0
        for (d, T, v) in rows:
            ld, lt = math.log(d), math.log(T)
            V1 = math.exp(predict(fitA, ld, lt))
            V2 = math.exp(predict(fitB, ld, lt))
            m = min(V1, V2)
            diff = abs(V1 - V2)
            arg = -k * diff
            if arg < -700:
                sm = m
            else:
                sm = m - math.log(1.0 + math.exp(arg)) / k
            if sm <= 0.0:
                loss += 1e6
                continue
            loss += (math.log(sm) - math.log(v)) ** 2
        if loss < best_loss:
            best_loss, best_k = loss, k
    return best_k


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    rows = []
    for _ in range(n):
        d = float(data[idx]); idx += 1
        T = float(data[idx]); idx += 1
        v = float(data[idx]); idx += 1
        rows.append((d, T, v))

    pts = [(math.log(d), math.log(T), math.log(v)) for (d, T, v) in rows]

    single_fit = fit_loglinear(pts)
    single_loss = None
    if single_fit is not None:
        single_loss = sum((predict(single_fit, ld, lt) - lv) ** 2 for (ld, lt, lv) in pts)

    result = best_envelope(pts)

    def emit_single():
        a, p, q = single_fit
        print("%.10g * powv(d,%.10g) * powv(T,%.10g)" % (math.exp(a), p, q))

    if result is None:
        if single_fit is None:
            print("%.10g" % (sum(v for (_, _, v) in rows) / len(rows)))
            return
        emit_single()
        return

    fitA, fitB, env_loss = result

    # Hypothesis test: only escalate to the two-channel envelope model if it
    # explains the training data MEANINGFULLY better than one global power
    # law -- guards against manufacturing a spurious second channel out of
    # noise when the true two-branch signal in this instance is weak.
    if single_loss is not None and env_loss > single_loss * 0.90:
        emit_single()
        return
    k = grid_search_k(rows, fitA, fitB)

    a1, p1, q1 = fitA
    a2, p2, q2 = fitB
    A1, A2 = math.exp(a1), math.exp(a2)

    v1expr = "(%.10g*powv(d,%.10g)*powv(T,%.10g))" % (A1, p1, q1)
    v2expr = "(%.10g*powv(d,%.10g)*powv(T,%.10g))" % (A2, p2, q2)
    m_expr = "minv(%s,%s)" % (v1expr, v2expr)
    diff_expr = "absv(%s-%s)" % (v1expr, v2expr)
    law = "%s-logv(1+expv(-%.10g*%s))/%.10g" % (m_expr, k, diff_expr, k)
    print(law)


if __name__ == "__main__":
    main()
