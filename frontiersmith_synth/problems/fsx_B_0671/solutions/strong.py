# TIER: strong
"""
Joint integer unfolding via the monotone-zone invariant.

Key insight: the hidden law f_true(x) = a*x^b + c is MONOTONE INCREASING in
x, and each reading is f_true folded (triangle-wave) into [0, fs/2] by
Nyquist zone z = floor(f_true / (fs/2)). Because floor() of a monotone
increasing function is itself non-decreasing, the *zone sequence is
non-decreasing along sorted x* -- this is a lattice-consistency invariant
that turns "which fold happened" from a per-point guess into a globally
constrained integer-unfolding problem, regardless of how many folds any
single training point has crossed.

Algorithm:
  1. Sort rows by x.  candidate(z, r) = z*half + r if z even, else
     (z+1)*half - r  (inverts the triangle fold for a hypothesised zone z).
  2. Bootstrap the first few (smallest-x) points with a small joint zone
     search (zones are near 0 there) that minimises curvature.
  3. Walk forward: for each subsequent point, extrapolate the next true
     frequency from the last two accepted points' local slope, then pick
     the SMALLEST zone z >= previous zone whose candidate is closest to that
     prediction -- monotonicity prunes the search to a single per-point
     scan instead of a combinatorial blow-up, even across a training GAP
     that jumps many zones at once.
  4. Refit a global power law a*x^b + c to the unfolded (x, f) pairs (grid
     search over b, closed-form linear least squares for a, c given b).

This recovers the true growth rate even when the raw readings never leave
the folded baseband window, unlike a direct fit to the aliased readings.
"""
import sys, math


def candidate(z, r, half):
    if z % 2 == 0:
        return z * half + r
    return (z + 1) * half - r


def fit_power_law(xs, fs_vals, nsteps=121):
    """Grid-search b in [1.0, 2.2], closed-form (A, C) for each b. Returns
    (sse, A, b, C) of the best fit -- sse doubles as a GLOBAL consistency
    score for disambiguating unfolding hypotheses across ALL points."""
    best = None
    for step in range(nsteps):
        b = 1.0 + step * (1.2 / (nsteps - 1))  # 1.00 .. 2.20
        us = [x ** b for x in xs]
        n = len(us)
        su = sum(us); sf = sum(fs_vals)
        suu = sum(u * u for u in us)
        suf = sum(u * v for u, v in zip(us, fs_vals))
        denom = n * suu - su * su
        if abs(denom) < 1e-9:
            continue
        A = (n * suf - su * sf) / denom
        C = (sf - A * su) / n
        if A <= 0:
            continue
        sse = sum((A * u + C - v) ** 2 for u, v in zip(us, fs_vals))
        if best is None or sse < best[0]:
            best = (sse, A, b, C)
    if best is None:
        xs_l = [math.log(x) for x in xs]
        ys_l = [math.log(max(1e-6, v)) for v in fs_vals]
        n = len(xs_l)
        sx = sum(xs_l); sy = sum(ys_l)
        sxx = sum(v * v for v in xs_l)
        sxy = sum(v * w for v, w in zip(xs_l, ys_l))
        denom = n * sxx - sx * sx
        slope = (n * sxy - sx * sy) / denom if abs(denom) > 1e-9 else 1.0
        icept = (sy - slope * sx) / n
        sse = sum((math.exp(icept) * (x ** max(1.0, slope)) - v) ** 2
                   for x, v in zip(xs, fs_vals))
        return sse, math.exp(icept), max(1.0, slope), 0.0
    return best


def sequential_unfold(xs, rs, half, zmax, z0, refit_every=4, fit_steps=25):
    """Deterministic forward walk for ONE hypothesised starting zone z0.
    Anchoring the search on a single scalar (z0) instead of letting every
    point's zone float freely avoids the degenerate "arithmetic zone
    progression mimics a smooth curve" trap that a purely-local curvature
    search falls into."""
    n = len(xs)
    accepted_x = [xs[0]]
    accepted_f = [candidate(z0, rs[0], half)]
    z_prev = z0
    fit = None
    for i in range(1, n):
        if len(accepted_x) >= 3 and (fit is None or i % refit_every == 0):
            _sse, A, b, C = fit_power_law(accepted_x, accepted_f, nsteps=fit_steps)
            fit = (A, b, C)
        if fit is not None:
            A, b, C = fit
            pred = A * (xs[i] ** b) + C
        elif len(accepted_f) >= 2:
            x_a, f_a = accepted_x[-2], accepted_f[-2]
            x_b, f_b = accepted_x[-1], accepted_f[-1]
            slope = (f_b - f_a) / (x_b - x_a) if x_b - x_a > 1e-9 else 0.0
            pred = f_b + slope * (xs[i] - x_b)
        else:
            pred = accepted_f[-1]

        best_z, best_val, best_d = z_prev, None, None
        for z in range(z_prev, zmax + 1):
            val = candidate(z, rs[i], half)
            d = abs(val - pred)
            if best_d is None or d < best_d:
                best_d, best_val, best_z = d, val, z
        accepted_x.append(xs[i])
        accepted_f.append(best_val)
        z_prev = best_z

    return accepted_f


def unfold_sequence(xs, rs, half, zmax):
    """Joint integer unfolding: try a handful of hypotheses for the FIRST
    (smallest-x) point's Nyquist zone, run the deterministic forward walk
    for each, and keep the hypothesis whose resulting curve is globally
    most consistent with a single smooth power law across ALL N points --
    a lattice-consistency criterion, not a local one."""
    n = len(xs)
    if n == 0:
        return []
    Z0_CANDIDATES = range(0, min(10, zmax) + 1)
    best_seq, best_sse = None, None
    for z0 in Z0_CANDIDATES:
        seq = sequential_unfold(xs, rs, half, zmax, z0)
        sse, _A, _b, _C = fit_power_law(xs, seq, nsteps=31)
        # A constant zone shift by an EVEN number of zones only moves the
        # additive offset C, leaving the SSE numerically tied (to float
        # noise) -- always prefer the SMALLEST z0 on a near-tie (Occam:
        # the fewest folds consistent with the data), so tiny floating
        # point jitter cannot flip the pick onto a huge, wrong offset.
        if best_sse is None or sse < best_sse - abs(best_sse) * 1e-6:
            best_sse, best_seq = sse, seq
    return best_seq


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    fs = float(data[idx]); idx += 1
    fmax = float(data[idx]); idx += 1
    rows = []
    for _ in range(n):
        x = float(data[idx]); idx += 1
        r = float(data[idx]); idx += 1
        rows.append((x, r))

    rows.sort(key=lambda p: p[0])
    xs = [p[0] for p in rows]
    rs = [max(0.0, p[1]) for p in rows]

    half = fs / 2.0
    zmax = int(math.ceil(fmax / half)) + 2

    f_unfold = unfold_sequence(xs, rs, half, zmax)
    _sse, A, b, C = fit_power_law(xs, f_unfold)

    if A <= 0 or not (A == A) or not (b == b):
        A, b, C = 1.0, 1.0, 0.0

    print("%.10g * powv(x, %.10g) + %.10g" % (A, b, C))


if __name__ == "__main__":
    main()
