# TIER: strong
# Insight: don't tune components locally. First map the target curve to its pole/cutoff
# parameter (the -3dB corner, read off the data itself), then invert the classical
# doubly-terminated ladder recurrence (g_k = 2*sin((2k-1)*pi/2N)) for ALL N components
# jointly -- this is the correct global taper, not a per-component guess. Only THEN
# quantize to the allowed discrete grid and do a small local refinement (including
# dropping marginal components) against the true cascaded response.
import sys, math


def cascade_mag(components, Z0, freqs):
    out = []
    for f in freqs:
        w = 2.0 * math.pi * f
        A, B, C, D = complex(1, 0), complex(0, 0), complex(0, 0), complex(1, 0)
        for typ, val in components:
            if typ == 'L':
                a, b, c, d = complex(1, 0), complex(0, w * val), complex(0, 0), complex(1, 0)
            else:
                a, b, c, d = complex(1, 0), complex(0, 0), complex(0, w * val), complex(1, 0)
            A, B, C, D = A * a + B * c, A * b + B * d, C * a + D * c, C * b + D * d
        denom = A * Z0 + B + C * (Z0 * Z0) + D * Z0
        mag = 0.0 if abs(denom) < 1e-300 else abs(Z0 / denom)
        out.append(mag)
    return out


DB_ERR_CAP = 12.0


def full_objective(indices, N, Z0, L_grid, C_grid, freqs, target_db, cost_per_component):
    comps = []
    populated = 0
    for i in range(1, N + 1):
        idx = indices[i - 1]
        if i % 2 == 1:
            comps.append(('L', L_grid[idx]))
        else:
            comps.append(('C', C_grid[idx]))
        if idx != 0:
            populated += 1
    mags = cascade_mag(comps, Z0, freqs)
    err = 0.0
    for m, td in zip(mags, target_db):
        db = 20.0 * math.log10(max(m, 1e-15))
        d = min(abs(db - td), DB_ERR_CAP)
        err += d * d
    err /= len(target_db)
    return err + cost_per_component * populated


def log1p10pow(x):
    """log10(1 + 10**x), overflow-safe."""
    if x > 50.0:
        return x
    return math.log10(1.0 + 10.0 ** x)


def fit_cutoff(N, freqs, target_db):
    """Least-squares fit of f_c against the maximally-flat magnitude template
    dB(f) = db0 - 10*log10(1 + (f/fc)^(2N)) over a log-spaced grid of candidate fc.
    Robust to bounded ripple/noise on the target curve (averages over ALL points,
    unlike a single threshold crossing)."""
    db0 = sum(target_db[:3]) / 3.0
    f_lo, f_hi = freqs[0], freqs[-1]
    ncand = 300
    best_fc, best_sse = None, None
    for j in range(ncand):
        fc = f_lo * ((f_hi / f_lo) ** (j / (ncand - 1)))
        sse = 0.0
        for f, td in zip(freqs, target_db):
            x = 2 * N * math.log10(f / fc)
            pred = db0 - 10.0 * log1p10pow(x)
            diff = pred - td
            sse += diff * diff
        if best_sse is None or sse < best_sse:
            best_sse, best_fc = sse, fc
    return best_fc


def nearest_idx(grid, val):
    best_i, best_d = 0, None
    for i, g in enumerate(grid):
        d = abs(g - val)
        if best_d is None or d < best_d:
            best_d, best_i = d, i
    return best_i


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    N = int(header[0]); Z0 = float(header[1]); M = int(header[2]); cost = float(header[3])

    l_tok = data[1].split()
    K_L = int(l_tok[0]); L_grid = [float(x) for x in l_tok[1:1 + K_L]]

    c_tok = data[2].split()
    K_C = int(c_tok[0]); C_grid = [float(x) for x in c_tok[1:1 + K_C]]

    freqs = [float(x) for x in data[3].split()]
    target_db = [float(x) for x in data[4].split()]

    # --- step 1: map the target curve to its pole/cutoff parameter, via a global
    # least-squares fit against the maximally-flat magnitude template (robust to the
    # bounded ripple/tolerance on the target -- NOT a single local threshold crossing) ---
    f_c_est = fit_cutoff(N, freqs, target_db)
    w_c = 2.0 * math.pi * f_c_est

    # --- step 2: invert the ladder recurrence for ALL components jointly ---
    g = [2.0 * math.sin((2 * k - 1) * math.pi / (2 * N)) for k in range(1, N + 1)]
    indices = []
    for k in range(1, N + 1):
        gk = g[k - 1]
        if k % 2 == 1:
            val = gk * Z0 / w_c
            indices.append(nearest_idx(L_grid[1:], val) + 1)  # never start by omitting
        else:
            val = gk / (Z0 * w_c)
            indices.append(nearest_idx(C_grid[1:], val) + 1)

    # --- step 3: local refinement (nearby grid points + drop-marginal-component) ---
    cur = full_objective(indices, N, Z0, L_grid, C_grid, freqs, target_db, cost)
    for _pass in range(2):
        for i in range(1, N + 1):
            grid = L_grid if i % 2 == 1 else C_grid
            cur_idx = indices[i - 1]
            candidates = {0, cur_idx}
            if cur_idx - 1 >= 0: candidates.add(cur_idx - 1)
            if cur_idx + 1 < len(grid): candidates.add(cur_idx + 1)
            best_idx, best_val = cur_idx, cur
            for cand in candidates:
                trial = list(indices); trial[i - 1] = cand
                val = full_objective(trial, N, Z0, L_grid, C_grid, freqs, target_db, cost)
                if val < best_val:
                    best_val, best_idx = val, cand
            indices[i - 1] = best_idx
            cur = best_val

    print(" ".join(map(str, indices)))


main()
