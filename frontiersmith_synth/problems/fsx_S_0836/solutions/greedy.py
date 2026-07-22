# TIER: greedy
# The obvious recipe: treat the calm-day snapshots as a single rigidly
# translating pattern -- cross-correlate consecutive profiles (fractional,
# sub-sensor-spacing shifts via linear interpolation, periodic) to find the
# bulk shift that best aligns snapshot k with snapshot k+1, convert to a
# speed, and average over all consecutive pairs. Emit the DENSITY-INDEPENDENT
# flux f(rho) = c*rho (pure constant-speed advection). This is a very natural
# first idea and fits the smooth training data tolerably well, but a constant
# characteristic speed can never make characteristics converge -- it cannot
# predict where or how fast the held-out steep initial condition breaks into
# a shockwave.
import sys


def interp_shift(a, s, Mn):
    """a shifted by s grid cells (periodic, linear interpolation)."""
    out = []
    for m in range(Mn):
        pos = (m - s) % Mn
        i0 = int(pos)
        frac = pos - i0
        i1 = (i0 + 1) % Mn
        out.append(a[i0] * (1.0 - frac) + a[i1] * frac)
    return out


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0"); return
    M, K, t = int(data[0]), int(data[1]), int(data[2])
    vals = data[3:]
    rows = []
    for i in range(M * K):
        tk = float(vals[3 * i]); xm = float(vals[3 * i + 1]); rv = float(vals[3 * i + 2])
        rows.append((tk, xm, rv))

    ts = sorted(set(round(r[0], 9) for r in rows))
    xs = sorted(set(round(r[1], 9) for r in rows))
    Mn, Kn = len(xs), len(ts)
    grid = [[0.0] * Mn for _ in range(Kn)]
    tidx = {tv: i for i, tv in enumerate(ts)}
    xidx = {xv: i for i, xv in enumerate(xs)}
    for tk, xm, rv in rows:
        grid[tidx[round(tk, 9)]][xidx[round(xm, 9)]] = rv

    dx = 1.0 / Mn if Mn > 0 else 1.0
    speeds = []
    max_shift = max(2.0, Mn / 4.0)
    n_steps = 240
    for k in range(Kn - 1):
        dt = ts[k + 1] - ts[k]
        if dt <= 0:
            continue
        a, b = grid[k], grid[k + 1]
        best_s, best_e = 0.0, None
        for j in range(-n_steps, n_steps + 1):
            s = max_shift * j / n_steps
            shifted = interp_shift(a, s, Mn)
            e = sum((shifted[m] - b[m]) ** 2 for m in range(Mn))
            if best_e is None or e < best_e:
                best_e = e; best_s = s
        speeds.append(best_s * dx / dt)

    c = sum(speeds) / len(speeds) if speeds else 0.0
    print("%.6f * rho" % c)


if __name__ == "__main__":
    main()
