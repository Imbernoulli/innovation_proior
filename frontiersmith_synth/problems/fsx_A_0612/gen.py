import sys, random, math

N = 24
T = 6
ALPHA = 0.12

def diffuse(u, steps, alpha):
    n = len(u)
    cur = [row[:] for row in u]
    for _ in range(steps):
        nxt = [[0.0] * n for _ in range(n)]
        for i in range(n):
            up = cur[(i - 1) % n]; dn = cur[(i + 1) % n]; me = cur[i]
            for j in range(n):
                lap = up[j] + dn[j] + me[(j - 1) % n] + me[(j + 1) % n] - 4.0 * me[j]
                nxt[i][j] = me[j] + alpha * lap
        cur = nxt
    return cur

def energy(u):
    return sum(v * v for row in u for v in row)

def main():
    tid = int(sys.argv[1])
    rng = random.Random(9600 + tid)

    K = 4 + (tid % 4)                       # 4..7 true sources
    hf_map = {1: 0.10, 2: 0.35, 3: 0.12, 4: 0.38, 5: 0.11,
              6: 0.32, 7: 0.14, 8: 0.40, 9: 0.15, 10: 0.30}
    hf = hf_map.get(tid, 0.25)
    trap = tid in (2, 4, 6, 8)              # >=3 cases where the obvious approach is fooled

    # --- clustered true sources (overlapping blobs -> joint fit matters) ---
    ncl = 2
    centers = [(rng.randrange(3, N - 3), rng.randrange(3, N - 3)) for _ in range(ncl)]
    cells = set()
    guard = 0
    while len(cells) < K and guard < 10000:
        guard += 1
        cx, cy = centers[rng.randrange(ncl)]
        i = (cx + rng.randint(-2, 2)) % N
        j = (cy + rng.randint(-2, 2)) % N
        cells.add((i, j))
    s_true = [[0.0] * N for _ in range(N)]
    for (i, j) in cells:
        s_true[i][j] = rng.uniform(4.0, 10.0)

    c = diffuse(s_true, T, ALPHA)           # reachable low-frequency component
    L = energy(c)
    cmax = max(v for row in c for v in row)

    # --- high-frequency (unreachable) content: r minus its smoothed version ---
    r = [[rng.gauss(0.0, 1.0) for _ in range(N)] for _ in range(N)]
    smooth = diffuse(r, 4, 0.20)
    h = [[r[i][j] - smooth[i][j] for j in range(N)] for i in range(N)]
    he = energy(h)
    scale = math.sqrt((hf * L) / max(1e-12, he))
    y = [[c[i][j] + scale * h[i][j] for j in range(N)] for i in range(N)]

    # --- trap: sharp high-frequency spikes that fool raw-peak greedy ---
    if trap:
        peak_amp = 1.2 * cmax
        for _ in range(4):
            i = rng.randint(1, N - 2); j = rng.randint(1, N - 2)
            y[i][j] += peak_amp
            y[i + 1][j] -= 0.6 * peak_amp
            y[i][j + 1] -= 0.6 * peak_amp

    H = sum((y[i][j] - c[i][j]) ** 2 for i in range(N) for j in range(N))
    cost = H / 60.0

    out = []
    out.append("%d %d %.10g %.10g" % (N, T, ALPHA, cost))
    for i in range(N):
        out.append(" ".join("%.9g" % y[i][j] for j in range(N)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
