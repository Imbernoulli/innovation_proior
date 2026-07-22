import sys, random

# ---- fixed apparatus constants (must match statement.md / verify.py) ----
P = 6                 # number of published formulations (kill-reach presets)
L_MIN, L_MAX = 10, 21 # legal range for target wavelengths (pixels)

def emit(R, seed, targets):
    out = [f"{R} {seed}"]
    for row in targets:
        out.append(" ".join(str(v) for v in row))
    sys.stdout.write("\n".join(out) + "\n")

def uniform_grid(R, rng, lo, hi):
    return [[rng.randint(lo, hi) for _ in range(R)] for _ in range(R)]

def checkerboard(R, rng, lo_val, hi_val):
    g = [[0]*R for _ in range(R)]
    for i in range(R):
        for j in range(R):
            g[i][j] = lo_val if (i + j) % 2 == 0 else hi_val
    return g

def half_split(R, rng, lo_val, hi_val):
    g = [[0]*R for _ in range(R)]
    for i in range(R):
        for j in range(R):
            g[i][j] = lo_val if j < R // 2 else hi_val
    return g

def three_band(R, rng, lo_val, mid_val, hi_val):
    g = [[0]*R for _ in range(R)]
    third = max(1, R // 3)
    for i in range(R):
        for j in range(R):
            if j < third:
                g[i][j] = lo_val
            elif j < 2 * third:
                g[i][j] = mid_val
            else:
                g[i][j] = hi_val
    return g

def asymmetric_bimodal(R, rng, lo_val, hi_val, hi_frac):
    g = [[0]*R for _ in range(R)]
    for i in range(R):
        for j in range(R):
            g[i][j] = hi_val if rng.random() < hi_frac else lo_val
    return g

def main():
    i = int(sys.argv[1])
    rng = random.Random(90200 + 17 * i)
    seed = rng.randint(1, 999999)

    if i == 1:
        R = 3
        targets = uniform_grid(R, rng, 15, 15)
    elif i == 2:
        R = 3
        targets = uniform_grid(R, rng, 13, 17)
    elif i == 3:
        R = 4
        targets = uniform_grid(R, rng, L_MIN, L_MAX)
    elif i == 4:
        R = 4
        targets = checkerboard(R, rng, 10, 21)
    elif i == 5:
        R = 4
        targets = half_split(R, rng, 10, 21)
    elif i == 6:
        R = 5
        targets = uniform_grid(R, rng, L_MIN, L_MAX)
    elif i == 7:
        R = 5
        targets = three_band(R, rng, 10, 15, 21)
    elif i == 8:
        R = 5
        targets = asymmetric_bimodal(R, rng, 10, 21, 0.35)
    elif i == 9:
        R = 6
        targets = uniform_grid(R, rng, L_MIN, L_MAX)
    else:  # i == 10
        R = 6
        targets = checkerboard(R, rng, 10, 21)

    emit(R, seed, targets)

if __name__ == "__main__":
    main()
