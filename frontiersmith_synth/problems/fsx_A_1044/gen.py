import sys, random

# Fixed field prime; ring size n always stays well below p. p is large enough that two
# unrelated targets matching the same single-seed guess by sheer coincidence (~(k-1)/p per
# try) is negligible across the whole suite -- the checker's single-seed baseline is then
# genuinely governed by structure, not by birthday-paradox luck among ~n*k candidates.
P = 1000003

def main():
    i = int(sys.argv[1])
    rng = random.Random(9100 + 37 * i)

    # difficulty ladder over 10 cases: ring size n, mixing radius r, sparsity budget s,
    # number of target portraits k (k <= n so all target positions can be made distinct).
    n = [10, 12, 14, 18, 20, 24, 28, 32, 36, 38][i - 1]
    r = 1 if i <= 5 else 2
    s = [3, 3, 4, 4, 5, 5, 6, 6, 7, 8][i - 1]
    k = [6, 7, 8, 9, 10, 10, 11, 12, 13, 14][i - 1]

    # rule coefficients c_{-r..r}, each drawn nonzero from F_p so the rule genuinely mixes cells.
    coeffs = [rng.randint(1, P - 1) for _ in range(2 * r + 1)]

    # k distinct planting/reading positions; index 0 is reserved as the "instant portrait"
    # anchor (time 0), the rest are the graded, evolved portraits.
    positions = rng.sample(range(n), k)
    pos_anchor = positions[0]

    targets = []
    # anchor: t=0, highest weight so any solver that ranks by importance handles it first.
    w_anchor = 30
    val_anchor = rng.randint(0, P - 1)
    targets.append((0, pos_anchor, val_anchor, w_anchor))

    # time ladder for the remaining k-1 evolved portraits (0 up to 1e9 across the suite;
    # later cases include genuinely huge times that forbid step-by-step simulation).
    if i <= 3:
        tlo, thi = 1, 60
    elif i <= 6:
        tlo, thi = 500, 200_000
    else:
        tlo, thi = 1_000_000, 1_000_000_000

    for idx in range(1, k):
        pos = positions[idx]
        val = rng.randint(0, P - 1)
        w = rng.randint(16, 26)
        t = rng.randint(tlo, thi)
        targets.append((t, pos, val, w))

    out = [f"{n} {P} {r} {s} {k}"]
    out.append(" ".join(map(str, coeffs)))
    for (t, pos, val, w) in targets:
        out.append(f"{t} {pos} {val} {w}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
