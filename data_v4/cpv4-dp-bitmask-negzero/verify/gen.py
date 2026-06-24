import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Keep cases tiny so the brute force (which enumerates (m+1)^n decisions)
    # stays fast.  n in [0..5], m in [1..4].
    mode = rng.randint(0, 4)
    n = rng.randint(0, 5)
    m = rng.randint(1, 4)

    rows = []
    for _ in range(n):
        row = []
        for _ in range(m):
            if mode == 0:
                v = rng.randint(-9, 9)          # mixed signs incl. zero
            elif mode == 1:
                v = rng.randint(-9, -1)         # all negative
            elif mode == 2:
                v = 0                            # all zero
            elif mode == 3:
                v = rng.randint(1, 9)            # all positive
            else:
                v = rng.choice([-9, -5, 0, 0, 3, 9])  # zeros sprinkled in
            row.append(v)
        rows.append(row)

    out = [f"{n} {m}"]
    for row in rows:
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")

main()
