import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so brute (C(n,K) enumeration) stays cheap.
    n = rng.randint(0, 9)
    # Capacity small to make the <= C constraint binding and exercise zero-weight items.
    C = rng.randint(0, 8)
    # K may be 0, may exceed n (INFEASIBLE), or be a normal value.
    r = rng.random()
    if r < 0.15:
        K = 0
    elif r < 0.30:
        K = rng.randint(n + 1, n + 3)   # impossible count -> INFEASIBLE
    else:
        K = rng.randint(0, n) if n > 0 else 0

    lines = ["{} {} {}".format(n, K, C)]
    for _ in range(n):
        # Weights include 0 frequently (zero-weight "paperwork" parcels) and some > C.
        wr = rng.random()
        if wr < 0.25:
            w = 0
        elif wr < 0.45:
            w = rng.randint(C + 1, C + 4)   # heavier than capacity sometimes
        else:
            w = rng.randint(0, max(0, C))
        # Values negative, zero, positive.
        v = rng.randint(-6, 6)
        lines.append("{} {}".format(w, v))

    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
