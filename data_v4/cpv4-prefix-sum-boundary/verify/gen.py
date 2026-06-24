import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 8)
    # Small magnitudes so window sums land in a tight, interesting band often.
    vmax = rng.choice([2, 3, 5])
    a = [rng.randint(-vmax, vmax) for _ in range(n)]

    # Choose a band [L, R] with L <= R, drawn around the realizable sum range so the
    # inclusive/exclusive boundaries get exercised (many exactly-equal-to-L or -R sums).
    # Realizable single-window sums fall within [-n*vmax, n*vmax].
    span = max(1, n * vmax)
    lo = rng.randint(-span - 1, span + 1)
    hi = rng.randint(-span - 1, span + 1)
    if lo > hi:
        lo, hi = hi, lo
    # Sometimes force a degenerate band L == R to hammer the exact-equality boundary.
    if rng.random() < 0.25:
        v = rng.randint(-span, span)
        lo = hi = v

    out = [f"{n} {lo} {hi}"]
    out.append(" ".join(str(x) for x in a))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
