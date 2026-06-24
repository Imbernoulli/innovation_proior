import sys, random

# Larger / harder generator: bigger n, wider values, bands tuned to hit boundaries,
# plus structural edge cases (n=0, all-negative, all-equal arrays).
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = rng.randint(0, 6)
    if mode == 0:
        n = 0
        a = []
        vmax = 1
    elif mode == 1:
        n = rng.randint(1, 40)
        vmax = rng.choice([1, 4, 10])
        a = [-rng.randint(0, vmax) for _ in range(n)]   # all <= 0
    elif mode == 2:
        n = rng.randint(1, 40)
        c = rng.randint(-5, 5)
        a = [c] * n                                      # constant array
        vmax = abs(c) if c != 0 else 1
    else:
        n = rng.randint(1, 60)
        vmax = rng.choice([1, 3, 7, 15])
        a = [rng.randint(-vmax, vmax) for _ in range(n)]

    span = max(1, n * vmax)
    lo = rng.randint(-span - 2, span + 2)
    hi = rng.randint(-span - 2, span + 2)
    if lo > hi:
        lo, hi = hi, lo
    if rng.random() < 0.3:
        v = rng.randint(-span, span)
        lo = hi = v                                      # degenerate band L==R

    out = [f"{n} {lo} {hi}", " ".join(str(x) for x in a)]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
