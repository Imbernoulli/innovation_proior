import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    # Tiny cases to make the inclusive/exclusive day boundary and the
    # binary-search bounds genuinely exercised. Keep n small, days small (lots of
    # ties so the <= vs < boundary actually matters), and k small.
    n = random.randint(1, 9)
    k = random.randint(1, max(1, n))
    # m chosen so that sometimes it's feasible, sometimes -1.
    maxb = floor_div = n // k
    # allow m from 0-ish... but spec requires m>=1; pick 1..(maxb+1) so -1 occurs.
    hi_m = max(1, maxb + 1)
    m = random.randint(1, hi_m)

    # bloom days drawn from a tiny range to force ties (boundary stress).
    day_hi = random.randint(1, 6)
    b = [random.randint(1, day_hi) for _ in range(n)]

    out = []
    out.append(f"{n} {m} {k}")
    out.append(" ".join(map(str, b)))
    sys.stdout.write("\n".join(out) + "\n")

main()
