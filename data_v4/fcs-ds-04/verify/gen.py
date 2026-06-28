#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
# Because queries are XOR-encoded with the previous answer, the generator must
# know the answers as it emits each query. It simulates the (brute) semantics
# internally and writes the ENCODED parameters that decode back to the intended
# triple. Output is a valid stdin for sol.cpp / brute.py.
#
# All emitted queries are valid: 1 <= l <= r <= n, and for type-2  1 <= k <= r-l+1.
# Values are nonnegative so the running XOR key stays nonnegative.
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Edge mix: sometimes n == 1, sometimes a single distinct value (heavy ties).
    roll = rng.random()
    if roll < 0.10:
        n = 1
    elif roll < 0.20:
        n = rng.randint(1, 10)
    else:
        n = rng.randint(1, 14)

    q = rng.randint(1, 14)
    vhi = rng.choice([0, 1, 3, 9, 1000000000])      # small domains force ties + compression
    a = [rng.randint(0, vhi) for _ in range(n)]

    lines = [f"{n} {q}"]
    lines.append(" ".join(map(str, a)))

    last = 0
    for _ in range(q):
        typ = rng.randint(1, 2)
        l = rng.randint(1, n)
        r = rng.randint(1, n)
        if l > r:
            l, r = r, l
        if typ == 1:
            x = rng.randint(-2, vhi + 2)             # x may fall below / above the domain
            ans = sum(1 for v in a[l - 1:r] if v <= x)
            C = x
        else:
            cnt = r - l + 1
            k = rng.randint(1, cnt)
            seg = sorted(a[l - 1:r])
            ans = seg[k - 1]
            C = k
        # encode by XOR with the previous answer (>= 0)
        lines.append(f"{typ} {l ^ last} {r ^ last} {C ^ last}")
        last = ans

    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
