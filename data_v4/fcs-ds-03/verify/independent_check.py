#!/usr/bin/env python3
import random
import subprocess
import sys


def oracle(case):
    a, b, c = case
    n = len(a)
    dp = [0] + [None] * n
    for i in range(1, n + 1):
        x = a[i - 1]
        best = None
        for j in range(i):
            val = dp[j] + b[j] * x
            if best is None or val < best:
                best = val
        dp[i] = c[i - 1] + best
    return dp[n] if n else 0


def render(case):
    a, b, c = case
    n = len(a)
    lines = [str(n)]
    if n:
        lines.append(" ".join(map(str, a)))
        lines.append(" ".join(map(str, b)))
        lines.append(" ".join(map(str, c)))
    return "\n".join(lines) + "\n"


def run_solution(exe, case):
    data = render(case)
    got = subprocess.check_output([exe], input=data.encode(), timeout=5)
    return int(got.decode().strip() or "0")


def adversarial_cases():
    cases = [
        ([], [], []),
        ([0], [0], [0]),
        ([5], [-3], [7]),
        ([-5], [3], [-7]),
        ([0, 0, 0, 0], [5, -5, 5, -5], [1, -1, 1, -1]),
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [0, 0, 0, 0, 0]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5], [0, 0, 0, 0, 0]),
        ([-5, -4, -3, -2, -1], [5, -4, 3, -2, 1], [9, -8, 7, -6, 5]),
        ([10, -10, 10, -10, 10], [-10, 10, -10, 10, -10], [3, 3, -3, -3, 0]),
        ([7, 7, 7, 7, 7, 7], [6, 5, 4, 3, 2, 1], [-1, -2, -3, -4, -5, -6]),
        ([-7, -7, -7, -7, -7, -7], [-6, -5, -4, -3, -2, -1], [1, 2, 3, 4, 5, 6]),
        ([1000000, -1000000, 999999, -999999], [1000000, -1000000, 999999, -999999], [1, -1, 2, -2]),
    ]
    return cases


def random_cases(count, seed):
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        mode = rng.randrange(8)
        if mode == 0:
            n = rng.randrange(0, 2)
            lo, hi = -1, 1
        elif mode == 1:
            n = rng.randrange(1, 8)
            lo, hi = -5, 5
        elif mode == 2:
            n = rng.randrange(1, 14)
            lo, hi = -30, 30
        elif mode == 3:
            n = rng.randrange(1, 18)
            vals = [rng.randrange(-3, 4) for _ in range(3)]
            a = [rng.choice(vals) for _ in range(n)]
            b = [rng.randrange(-20, 21) for _ in range(n)]
            c = [rng.randrange(-20, 21) for _ in range(n)]
            cases.append((a, b, c))
            continue
        elif mode == 4:
            n = rng.randrange(2, 20)
            a = sorted(rng.randrange(-50, 51) for _ in range(n))
            b = sorted((rng.randrange(-50, 51) for _ in range(n)), reverse=True)
            c = [rng.randrange(-50, 51) for _ in range(n)]
            cases.append((a, b, c))
            continue
        elif mode == 5:
            n = rng.randrange(2, 20)
            a = [(-1) ** i * rng.randrange(0, 50) for i in range(n)]
            b = [(-1) ** (i + 1) * rng.randrange(0, 50) for i in range(n)]
            c = [rng.randrange(-50, 51) for _ in range(n)]
            cases.append((a, b, c))
            continue
        elif mode == 6:
            n = rng.randrange(1, 12)
            lo, hi = -1000, 1000
        else:
            n = rng.randrange(1, 30)
            lo, hi = -100, 100

        a = [rng.randrange(lo, hi + 1) for _ in range(n)]
        b = [rng.randrange(lo, hi + 1) for _ in range(n)]
        c = [rng.randrange(lo, hi + 1) for _ in range(n)]
        cases.append((a, b, c))
    return cases


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SOLUTION_EXE")
    exe = sys.argv[1]
    cases = adversarial_cases() + random_cases(1000, 20260628)
    for idx, case in enumerate(cases):
        expected = oracle(case)
        actual = run_solution(exe, case)
        if actual != expected:
            sys.stderr.write(f"Mismatch on case {idx}\n")
            sys.stderr.write(render(case))
            sys.stderr.write(f"expected {expected}, got {actual}\n")
            return 1
    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
