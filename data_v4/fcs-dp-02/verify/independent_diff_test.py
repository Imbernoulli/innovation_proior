#!/usr/bin/env python3
import random
import subprocess
import sys


INF = 10**30


def block_cost(values, left, right):
    med = values[(left + right) // 2]
    total = 0
    for idx in range(left, right + 1):
        total += abs(values[idx] - med)
    return total


def oracle(values, pumps):
    values = sorted(values)
    n = len(values)
    dp = [[INF] * (n + 1) for _ in range(pumps + 1)]
    dp[0][0] = 0
    for used in range(1, pumps + 1):
        for covered in range(used, n + 1):
            best = INF
            for split in range(used - 1, covered):
                cand = dp[used - 1][split] + block_cost(values, split, covered - 1)
                if cand < best:
                    best = cand
            dp[used][covered] = best
    return dp[pumps][n]


def run_solution(exe, values, pumps):
    payload = f"{len(values)} {pumps}\n" + " ".join(map(str, values)) + "\n"
    proc = subprocess.run(
        [exe],
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}: {proc.stderr}")
    return int(proc.stdout.strip())


def edge_cases():
    cases = [
        ([0], 1),
        ([0, 10], 1),
        ([0, 10], 2),
        ([0, 5, 10], 1),
        ([0, 5, 10], 2),
        ([0, 5, 10], 3),
        ([7] * 12, 1),
        ([7] * 12, 6),
        ([7] * 12, 12),
        ([0, 0, 0, 1, 1, 1], 2),
        ([0, 0, 100, 100, 1000, 1000], 3),
        ([0, 1, 2, 1000, 1001, 1002], 2),
        ([0, 1, 2, 3, 10**9], 1),
        ([0, 1, 2, 3, 10**9], 2),
        ([0, 10**9, 10**9, 10**9], 2),
        ([0, 0, 10**9, 10**9], 1),
        ([0, 0, 10**9, 10**9], 2),
        (list(range(15)), 1),
        (list(range(15)), 7),
        (list(range(15)), 15),
        ([0, 100, 101, 102, 10000, 10001, 10002, 10003], 3),
        ([0, 0, 0, 50, 51, 52, 10**9], 3),
    ]
    for n in range(1, 18):
        cases.append((list(range(n)), max(1, n // 2)))
        cases.append(([0] * (n // 2) + [10**9] * (n - n // 2), min(n, 2)))
        cases.append(([i * i for i in range(n)], n))
    return cases


def random_cases(seed=20260628, count=500):
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        n = rng.randint(1, 18)
        pumps = rng.randint(1, n)
        style = rng.randrange(6)
        if style == 0:
            values = [rng.randint(0, 12) for _ in range(n)]
        elif style == 1:
            values = [rng.choice([0, 1, 2, 100, 101, 10**9]) for _ in range(n)]
        elif style == 2:
            start = rng.randint(0, 50)
            values = [start + rng.randint(0, 3) for _ in range(n)]
        elif style == 3:
            values = [rng.randint(0, 10**9) for _ in range(n)]
        elif style == 4:
            base = sorted(rng.sample(range(0, 300), n))
            values = [v + rng.randint(0, 1) for v in base]
        else:
            values = []
            cur = rng.randint(0, 5)
            for _idx in range(n):
                cur += rng.choice([0, 0, 1, 7, 100])
                values.append(cur)
        cases.append((sorted(values), pumps))
    return cases


def main():
    if len(sys.argv) != 2:
        print("usage: independent_diff_test.py SOLUTION_EXE", file=sys.stderr)
        return 2
    exe = sys.argv[1]
    cases = edge_cases() + random_cases()
    for case_no, (values, pumps) in enumerate(cases, 1):
        want = oracle(values, pumps)
        got = run_solution(exe, values, pumps)
        if got != want:
            print("MISMATCH")
            print(f"case_no={case_no}")
            print(f"n={len(values)} p={pumps}")
            print("values=" + " ".join(map(str, values)))
            print(f"expected={want}")
            print(f"actual={got}")
            return 1
    print(f"PASS {len(edge_cases())} adversarial/edge cases + 500 random small cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
