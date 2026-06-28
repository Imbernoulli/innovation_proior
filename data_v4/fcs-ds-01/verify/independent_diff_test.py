#!/usr/bin/env python3
import argparse
import random
import subprocess
from pathlib import Path


EXTREMES = [-10**9, -999_999_999, -7, -1, 0, 1, 2, 7, 999_999_999, 10**9]


def oracle(a, queries):
    return [len(set(a[l:r + 1])) for l, r in queries]


def format_case(a, queries):
    lines = [f"{len(a)} {len(queries)}", " ".join(map(str, a))]
    lines.extend(f"{l} {r}" for l, r in queries)
    return "\n".join(lines) + "\n"


def run_solution(exe, a, queries):
    proc = subprocess.run(
        [exe],
        input=format_case(a, queries),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}:\n{proc.stderr}")
    out = proc.stdout.split()
    return [int(x) for x in out]


def assert_case(exe, a, queries, label):
    want = oracle(a, queries)
    got = run_solution(exe, a, queries)
    if got != want:
        raise AssertionError(
            "\n".join(
                [
                    f"mismatch on {label}",
                    f"n={len(a)} q={len(queries)}",
                    f"a={a}",
                    f"queries={queries}",
                    f"want={want}",
                    f"got={got}",
                ]
            )
        )


def all_ranges(n):
    return [(l, r) for l in range(n) for r in range(l, n)]


def adversarial_cases():
    cases = []
    cases.append(([42], [(0, 0)]))
    cases.append(([7] * 12, all_ranges(12)))
    cases.append((list(range(12)), all_ranges(12)))
    cases.append(([-10**9, 10**9] * 7, all_ranges(14)))
    cases.append(([0, -1, 0, 1, -1, 1, 0], all_ranges(7)))
    cases.append(([5, 6], [(1, 1), (0, 0), (0, 1), (1, 1), (0, 0)]))
    cases.append(([3, 3, 4, 4, 3, 5, 5], [(0, 6)] * 10 + [(2, 3), (3, 4), (4, 4)]))
    cases.append((EXTREMES, all_ranges(len(EXTREMES))))
    for n in [2, 3, 5, 16, 31]:
        a = [i % 2 for i in range(n)]
        queries = []
        for i in range(n):
            queries.append((i, i))
            queries.append((0, i))
            queries.append((i, n - 1))
        queries.extend([(0, n - 1), (n // 2, n // 2), (1, n - 2 if n > 2 else 1)])
        cases.append((a, queries))
    return cases


def random_case(rng, case_id):
    n = rng.randint(1, 45)
    q = rng.randint(1, 90)
    shape = case_id % 6
    if shape == 0:
        a = [rng.choice([-1, 0, 1]) for _ in range(n)]
    elif shape == 1:
        a = [rng.choice(EXTREMES) for _ in range(n)]
    elif shape == 2:
        a = [rng.randint(-5, 5) for _ in range(n)]
    elif shape == 3:
        base = rng.randint(-100, 100)
        a = [base + i for i in range(n)]
    elif shape == 4:
        a = [rng.choice([case_id, -case_id, 0, 10**9, -10**9]) for _ in range(n)]
    else:
        a = [rng.randint(-10**9, 10**9) for _ in range(n)]

    queries = []
    for j in range(q):
        mode = (case_id + j) % 8
        if mode == 0:
            l = r = rng.randrange(n)
        elif mode == 1:
            l, r = 0, n - 1
        elif mode == 2:
            l = 0
            r = rng.randrange(n)
        elif mode == 3:
            l = rng.randrange(n)
            r = n - 1
        else:
            l = rng.randrange(n)
            r = rng.randrange(l, n)
        queries.append((l, r))
    return a, queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--random-cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    exe = str(Path(args.exe).resolve())
    rng = random.Random(args.seed)

    total = 0
    for i, (a, queries) in enumerate(adversarial_cases()):
        assert_case(exe, a, queries, f"adversarial-{i}")
        total += 1
    for i in range(args.random_cases):
        a, queries = random_case(rng, i)
        assert_case(exe, a, queries, f"random-{i}")
        total += 1
    print(f"PASS {total} cases ({args.random_cases} random + adversarial)")


if __name__ == "__main__":
    main()
