#!/usr/bin/env python3
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "verify"
SOL = VERIFY / "sol.cpp"
BIN = Path("/tmp/fcs_p2_16_sol_own")
sys.path.insert(0, str(VERIFY))
from oracle_own import brute_force


def render(case):
    n, capacity, items = case
    lines = [f"{n} {capacity}"]
    lines.extend(f"{w} {v}" for w, v in items)
    return "\n".join(lines) + "\n"


def brute(case):
    _, capacity, items = case
    return str(brute_force(items, capacity))


def run_sol(case):
    proc = subprocess.run(
        [str(BIN)],
        input=render(case),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return proc.stdout.strip()


def edge_cases():
    big = 10**9
    return [
        (0, 0, []),
        (0, 10**18, []),
        (1, 0, [(1, 5)]),
        (1, 7, [(8, 100)]),
        (1, 7, [(7, 100)]),
        (3, 10, [(6, 13), (5, 10), (5, 10)]),
        (3, 4, [(5, 10), (6, 20), (7, 30)]),
        (4, 10, [(10, 1), (9, 50), (1, 49), (5, 51)]),
        (5, 15, [(3, 4), (3, 5), (3, 6), (12, 40), (15, 41)]),
        (6, 21, [(7, 9), (7, 9), (7, 9), (8, 10), (13, 18), (21, 28)]),
        (8, 10**18, [(big, big)] * 8),
        (8, 0, [(big, big)] * 8),
        (10, 5 * big, [(big, i + 1) for i in range(10)]),
        (12, 6 * big - 1, [(big, big - i) for i in range(12)]),
        (18, 9, [(1, i + 1) for i in range(18)]),
        (18, 17, [(i + 1, 18 - i) for i in range(18)]),
    ]


def random_case(rng):
    n = rng.randint(0, 18)
    mode = rng.randrange(8)
    if mode == 0:
        items = [(rng.randint(1, 12), rng.randint(1, 40)) for _ in range(n)]
    elif mode == 1:
        items = [(rng.randint(1, 10**9), rng.randint(1, 10**9)) for _ in range(n)]
    elif mode == 2:
        items = [(rng.randint(1, 25), rng.randint(1, 25)) for _ in range(n)]
    elif mode == 3:
        items = [(rng.randint(1, 30), rng.randint(1000, 2000)) for _ in range(n)]
    elif mode == 4:
        items = [(rng.randint(100, 200), rng.randint(1, 30)) for _ in range(n)]
    else:
        items = [(rng.randint(1, 1000), rng.randint(1, 1000)) for _ in range(n)]

    total_weight = sum(w for w, _ in items)
    if n == 0:
        capacity = rng.choice([0, 1, 10**18])
    elif mode == 0:
        capacity = 0
    elif mode == 1:
        capacity = rng.choice([0, total_weight, total_weight + rng.randint(0, 1000), 10**18])
    elif mode == 2:
        capacity = max(0, rng.choice([total_weight // 2, total_weight - 1, total_weight]))
    elif mode == 3:
        pivot = rng.choice(items)[0]
        capacity = rng.choice([pivot - 1, pivot, pivot + 1])
    elif mode == 4:
        capacity = rng.randint(1, max(1, total_weight // 3 + 1))
    else:
        capacity = rng.randint(0, max(0, total_weight + 100))
    return n, capacity, items


def main():
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", "-Wall", "-Wextra", "-pedantic", str(SOL), "-o", str(BIN)],
        check=True,
    )

    rng = random.Random(20260628)
    cases = edge_cases()
    cases.extend(random_case(rng) for _ in range(700))

    for index, case in enumerate(cases, 1):
        expected = brute(case)
        actual = run_sol(case)
        if actual != expected:
            sys.stderr.write(f"Mismatch on case {index}\n")
            sys.stderr.write(render(case))
            sys.stderr.write(f"expected={expected} actual={actual}\n")
            return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
