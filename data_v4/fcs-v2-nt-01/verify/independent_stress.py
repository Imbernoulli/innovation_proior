#!/usr/bin/env python3
import random
import subprocess
import sys
from pathlib import Path

MOD = 998244353
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "sol.cpp"
BIN = ROOT / "sol_stress_bin"


def oracle(values, V, queries):
    counts = [0] * (3 * V + 1)
    n = len(values)
    for i in range(n):
        ai = values[i]
        for j in range(n):
            aij = ai + values[j]
            for k in range(n):
                counts[aij + values[k]] = (counts[aij + values[k]] + 1) % MOD
    return [counts[s] if 0 <= s < len(counts) else 0 for s in queries]


def run_case(values, V, queries):
    inp = []
    inp.append(f"{len(values)} {V}")
    inp.append(" ".join(map(str, values)))
    inp.append(str(len(queries)))
    inp.append(" ".join(map(str, queries)))
    data = "\n".join(inp) + "\n"
    proc = subprocess.run([str(BIN)], input=data, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}\nstderr:\n{proc.stderr}\ninput:\n{data}")
    got = [int(x) for x in proc.stdout.split()]
    want = oracle(values, V, queries)
    if got != want:
        raise AssertionError(
            "mismatch\n"
            f"V={V}\n"
            f"values={values}\n"
            f"queries={queries}\n"
            f"got={got}\n"
            f"want={want}\n"
            f"stdout={proc.stdout!r}\n"
        )


def adversarial_cases():
    cases = [
        ([], 0, [0]),
        ([], 5, [0, 1, 7, 15]),
        ([0], 0, [0]),
        ([0, 0, 0, 0], 0, [0]),
        ([2], 3, [0, 1, 5, 6, 9]),
        ([5, 5, 5], 5, [0, 14, 15]),
        ([4] * 12, 4, [0, 4, 8, 11, 12]),
        ([0, 7] * 8, 7, [0, 7, 14, 21]),
        ([0, 1, 1], 1, [0, 1, 2, 3]),
        ([0, 0, 3, 3, 6, 6], 6, list(range(19))),
        ([0, 9, 9, 9, 1, 8, 2, 7, 3, 6, 4, 5], 9, [0, 1, 2, 8, 9, 10, 17, 18, 26, 27]),
        ([0, 10] + [5] * 10, 10, [0, 5, 10, 15, 20, 25, 30]),
        (list(range(11)), 10, list(range(31))),
    ]
    for case in cases:
        yield case


def random_cases(seed=20260628, count=700):
    rng = random.Random(seed)
    for _ in range(count):
        V = rng.randint(0, 25)
        if V == 0:
            n = rng.randint(0, 30)
            values = [0] * n
        else:
            n = rng.randint(0, 18)
            mode = rng.randrange(5)
            if mode == 0:
                values = [rng.choice([0, V])] * n
            elif mode == 1:
                spike = rng.randint(0, V)
                values = [spike] * n
            elif mode == 2:
                values = [rng.choice([0, V, rng.randint(0, V)]) for _ in range(n)]
            else:
                values = [rng.randint(0, V) for _ in range(n)]
        q = rng.randint(0, 30)
        base_queries = [0, 3 * V]
        if V > 0:
            base_queries += [1, V, 2 * V, max(0, 3 * V - 1)]
        queries = base_queries[:]
        while len(queries) < q:
            queries.append(rng.randint(0, 3 * V))
        rng.shuffle(queries)
        queries = queries[:q]
        yield values, V, queries


def main():
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", "-Wall", "-Wextra", str(SRC), "-o", str(BIN)],
        check=True,
    )
    total = 0
    for values, V, queries in adversarial_cases():
        run_case(values, V, queries)
        total += 1
    for values, V, queries in random_cases():
        run_case(values, V, queries)
        total += 1
    print(f"PASS {total} cases")


if __name__ == "__main__":
    sys.exit(main())
