#!/usr/bin/env python3
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOL = ROOT / "sol.cpp"
BIN = Path("/tmp/fcs_ds_04_sol")


def run_case(a, decoded_queries):
    last = 0
    encoded = []
    expected = []
    for typ, l, r, c in decoded_queries:
        encoded.append((typ, l ^ last, r ^ last, c ^ last))
        segment = a[l - 1:r]
        if typ == 1:
            ans = sum(1 for v in segment if v <= c)
        else:
            ans = sorted(segment)[c - 1]
        expected.append(ans)
        last = ans

    data = [f"{len(a)} {len(encoded)}", " ".join(map(str, a))]
    data.extend(f"{typ} {x} {y} {z}" for typ, x, y, z in encoded)
    proc = subprocess.run(
        [str(BIN)],
        input="\n".join(data) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}\n{proc.stderr}")
    got = [int(x) for x in proc.stdout.split()]
    if got != expected:
        return {
            "array": a,
            "decoded_queries": decoded_queries,
            "encoded_queries": encoded,
            "expected": expected,
            "got": got,
            "stderr": proc.stderr,
        }
    return None


def all_single_queries(a):
    n = len(a)
    values = sorted(set(a))
    probes = [-5, -1, 0, 1]
    if values:
        for v in values:
            probes.extend([v - 1, v, v + 1])
        probes.extend([max(values) + 10, 10**9, 10**9 + 7])
    queries = []
    for l in range(1, n + 1):
        for r in range(l, n + 1):
            for x in probes:
                queries.append((1, l, r, x))
            for k in range(1, r - l + 2):
                queries.append((2, l, r, k))
    return queries


def random_queries(rng, a, q):
    n = len(a)
    queries = []
    base_values = [-10, -1, 0, 1, 10**9, 10**9 + 1]
    base_values.extend(a)
    for v in a:
        base_values.extend([v - 1, v + 1])
    for _ in range(q):
        l = rng.randint(1, n)
        r = rng.randint(l, n)
        if rng.randrange(2) == 0:
            if rng.randrange(3) == 0:
                x = rng.choice(base_values)
            else:
                x = rng.randint(-20, 60)
            queries.append((1, l, r, x))
        else:
            k = rng.randint(1, r - l + 1)
            queries.append((2, l, r, k))
    return queries


def main():
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", str(SOL), "-o", str(BIN)],
        check=True,
    )

    rng = random.Random(20260628)
    cases = []

    adversarial_arrays = [
        [0],
        [10**9],
        [0, 0, 0, 0],
        [7, 7, 7, 7, 7],
        [0, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 0],
        [0, 10**9, 0, 10**9, 1, 999999999],
        [2, 7, 1, 8, 2, 8, 1, 8],
    ]
    for a in adversarial_arrays:
        cases.append((a, all_single_queries(a)))

    for _ in range(400):
        n = rng.randint(1, 9)
        mode = rng.randrange(5)
        if mode == 0:
            a = [rng.choice([0, 1, 2]) for _ in range(n)]
        elif mode == 1:
            start = rng.randint(0, 5)
            a = list(range(start, start + n))
        elif mode == 2:
            a = list(range(n, 0, -1))
        elif mode == 3:
            a = [rng.choice([0, 10**9, rng.randint(0, 20)]) for _ in range(n)]
        else:
            a = [rng.randint(0, 50) for _ in range(n)]
        q = rng.randint(1, 80)
        cases.append((a, random_queries(rng, a, q)))

    for i, (a, queries) in enumerate(cases, 1):
        mismatch = run_case(a, queries)
        if mismatch:
            print(f"MISMATCH case {i}", file=sys.stderr)
            print(mismatch, file=sys.stderr)
            return 1

    total_queries = sum(len(qs) for _, qs in cases)
    print(f"PASS {len(cases)} cases {total_queries} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
