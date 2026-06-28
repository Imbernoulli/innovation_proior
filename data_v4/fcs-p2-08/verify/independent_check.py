#!/usr/bin/env python3
import random
import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOL = Path(os.environ.get("SOL_BIN", ROOT / "sol"))


def render_case(n, edges):
    lines = [f"{n} {len(edges)}"]
    lines.extend(f"{u} {v}" for u, v in edges)
    return "\n".join(lines) + "\n"


def brute_oracle(n, edges):
    adj = [[] for _ in range(n + 1)]
    indeg = [0] * (n + 1)
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1

    best = 0

    def walk(u, depth):
        nonlocal best
        best = max(best, depth)
        for v in adj[u]:
            walk(v, depth + 1)

    for u in range(1, n + 1):
        if indeg[u] == 0:
            walk(u, 0)
    return best


def solver_answer(inp):
    proc = subprocess.run(
        [str(SOL)],
        input=inp,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solver exited {proc.returncode}: {proc.stderr}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("solver produced no output")
    return int(out)


def random_dag(rng, n, density, allow_parallel=False):
    labels = list(range(1, n + 1))
    rng.shuffle(labels)
    edges = []
    for i, u in enumerate(labels):
        for v in labels[i + 1:]:
            if rng.random() < density:
                edges.append((u, v))
                if allow_parallel and rng.random() < 0.15:
                    edges.append((u, v))
    rng.shuffle(edges)
    return edges


def edge_cases():
    yield "single vertex", 1, []
    yield "no edges", 8, []
    yield "backward id edge", 2, [(2, 1)]
    yield "reverse id chain", 5, [(5, 4), (4, 3), (3, 2), (2, 1)]
    yield "statement sample", 9, [(1, 2), (2, 3), (2, 4), (2, 5), (1, 6), (6, 7), (7, 8), (8, 9)]
    yield "diamond", 4, [(1, 2), (1, 3), (2, 4), (3, 4)]
    yield "parallel edges", 3, [(1, 2), (1, 2), (2, 3)]
    yield "multi source tail", 7, [(3, 1), (5, 1), (1, 2), (2, 4), (6, 4), (4, 7)]
    yield "wide short vs thin long", 10, [(1, 2), (2, 3), (2, 4), (2, 5), (1, 6), (6, 7), (7, 8), (8, 9), (9, 10)]


def random_cases(count):
    rng = random.Random(20260628)
    for i in range(count):
        mode = i % 7
        if mode == 0:
            n = rng.randint(1, 5)
            edges = random_dag(rng, n, 0.5, allow_parallel=True)
        elif mode == 1:
            n = rng.randint(2, 9)
            edges = random_dag(rng, n, 0.2, allow_parallel=False)
        elif mode == 2:
            n = rng.randint(2, 9)
            edges = random_dag(rng, n, 0.75, allow_parallel=True)
        elif mode == 3:
            n = rng.randint(1, 12)
            edges = []
        elif mode == 4:
            n = rng.randint(4, 10)
            order = list(range(1, n + 1))
            rng.shuffle(order)
            edges = [(order[i], order[i + 1]) for i in range(n - 1)]
        elif mode == 5:
            layers = [[1, 2], [3, 4, 5], [6, 7], [8, 9]]
            n = 9
            edges = []
            for a, b in zip(layers, layers[1:]):
                for u in a:
                    for v in b:
                        if rng.random() < 0.55:
                            edges.append((u, v))
            rng.shuffle(edges)
        else:
            n = 10
            edges = [(1, v) for v in range(2, 7)]
            edges.extend([(1, 7), (7, 8), (8, 9), (9, 10)])
            if rng.random() < 0.5:
                edges.append((1, 7))
            rng.shuffle(edges)
        yield f"random {i}", n, edges


def check_case(name, n, edges):
    expected = brute_oracle(n, edges)
    got = solver_answer(render_case(n, edges))
    if got != expected:
        print(f"Mismatch on {name}", file=sys.stderr)
        print(render_case(n, edges), file=sys.stderr)
        print(f"expected {expected}, got {got}", file=sys.stderr)
        return False
    return True


def main():
    total = 0
    for name, n, edges in edge_cases():
        total += 1
        if not check_case(name, n, edges):
            return 1
    for name, n, edges in random_cases(500):
        total += 1
        if not check_case(name, n, edges):
            return 1
    print(f"PASS {total} cases")
    return 0


if __name__ == "__main__":
    sys.setrecursionlimit(1000000)
    raise SystemExit(main())
