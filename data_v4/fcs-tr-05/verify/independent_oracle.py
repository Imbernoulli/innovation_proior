#!/usr/bin/env python3
import argparse
import collections
import random
import subprocess
import sys
from pathlib import Path


MASK30 = (1 << 30) - 1


def path_nodes(n, adj, start, goal):
    parent = [-1] * (n + 1)
    parent[start] = 0
    q = collections.deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                q.append(v)
    nodes = []
    cur = goal
    while cur:
        nodes.append(cur)
        cur = parent[cur]
    return nodes


def solve_naively(case_text):
    rows = [line.split() for line in case_text.strip().splitlines()]
    n, q = map(int, rows[0])
    values = [0] + list(map(int, rows[1]))
    adj = [[] for _ in range(n + 1)]
    index = 2
    for _ in range(n - 1):
        a, b = map(int, rows[index])
        index += 1
        adj[a].append(b)
        adj[b].append(a)

    out = []
    for _ in range(q):
        op = list(map(int, rows[index]))
        index += 1
        nodes = path_nodes(n, adj, op[1], op[2])
        if op[0] == 1:
            x = op[3]
            for node in nodes:
                values[node] ^= x
        else:
            out.append(str(sum(values[node] for node in nodes)))
    return "\n".join(out) + ("\n" if out else "")


def format_case(n, values, edges, ops):
    lines = [f"{n} {len(ops)}", " ".join(map(str, values))]
    lines.extend(f"{a} {b}" for a, b in edges)
    lines.extend(" ".join(map(str, op)) for op in ops)
    return "\n".join(lines) + "\n"


def chain_edges(n):
    return [(i, i + 1) for i in range(1, n)]


def star_edges(n):
    return [(1, i) for i in range(2, n + 1)]


def deterministic_cases():
    cases = []

    cases.append(format_case(
        1,
        [7],
        [],
        [
            (2, 1, 1),
            (1, 1, 1, 0),
            (2, 1, 1),
            (1, 1, 1, MASK30),
            (2, 1, 1),
            (1, 1, 1, MASK30),
            (2, 1, 1),
        ],
    ))

    cases.append(format_case(
        2,
        [0, MASK30],
        [(1, 2)],
        [
            (2, 1, 2),
            (1, 1, 2, MASK30),
            (2, 1, 2),
            (1, 1, 1, 123456789),
            (2, 1, 1),
            (2, 2, 2),
            (1, 2, 1, 123456789),
            (1, 2, 1, 123456789),
            (2, 1, 2),
        ],
    ))

    cases.append(format_case(
        8,
        [i * 17 for i in range(8)],
        chain_edges(8),
        [
            (2, 1, 8),
            (1, 1, 8, 0),
            (2, 1, 8),
            (1, 3, 7, MASK30),
            (2, 1, 8),
            (1, 3, 7, MASK30),
            (2, 1, 8),
            (1, 4, 4, 42),
            (2, 4, 4),
            (2, 2, 6),
        ],
    ))

    cases.append(format_case(
        9,
        [MASK30, 1, 2, 3, 4, 5, 6, 7, 8],
        star_edges(9),
        [
            (2, 2, 3),
            (1, 2, 3, MASK30),
            (2, 2, 3),
            (1, 4, 8, 777),
            (2, 4, 8),
            (1, 4, 8, 777),
            (2, 4, 8),
            (1, 1, 9, 0),
            (2, 1, 9),
            (2, 5, 5),
        ],
    ))

    return cases


def random_tree(rng, n):
    shape = rng.randrange(5)
    if shape == 0:
        return chain_edges(n)
    if shape == 1:
        return star_edges(n)
    edges = []
    for v in range(2, n + 1):
        edges.append((rng.randint(1, v - 1), v))
    return edges


def random_case(rng):
    n = rng.randint(1, 35)
    q = rng.randint(1, 70)
    bit_cap = rng.choice([1, 2, 5, 13, 29, 30])
    value_max = (1 << bit_cap) - 1
    values = [rng.randint(0, value_max) for _ in range(n)]
    edges = random_tree(rng, n)
    ops = []

    for i in range(q):
        if i % 11 == 0:
            u = v = rng.randint(1, n)
        else:
            u = rng.randint(1, n)
            v = rng.randint(1, n)

        if rng.random() < 0.58:
            x = rng.choice([0, MASK30, rng.randint(0, value_max), rng.getrandbits(30)])
            ops.append((1, u, v, x))
            if rng.random() < 0.16:
                ops.append((1, u, v, x))
        else:
            ops.append((2, u, v))

    return format_case(n, values, edges, ops[:q])


def run_solution(exe, case_text):
    proc = subprocess.run(
        [str(exe)],
        input=case_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}\n{proc.stderr}")
    return proc.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--random-cases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = deterministic_cases()
    cases.extend(random_case(rng) for _ in range(args.random_cases))

    for case_no, case_text in enumerate(cases, 1):
        expected = solve_naively(case_text)
        actual = run_solution(args.exe, case_text)
        if actual != expected:
            print(f"Mismatch on case {case_no}", file=sys.stderr)
            print("Input:", file=sys.stderr)
            print(case_text, file=sys.stderr)
            print("Expected:", file=sys.stderr)
            print(expected, file=sys.stderr)
            print("Actual:", file=sys.stderr)
            print(actual, file=sys.stderr)
            return 1

    print(f"PASS {len(cases)} cases ({len(deterministic_cases())} adversarial + {args.random_cases} random)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
