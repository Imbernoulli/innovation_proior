#!/usr/bin/env python3
import argparse
import random
import subprocess
from pathlib import Path


def oracle_case(n, f, queries):
    out = []
    for s, t in queries:
        seen = {}
        path = []
        v = s
        while v not in seen:
            seen[v] = len(path)
            path.append(v)
            v = f[v]
        first = seen[v]
        if t < len(path):
            out.append(path[t])
        else:
            cycle = path[first:]
            out.append(cycle[(t - first) % len(cycle)])
    return out


def run_solution(exe, n, f, queries):
    data = [str(n), " ".join(map(str, f)), str(len(queries))]
    data.extend(f"{s} {t}" for s, t in queries)
    proc = subprocess.run(
        [str(exe)],
        input="\n".join(data) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}\n{proc.stderr}")
    got = [int(x) for x in proc.stdout.split()]
    if len(got) != len(queries):
        raise RuntimeError(f"expected {len(queries)} lines, got {len(got)}")
    return got


def random_functional_graph(rng, n):
    return [rng.randrange(n) for _ in range(n)]


def cycle_with_tails(rng, n):
    nodes = list(range(n))
    rng.shuffle(nodes)
    cycle_len = rng.randint(1, n)
    cycle = nodes[:cycle_len]
    f = [0] * n
    for i, v in enumerate(cycle):
        f[v] = cycle[(i + 1) % cycle_len]
    placed = cycle[:]
    for v in nodes[cycle_len:]:
        f[v] = rng.choice(placed)
        placed.append(v)
    return f


def long_chain_to_loop(n):
    if n == 1:
        return [0]
    f = list(range(1, n)) + [n - 1]
    return f


def one_big_cycle(n):
    return [(i + 1) % n for i in range(n)]


def many_self_loops_with_leaves(n):
    f = list(range(n))
    for i in range(1, n):
        f[i] = i - 1 if i % 3 else i
    return f


def queries_for(rng, n, count):
    special_t = [
        0,
        1,
        2,
        3,
        max(0, n - 2),
        n - 1,
        n,
        n + 1,
        2 * n + 3,
        10**18,
        10**18 - 1,
    ]
    base_t = list(dict.fromkeys(list(range(n + 2)) + special_t))
    queries = []
    for s in range(n):
        for t in base_t:
            queries.append((s, t))
    while len(queries) < count:
        s = rng.randrange(n)
        if rng.random() < 0.7:
            t = rng.randrange(0, 5 * n + 20)
        else:
            t = rng.choice(special_t)
        queries.append((s, t))
    rng.shuffle(queries)
    return queries[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exe", type=Path)
    parser.add_argument("--random-cases", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = []

    for n in range(1, 18):
        cases.append((n, [0] * n, f"all_to_zero_n{n}"))
        cases.append((n, one_big_cycle(n), f"one_big_cycle_n{n}"))
        cases.append((n, long_chain_to_loop(n), f"chain_to_loop_n{n}"))
        cases.append((n, many_self_loops_with_leaves(n), f"self_loops_leaves_n{n}"))

    for i in range(args.random_cases):
        n = rng.randint(1, 35)
        if i % 3 == 0:
            f = cycle_with_tails(rng, n)
        else:
            f = random_functional_graph(rng, n)
        cases.append((n, f, f"random_{i}_n{n}"))

    for idx, (n, f, label) in enumerate(cases):
        if label.startswith("random_"):
            q = rng.randint(1, 160)
        else:
            q = n * (n + 2 + 11)
        queries = queries_for(rng, n, q)
        expected = oracle_case(n, f, queries)
        got = run_solution(args.exe, n, f, queries)
        if got != expected:
            for j, (g, e) in enumerate(zip(got, expected)):
                if g != e:
                    print(f"Mismatch in case {idx} ({label}), query {j}", flush=True)
                    print(f"n={n}", flush=True)
                    print(f"f={f}", flush=True)
                    print(f"query={queries[j]}", flush=True)
                    print(f"got={g} expected={e}", flush=True)
                    return 1
            print(f"Output length/content mismatch in case {idx} ({label})", flush=True)
            return 1

    print(f"PASS {len(cases)} cases ({args.random_cases} randomized plus adversarial)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
