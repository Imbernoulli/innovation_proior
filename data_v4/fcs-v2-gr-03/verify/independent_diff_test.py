#!/usr/bin/env python3
import random
import subprocess
import sys
from collections import deque


SOL = "/tmp/fcs_sol"


def oracle(n, edges, queries):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)

    def reachable(src, dst, banned):
        if src == banned or dst == banned:
            return False
        seen = [False] * (n + 1)
        seen[src] = True
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                return True
            for v in adj[u]:
                if v != banned and not seen[v]:
                    seen[v] = True
                    q.append(v)
        return False

    ans = []
    for u, v in queries:
        if u == v:
            ans.append(0)
        elif not reachable(u, v, 0):
            ans.append(-1)
        else:
            cnt = 0
            for w in range(1, n + 1):
                if w != u and w != v and not reachable(u, v, w):
                    cnt += 1
            ans.append(cnt)
    return ans


def run_sol(n, edges, queries):
    data = [f"{n} {len(edges)}"]
    data += [f"{u} {v}" for u, v in edges]
    data.append(str(len(queries)))
    data += [f"{u} {v}" for u, v in queries]
    inp = "\n".join(data) + "\n"
    got = subprocess.check_output([SOL], input=inp.encode(), timeout=3)
    return [int(x) for x in got.split()]


def all_queries(n):
    return [(u, v) for u in range(1, n + 1) for v in range(1, n + 1)]


def check_case(name, n, edges, queries):
    expected = oracle(n, edges, queries)
    got = run_sol(n, edges, queries)
    if got != expected:
        print(f"mismatch in {name}", file=sys.stderr)
        print(f"n={n}", file=sys.stderr)
        print(f"edges={edges}", file=sys.stderr)
        print(f"queries={queries}", file=sys.stderr)
        print(f"expected={expected}", file=sys.stderr)
        print(f"got={got}", file=sys.stderr)
        sys.exit(1)


def adversarial_cases():
    cases = []
    cases.append(("single", 1, [], [(1, 1)]))
    cases.append(("two-isolated", 2, [], all_queries(2)))
    cases.append(("path5", 5, [(1, 2), (2, 3), (3, 4), (4, 5)], all_queries(5)))
    cases.append(("cycle4", 4, [(1, 2), (2, 3), (3, 4), (4, 1)], all_queries(4)))
    cases.append(("parallel-pendant", 3, [(1, 2), (1, 2), (2, 3)], all_queries(3)))
    cases.append(("parallel-chain", 5, [(1, 2), (1, 2), (2, 3), (3, 4), (3, 4), (4, 5)], all_queries(5)))
    cases.append(("self-loop-path", 4, [(1, 1), (1, 2), (2, 2), (2, 3), (3, 4), (4, 4)], all_queries(4)))
    cases.append(("star", 6, [(1, i) for i in range(2, 7)], all_queries(6)))
    cases.append(("two-triangles", 5, [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3)], all_queries(5)))
    cases.append(("disconnected-mix", 7, [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (6, 7), (7, 7)], all_queries(7)))
    cases.append(("root-many-children", 7, [(1, 2), (1, 3), (1, 4), (4, 5), (5, 6), (6, 4), (4, 7)], all_queries(7)))
    return cases


def random_case(rng, idx):
    n = rng.randint(1, 9)
    max_edges = min(22, n * n)
    m = rng.randint(0, max_edges)
    edges = []
    for _ in range(m):
        roll = rng.random()
        if roll < 0.15:
            u = rng.randint(1, n)
            v = u
        elif roll < 0.35 and edges:
            u, v = rng.choice(edges)
        else:
            u = rng.randint(1, n)
            v = rng.randint(1, n)
        edges.append((u, v))

    if idx % 7 == 0 and n >= 2:
        edges.extend((i, i + 1) for i in range(1, n))
    if idx % 11 == 0 and n >= 3:
        edges.extend((i, i % n + 1) for i in range(1, n + 1))
    if idx % 13 == 0 and n >= 2:
        a = rng.randint(1, n - 1)
        edges.extend([(a, a + 1), (a, a + 1)])

    queries = all_queries(n)
    for _ in range(rng.randint(0, 12)):
        queries.append((rng.randint(1, n), rng.randint(1, n)))
    rng.shuffle(queries)
    return f"random-{idx}", n, edges, queries


def main():
    rng = random.Random(20260628)
    count = 0
    for case in adversarial_cases():
        check_case(*case)
        count += 1
    for i in range(500):
        check_case(*random_case(rng, i))
        count += 1
    print(f"PASS {count} cases")


if __name__ == "__main__":
    main()
