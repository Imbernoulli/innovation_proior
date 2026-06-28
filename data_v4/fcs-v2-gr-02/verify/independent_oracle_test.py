#!/usr/bin/env python3
import collections
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOL = ROOT / "sol.fresh"


def reachable_after_deleting(n, edges, s, deleted):
    if deleted == s:
        return [False] * (n + 1)
    g = [[] for _ in range(n + 1)]
    for a, b in edges:
        if a == deleted or b == deleted:
            continue
        g[a].append(b)
    seen = [False] * (n + 1)
    q = collections.deque([s])
    seen[s] = True
    while q:
        u = q.popleft()
        for v in g[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return seen


def oracle(n, edges, s):
    base = reachable_after_deleting(n, edges, s, 0)
    dominates = [set() for _ in range(n + 1)]

    for u in range(1, n + 1):
        if not base[u]:
            continue
        if u == s:
            for v in range(1, n + 1):
                if base[v]:
                    dominates[v].add(u)
            continue

        after = reachable_after_deleting(n, edges, s, u)
        for v in range(1, n + 1):
            if base[v] and (v == u or not after[v]):
                dominates[v].add(u)

    ans = [0] * (n + 1)
    for v in range(1, n + 1):
        if not base[v] or v == s:
            ans[v] = 0
            continue
        proper = dominates[v] - {v}
        found = None
        for candidate in proper:
            if all(other in dominates[candidate] for other in proper):
                found = candidate
                break
        if found is None:
            raise AssertionError(f"no immediate dominator for node {v}")
        ans[v] = found
    return ans[1:]


def run_solution(n, edges, s):
    data = [f"{n} {len(edges)} {s}"]
    data.extend(f"{a} {b}" for a, b in edges)
    proc = subprocess.run(
        [str(SOL)],
        input="\n".join(data) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}: {proc.stderr}")
    out = proc.stdout.strip().split()
    got = [int(x) for x in out]
    if len(got) != n:
        raise RuntimeError(f"expected {n} output integers, got {len(got)}: {proc.stdout!r}")
    return got


def adversarial_cases():
    cases = []
    cases.append((1, [], 1, "single_no_loop"))
    cases.append((1, [(1, 1)], 1, "single_self_loop"))
    cases.append((6, [(1, 1), (2, 3), (3, 2), (4, 5), (5, 5), (6, 2)], 1, "all_unreachable_except_source"))
    cases.append((4, [(1, 2), (1, 3), (2, 4), (3, 4)], 1, "diamond_merge"))
    cases.append((9, [(i, i + 1) for i in range(1, 9)], 1, "deep_chain"))
    cases.append((5, [(i, j) for i in range(1, 6) for j in range(1, 6)] * 2, 3, "dense_cycle_multiedge"))
    cases.append((7, [(1, 2), (2, 3), (3, 4), (4, 2), (3, 5), (5, 6), (6, 5), (7, 3)], 1, "cycle_with_unreachable_pred"))
    cases.append((8, [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (6, 4), (2, 7), (7, 5), (8, 5)], 1, "diamond_cycle_unreachable_pred"))
    return cases


def random_cases(count, seed):
    rng = random.Random(seed)
    cases = []
    for idx in range(count):
        n = rng.randint(1, 9)
        s = rng.randint(1, n)
        max_edges = n * n + 12
        m = rng.randint(0, max_edges)
        edges = [(rng.randint(1, n), rng.randint(1, n)) for _ in range(m)]

        pattern = rng.randrange(5)
        if pattern == 0:
            edges.extend((u, u) for u in range(1, n + 1) if rng.randrange(2))
        elif pattern == 1:
            for u in range(1, n):
                if rng.randrange(2):
                    edges.append((u, u + 1))
                    edges.append((u, u + 1))
        elif pattern == 2 and n >= 4:
            a, b, c, d = rng.sample(range(1, n + 1), 4)
            edges.extend([(a, b), (a, c), (b, d), (c, d)])
            s = a
        elif pattern == 3:
            edges = [(u, v) for u in range(1, n + 1) for v in range(1, n + 1) if rng.randrange(3) != 0]
            edges.extend(rng.choice(edges) for _ in range(min(10, len(edges)))) if edges else None

        cases.append((n, edges, s, f"random_{idx}"))
    return cases


def main():
    if not SOL.exists():
        print(f"missing compiled binary: {SOL}", file=sys.stderr)
        return 2
    cases = adversarial_cases() + random_cases(400, 20260628)
    for index, (n, edges, s, name) in enumerate(cases, 1):
        expected = oracle(n, edges, s)
        got = run_solution(n, edges, s)
        if got != expected:
            print(f"Mismatch in case {index} ({name})")
            print(f"n={n} m={len(edges)} s={s}")
            print("edges:")
            for a, b in edges:
                print(a, b)
            print("expected:", " ".join(map(str, expected)))
            print("got:     ", " ".join(map(str, got)))
            return 1
    print(f"ALL_MATCH cases={len(cases)} adversarial={len(adversarial_cases())} random={len(cases) - len(adversarial_cases())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
