#!/usr/bin/env python3
import os
import random
import subprocess
import sys

MOD = 1000000007


def oracle_by_subset_enumeration(n, edges):
    adj = [0] * n
    for u, v in edges:
        u -= 1
        v -= 1
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    ans = [0] * n
    for mask in range(1, 1 << n):
        start = mask & -mask
        seen = 0
        frontier = start
        while frontier:
            seen |= frontier
            x = frontier
            reach = 0
            while x:
                bit = x & -x
                i = bit.bit_length() - 1
                reach |= adj[i]
                x -= bit
            frontier = reach & mask & ~seen

        if seen == mask:
            x = mask
            while x:
                bit = x & -x
                i = bit.bit_length() - 1
                ans[i] = (ans[i] + 1) % MOD
                x -= bit
    return ans


def run_solution(binary, n, edges):
    data = [str(n)]
    data.extend(f"{u} {v}" for u, v in edges)
    proc = subprocess.run(
        [binary],
        input="\n".join(data) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}: {proc.stderr}")
    got = [int(x) for x in proc.stdout.split()]
    if len(got) != n:
        raise RuntimeError(f"expected {n} outputs, got {len(got)}: {proc.stdout!r}")
    return got


def relabel_and_orient(n, edges, rng):
    labels = list(range(1, n + 1))
    rng.shuffle(labels)
    mapped = [(labels[u - 1], labels[v - 1]) for u, v in edges]
    rng.shuffle(mapped)
    return [(v, u) if rng.randrange(2) else (u, v) for u, v in mapped]


def path_tree(n):
    return [(i, i + 1) for i in range(1, n)]


def star_tree(n):
    return [(1, i) for i in range(2, n + 1)]


def binary_tree(n):
    return [(i // 2, i) for i in range(2, n + 1)]


def broom_tree(n):
    if n <= 2:
        return path_tree(n)
    handle = max(2, n // 2)
    edges = path_tree(handle)
    edges.extend((handle, i) for i in range(handle + 1, n + 1))
    return edges


def double_star(n):
    if n <= 2:
        return path_tree(n)
    edges = [(1, 2)]
    for i in range(3, n + 1):
        edges.append((1 if i % 2 else 2, i))
    return edges


def random_parent_tree(n, rng):
    return [(rng.randrange(1, i), i) for i in range(2, n + 1)]


def prufer_tree(n, rng):
    if n <= 2:
        return path_tree(n)
    prufer = [rng.randrange(1, n + 1) for _ in range(n - 2)]
    deg = [1] * (n + 1)
    for x in prufer:
        deg[x] += 1
    leaves = {i for i in range(1, n + 1) if deg[i] == 1}
    edges = []
    for x in prufer:
        leaf = min(leaves)
        leaves.remove(leaf)
        edges.append((leaf, x))
        deg[leaf] -= 1
        deg[x] -= 1
        if deg[x] == 1:
            leaves.add(x)
    a, b = sorted(leaves)
    edges.append((a, b))
    return edges


def path_formula(n):
    return [((i + 1) * (n - i)) % MOD for i in range(n)]


def star_formula(n):
    if n == 1:
        return [1]
    center = pow(2, n - 1, MOD)
    leaf = (1 + pow(2, n - 2, MOD)) % MOD
    return [center] + [leaf] * (n - 1)


def check_case(binary, n, edges, expected, name):
    got = run_solution(binary, n, edges)
    if got != expected:
        print(f"MISMATCH in {name}", file=sys.stderr)
        print(f"n={n}", file=sys.stderr)
        print(f"edges={edges}", file=sys.stderr)
        print(f"expected={expected}", file=sys.stderr)
        print(f"got={got}", file=sys.stderr)
        sys.exit(1)


def main():
    binary = os.environ.get("SOL", "/tmp/fcs_dp_05_sol")
    rng = random.Random(20260628)

    adversarial = []
    for n in range(1, 15):
        adversarial.extend(
            [
                (f"path-{n}", n, path_tree(n)),
                (f"star-{n}", n, star_tree(n)),
                (f"binary-{n}", n, binary_tree(n)),
                (f"broom-{n}", n, broom_tree(n)),
                (f"double-star-{n}", n, double_star(n)),
            ]
        )
    for name, n, edges in adversarial:
        edges = relabel_and_orient(n, edges, rng)
        check_case(binary, n, edges, oracle_by_subset_enumeration(n, edges), name)

    random_cases = 500
    generators = [random_parent_tree, prufer_tree, path_tree, star_tree, broom_tree, double_star]
    for case_id in range(random_cases):
        n = rng.randrange(1, 13)
        gen = rng.choice(generators)
        if gen in (random_parent_tree, prufer_tree):
            edges = gen(n, rng)
        else:
            edges = gen(n)
        edges = relabel_and_orient(n, edges, rng)
        expected = oracle_by_subset_enumeration(n, edges)
        check_case(binary, n, edges, expected, f"random-{case_id}-{gen.__name__}")

    for n in [1, 2, 3, 5, 17, 1000, 200000]:
        check_case(binary, n, path_tree(n), path_formula(n), f"large-path-{n}")
        check_case(binary, n, star_tree(n), star_formula(n), f"large-star-{n}")

    print(f"PASS: {random_cases} random small cases, {len(adversarial)} small adversarial cases, and large path/star formula checks")


if __name__ == "__main__":
    main()
