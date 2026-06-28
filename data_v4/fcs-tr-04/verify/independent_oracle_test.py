#!/usr/bin/env python3
import collections
import random
import subprocess
import sys


def oracle(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u - 1].append(v - 1)
        adj[v - 1].append(u - 1)

    answer = []
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = collections.deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        answer.append(sum(dist))
    return answer


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
    out = [int(x) for x in proc.stdout.split()]
    return out


def relabel_and_shuffle(rng, n, edges):
    perm = list(range(1, n + 1))
    rng.shuffle(perm)

    mapped = []
    for u, v in edges:
        a, b = perm[u - 1], perm[v - 1]
        if rng.randrange(2):
            a, b = b, a
        mapped.append((a, b))
    rng.shuffle(mapped)
    return mapped


def path_tree(n):
    return [(i, i + 1) for i in range(1, n)]


def star_tree(n, center=1):
    return [(center, v) for v in range(1, n + 1) if v != center]


def balanced_binary_tree(n):
    return [(i // 2, i) for i in range(2, n + 1)]


def random_attach_tree(rng, n):
    return [(rng.randint(1, v - 1), v) for v in range(2, n + 1)]


def broom_tree(n, handle):
    edges = path_tree(handle)
    hub = handle
    edges.extend((hub, v) for v in range(handle + 1, n + 1))
    return edges


def double_star_tree(n):
    if n == 1:
        return []
    edges = [(1, 2)]
    for v in range(3, n + 1):
        edges.append((1 if v % 2 else 2, v))
    return edges


def adversarial_cases():
    cases = [(1, [])]
    for n in range(2, 35):
        cases.append((n, path_tree(n)))
        cases.append((n, list(reversed(path_tree(n)))))
        cases.append((n, star_tree(n, 1)))
        cases.append((n, star_tree(n, n)))
        cases.append((n, balanced_binary_tree(n)))
        cases.append((n, broom_tree(n, max(1, n // 2))))
        cases.append((n, double_star_tree(n)))
    return cases


def random_cases(rng, count):
    cases = []
    for _ in range(count):
        n = rng.randint(1, 45)
        shape = rng.randrange(6)
        if n == 1:
            edges = []
        elif shape == 0:
            edges = path_tree(n)
        elif shape == 1:
            edges = star_tree(n, rng.randint(1, n))
        elif shape == 2:
            edges = balanced_binary_tree(n)
        elif shape == 3:
            edges = broom_tree(n, rng.randint(1, n))
        elif shape == 4:
            edges = double_star_tree(n)
        else:
            edges = random_attach_tree(rng, n)
        cases.append((n, relabel_and_shuffle(rng, n, edges)))
    return cases


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/compiled_solution", file=sys.stderr)
        return 2

    binary = sys.argv[1]
    rng = random.Random(20260628)
    random_count = 600
    cases = adversarial_cases() + random_cases(rng, random_count)

    for index, (n, edges) in enumerate(cases, 1):
        expected = oracle(n, edges)
        actual = run_solution(binary, n, edges)
        if actual != expected:
            print("MISMATCH", file=sys.stderr)
            print(f"case_index={index}", file=sys.stderr)
            print(f"n={n}", file=sys.stderr)
            print("edges:", file=sys.stderr)
            for u, v in edges:
                print(u, v, file=sys.stderr)
            print(f"expected={expected}", file=sys.stderr)
            print(f"actual={actual}", file=sys.stderr)
            return 1

    print(f"PASS {len(cases)} cases ({random_count} random small + {len(cases) - random_count} adversarial)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
