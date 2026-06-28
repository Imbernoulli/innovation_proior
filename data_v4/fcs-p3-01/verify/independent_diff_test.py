#!/usr/bin/env python3
import random
import subprocess
import sys


SOL = "/tmp/fcs_p3_01_sol"
MAX_N = 10**18
MAX_P = 4 * 10**18
MAX_SEED = 10**18
SMALL_MODS = [2, 3, 7, 998244353, 1000000007]
LARGE_MODS = [
    MAX_P,
    MAX_P - 1,
    MAX_P - 33,
    3999999999999999967,
    3999999999999999999,
]


def brute(n, p, f0, f1, f2):
    a, b, c = f0 % p, f1 % p, f2 % p
    if n == 0:
        return a
    if n == 1:
        return b
    if n == 2:
        return c
    for _ in range(3, n + 1):
        a, b, c = b, c, (a + b + c) % p
    return c


def mat_mul(a, b, p):
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) % p for j in range(3)]
        for i in range(3)
    ]


def mat_vec_mul(a, v, p):
    return [sum(a[i][j] * v[j] for j in range(3)) % p for i in range(3)]


def big_ref(n, p, f0, f1, f2):
    state = [f0 % p, f1 % p, f2 % p]
    if n < 3:
        return state[n]

    # Independent orientation from sol.cpp:
    # [f(k), f(k+1), f(k+2)] -> [f(k+1), f(k+2), f(k)+f(k+1)+f(k+2)].
    trans = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ]
    acc = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    e = n
    while e:
        if e & 1:
            acc = mat_mul(acc, trans, p)
        trans = mat_mul(trans, trans, p)
        e >>= 1
    return mat_vec_mul(acc, state, p)[0]


def edge_cases():
    cases = []
    seed_sets = [
        (0, 0, 0),
        (1, 1, 1),
        (0, 1, 1),
        (MAX_SEED, MAX_SEED - 1, MAX_SEED - 2),
        (MAX_SEED, MAX_SEED, MAX_SEED),
    ]
    for p in SMALL_MODS + LARGE_MODS:
        for seeds in seed_sets:
            for n in [0, 1, 2, 3, 4, 5, 10, 50, 200, MAX_N - 1, MAX_N]:
                cases.append((n, p, *seeds))

    for n in range(0, 80):
        cases.append((n, 1000000007, MAX_SEED - n, MAX_SEED - 2 * n, MAX_SEED - 3 * n))
        cases.append((n, MAX_P, MAX_SEED - n, MAX_SEED - n - 1, MAX_SEED - n - 2))
    return cases


def random_cases():
    rng = random.Random(20260628)
    cases = []
    for _ in range(500):
        p = rng.choice(SMALL_MODS + LARGE_MODS + [rng.randrange(2, MAX_P + 1)])
        if rng.random() < 0.55:
            n = rng.randrange(0, 5000)
        else:
            n = rng.choice([MAX_N, MAX_N - rng.randrange(0, 1000000), rng.randrange(10**12, MAX_N + 1)])
        f0 = rng.randrange(0, MAX_SEED + 1)
        f1 = rng.randrange(0, MAX_SEED + 1)
        f2 = rng.randrange(0, MAX_SEED + 1)
        cases.append((n, p, f0, f1, f2))
    return cases


def main():
    cases = edge_cases() + random_cases()
    expected = []
    for case in cases:
        n, p, f0, f1, f2 = case
        ref = big_ref(n, p, f0, f1, f2)
        if n <= 5000:
            b = brute(n, p, f0, f1, f2)
            if b != ref:
                print("oracle mismatch", case, b, ref, file=sys.stderr)
                return 2
        expected.append(ref)

    payload = str(len(cases)) + "\n" + "\n".join(" ".join(map(str, c)) for c in cases) + "\n"
    proc = subprocess.run([SOL], input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    got = [int(x) for x in proc.stdout.split()]
    if len(got) != len(expected):
        print(f"output length mismatch: got {len(got)}, expected {len(expected)}", file=sys.stderr)
        return 1
    for idx, (case, g, e) in enumerate(zip(cases, got, expected)):
        if g != e:
            print(f"mismatch at case {idx}: {case}", file=sys.stderr)
            print(f"got {g}, expected {e}", file=sys.stderr)
            return 1
    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
