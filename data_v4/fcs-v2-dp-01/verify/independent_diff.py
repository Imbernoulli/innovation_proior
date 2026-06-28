#!/usr/bin/env python3
import random
import subprocess
import sys

MOD = 998244353


def norm(x):
    return x % MOD


def oracle(k, coeffs, seeds, n):
    c = [norm(x) for x in coeffs]
    seq = [norm(x) for x in seeds]
    if n < k:
        return seq[n]
    for i in range(k, n + 1):
        v = 0
        for j in range(k):
            v = (v + c[j] * seq[i - 1 - j]) % MOD
        seq.append(v)
    return seq[n]


def run_solution(binary, k, coeffs, seeds, n):
    payload = (
        str(k)
        + "\n"
        + " ".join(map(str, coeffs))
        + "\n"
        + " ".join(map(str, seeds))
        + "\n"
        + str(n)
        + "\n"
    )
    proc = subprocess.run(
        [binary],
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}: {proc.stderr}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("solution produced no output")
    return int(out)


def case(k, coeffs, seeds, n, label):
    return (k, coeffs, seeds, n, label)


def adversarial_cases():
    cases = []

    for n in [0, 1, 2, 3, 10, 50]:
        cases.append(case(2, [1, 1], [0, 1], n, "fibonacci"))

    for c in [0, 1, 2, -1, MOD - 1, MOD + 7, -(MOD + 5)]:
        for a0 in [0, 1, 3, -4, MOD + 11]:
            for n in [0, 1, 2, 5, 25, 80]:
                cases.append(case(1, [c], [a0], n, "k1_geometric"))

    for k in range(2, 18):
        coeffs = [0] * k
        seeds = [i * i - 3 * i + 5 for i in range(k)]
        for n in [0, k - 1, k, k + 1, 3 * k + 2]:
            cases.append(case(k, coeffs, seeds, n, "all_zero_coeffs"))

    for k in [2, 3, 4, 5, 8, 9, 16, 17, 31, 32, 33, 40]:
        seeds = [((i + 1) * 17) % MOD for i in range(k)]

        coeffs = [0] * k
        coeffs[0] = 1
        cases.append(case(k, coeffs, seeds, 4 * k + 7, "copy_previous"))

        coeffs = [0] * k
        coeffs[-1] = 1
        cases.append(case(k, coeffs, seeds, 5 * k + 3, "period_k"))

        coeffs = [0] * k
        coeffs[k // 2] = -1
        coeffs[-1] = MOD + 13
        cases.append(case(k, coeffs, seeds, 6 * k + 5, "sparse_signed"))

        coeffs = [MOD - 1 if i % 2 else MOD + i * 12345 for i in range(k)]
        seeds = [-(MOD + i * 19 + 7) if i % 3 == 0 else MOD + i * 29 for i in range(k)]
        cases.append(case(k, coeffs, seeds, 3 * k + 11, "wide_signed_values"))

    return cases


def random_cases(count, seed=20260628):
    rng = random.Random(seed)
    cases = []
    interesting = [0, 1, -1, MOD - 1, MOD, MOD + 1, 2 * MOD + 123, -2 * MOD - 77]
    for idx in range(count):
        k = rng.randint(1, 45)
        n_mode = rng.randrange(5)
        if n_mode == 0:
            n = rng.randrange(k)
        elif n_mode == 1:
            n = k + rng.randint(0, 5)
        elif n_mode == 2:
            n = rng.randint(0, 4 * k + 20)
        else:
            n = rng.randint(0, 450)

        coeffs = []
        seeds = []
        for _ in range(k):
            if rng.random() < 0.25:
                coeffs.append(rng.choice(interesting))
            elif rng.random() < 0.45:
                coeffs.append(0)
            else:
                coeffs.append(rng.randint(-2 * MOD, 2 * MOD))
        for _ in range(k):
            if rng.random() < 0.25:
                seeds.append(rng.choice(interesting))
            else:
                seeds.append(rng.randint(-2 * MOD, 2 * MOD))
        cases.append(case(k, coeffs, seeds, n, f"random_{idx}"))
    return cases


def main():
    if len(sys.argv) != 2:
        print("usage: independent_diff.py /path/to/compiled_solution", file=sys.stderr)
        return 2
    binary = sys.argv[1]
    cases = adversarial_cases() + random_cases(700)
    for idx, (k, coeffs, seeds, n, label) in enumerate(cases, 1):
        expected = oracle(k, coeffs, seeds, n)
        got = run_solution(binary, k, coeffs, seeds, n)
        if got != expected:
            print(f"MISMATCH case #{idx} {label}")
            print(f"k={k} n={n}")
            print("coeffs=" + " ".join(map(str, coeffs)))
            print("seeds=" + " ".join(map(str, seeds)))
            print(f"expected={expected} got={got}")
            return 1
    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
