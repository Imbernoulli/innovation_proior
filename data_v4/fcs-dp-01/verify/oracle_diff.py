#!/usr/bin/env python3
import argparse
import functools
import random
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "verify" / "sol.cpp"


def brute_force(a, k):
    n = len(a)
    pref = [0]
    for x in a:
        pref.append(pref[-1] + x)

    @functools.lru_cache(maxsize=None)
    def dfs(pos, left):
        if left == 0:
            return 0
        if pos >= n:
            return None
        if (n - pos + 1) // 2 < left:
            return None

        best = dfs(pos + 1, left)
        for start in range(pos, n):
            for end in range(start, n):
                tail = dfs(end + 2, left - 1)
                if tail is None:
                    continue
                total = pref[end + 1] - pref[start] + tail
                if best is None or total > best:
                    best = total
        return best

    ans = dfs(0, k)
    if ans is None:
        raise ValueError("invalid test case")
    return ans


def run_candidate(binary, a, k):
    data = f"{len(a)} {k}\n" + " ".join(map(str, a)) + "\n"
    proc = subprocess.run(
        [str(binary)],
        input=data,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"candidate exited {proc.returncode}: {proc.stderr}")
    out = proc.stdout.strip()
    try:
        return int(out)
    except ValueError as exc:
        raise RuntimeError(f"non-integer candidate output: {out!r}") from exc


def adversarial_cases():
    cases = [
        ([0], 1),
        ([10**9], 1),
        ([-10**9], 1),
        ([5, -100, 5, -100, 5], 1),
        ([5, -100, 5, -100, 5], 2),
        ([5, -100, 5, -100, 5], 3),
        ([-3, -1, -4, -1, -5], 2),
        ([2, 2, 2, 2, 2, 2], 3),
        ([3, -1, 4, -1, 5, -9, 2, 6], 2),
        ([0, 0, 0, 0, 0, 0, 0], 4),
        ([10**9, -10**9, 10**9, -10**9, 10**9], 3),
        ([-10**9, 10**9, -10**9, 10**9, -10**9], 2),
        ([7, -8, 9, -10, 11, -12, 13], 4),
        ([-7, 8, -9, 10, -11, 12, -13], 3),
    ]

    for n in range(1, 13):
        cases.append(([1] * n, (n + 1) // 2))
        cases.append(([-1] * n, (n + 1) // 2))
        cases.append(([(-1) ** i * (i + 1) for i in range(n)], (n + 1) // 2))
        cases.append(([(i % 3) - 1 for i in range(n)], max(1, (n + 1) // 3)))
    return cases


def random_cases(count, seed):
    rng = random.Random(seed)
    cases = []
    palettes = [
        [-5, -4, -3, -2, -1],
        [0],
        [1, 2, 3, 4, 5],
        [-10**9, -1, 0, 1, 10**9],
        [-20, -3, -1, 0, 2, 7, 30],
    ]
    for _ in range(count):
        n = rng.randint(1, 12)
        k = rng.randint(1, (n + 1) // 2)
        mode = rng.randrange(7)
        if mode < len(palettes):
            vals = palettes[mode]
            a = [rng.choice(vals) for _ in range(n)]
        elif mode == 5:
            a = [rng.randint(-25, 25) for _ in range(n)]
        else:
            a = []
            sign = rng.choice([-1, 1])
            for _ in range(n):
                if rng.random() < 0.25:
                    sign *= -1
                a.append(sign * rng.randint(0, 50))
        cases.append((a, k))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--keep-binary", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        binary = args.keep_binary or Path(td) / "sol"
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "-pipe", str(SOL), "-o", str(binary)],
            check=True,
        )

        cases = adversarial_cases() + random_cases(args.random, args.seed)
        for idx, (a, k) in enumerate(cases, 1):
            expected = brute_force(tuple(a), k)
            actual = run_candidate(binary, a, k)
            if actual != expected:
                print("MISMATCH")
                print(f"case_index={idx}")
                print(f"n={len(a)} k={k}")
                print("a=" + " ".join(map(str, a)))
                print(f"expected={expected}")
                print(f"actual={actual}")
                raise SystemExit(1)

    print(f"PASS {len(adversarial_cases())} adversarial + {args.random} random cases")


if __name__ == "__main__":
    main()
