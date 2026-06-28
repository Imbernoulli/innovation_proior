#!/usr/bin/env python3
"""Independent differential tester for verify/sol.cpp.

The oracle here is intentionally brute-force and contract-shaped: for small n it
enumerates every subsequence by bitmask and accepts it only if some peak p >= 1
makes it strictly increase through p and strictly decrease after p. This does
not reuse the submitted two-sided LIS/LDS recurrence.
"""

import random
import subprocess
import sys


def is_bitonic(seq):
    k = len(seq)
    if k < 2:
        return False
    for peak in range(1, k):
        ok = True
        for i in range(peak):
            if not seq[i] < seq[i + 1]:
                ok = False
                break
        if not ok:
            continue
        for i in range(peak, k - 1):
            if not seq[i] > seq[i + 1]:
                ok = False
                break
        if ok:
            return True
    return False


def brute(a):
    best = 0
    n = len(a)
    for mask in range(1, 1 << n):
        seq = [a[i] for i in range(n) if (mask >> i) & 1]
        if len(seq) > best and is_bitonic(seq):
            best = len(seq)
    return best


def run_solution(exe, a):
    data = f"{len(a)}\n{' '.join(map(str, a))}\n"
    proc = subprocess.run(
        [exe],
        input=data.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    out = proc.stdout.decode().strip()
    try:
        return int(out)
    except ValueError as exc:
        raise AssertionError(f"non-integer output {out!r} for {a}") from exc


def edge_cases():
    cases = [
        [],
        [7],
        [1, 2],
        [2, 1],
        [1, 1],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 3, 3, 3],
        [1, 11, 2, 10, 4, 5, 2, 1],
        [3, 2, 1],
        [2, 2, 3],
        [2, 3, 3, 1],
        [1, 3, 2, 2, 1],
        [-10**9, 0, 10**9, 0, -10**9],
        [0, -1, -2, 5, 4, 4, 3],
        [5, 1, 2, 3],
        [1, 5, 4, 6, 3, 2],
        [2, 1, 2, 1, 2, 1],
    ]
    for n in range(0, 8):
        cases.append(list(range(n)))
        cases.append(list(range(n, 0, -1)))
        cases.append([0] * n)
    return cases


def random_cases(seed, count):
    rng = random.Random(seed)
    modes = ["small", "dups", "wide", "sorted", "revsorted", "mountainish"]
    for _ in range(count):
        mode = rng.choice(modes)
        n = rng.randint(0, 14)
        if mode == "small":
            a = [rng.randint(-4, 4) for _ in range(n)]
        elif mode == "dups":
            a = [rng.choice([-2, -1, 0, 0, 1, 1, 2]) for _ in range(n)]
        elif mode == "wide":
            a = [rng.randint(-10**9, 10**9) for _ in range(n)]
        elif mode == "sorted":
            a = sorted(rng.randint(-8, 8) for _ in range(n))
        elif mode == "revsorted":
            a = sorted((rng.randint(-8, 8) for _ in range(n)), reverse=True)
        else:
            left = sorted(rng.randint(-8, 8) for _ in range(rng.randint(0, 7)))
            right = sorted((rng.randint(-8, 8) for _ in range(rng.randint(0, 7))), reverse=True)
            noise = [rng.randint(-8, 8) for _ in range(rng.randint(0, 4))]
            a = (left + noise + right)[:14]
        yield a


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/solution_exe", file=sys.stderr)
        return 2
    exe = sys.argv[1]
    cases = edge_cases() + list(random_cases(seed=20260628, count=800))
    for idx, a in enumerate(cases, 1):
        got = run_solution(exe, a)
        want = brute(a)
        if got != want:
            print(f"mismatch on case {idx}", file=sys.stderr)
            print(f"a = {a}", file=sys.stderr)
            print(f"got {got}, want {want}", file=sys.stderr)
            return 1
    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
