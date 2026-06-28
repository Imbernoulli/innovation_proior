#!/usr/bin/env python3
import math
import random
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOL_CPP = ROOT / "verify" / "sol.cpp"
SOL_BIN = Path("/tmp/fcs-nt-03_independent_sol")


def compile_solution():
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", str(SOL_CPP), "-o", str(SOL_BIN)],
        check=True,
    )


def run_solution(n):
    proc = subprocess.run(
        [str(SOL_BIN)],
        input=f"{n}\n",
        text=True,
        capture_output=True,
        check=True,
    )
    return int(proc.stdout.strip())


def trial_division_prefix(max_n):
    prefix = [0] * (max_n + 1)
    running = 0
    for k in range(1, max_n + 1):
        count = 0
        root = math.isqrt(k)
        for d in range(1, root + 1):
            if k % d == 0:
                count += 1
                if d != k // d:
                    count += 1
        running += count
        prefix[k] = running
    return prefix


def exact_hyperbola_reference(n):
    if n <= 0:
        return 0
    s = math.isqrt(n)
    return 2 * sum(n // i for i in range(1, s + 1)) - s * s


def markdown_cpp_blocks_match_sol():
    sol = SOL_CPP.read_text()
    for name in ("train_answer.md", "reasoning.md"):
        text = (ROOT / name).read_text()
        start = text.index("```cpp\n") + len("```cpp\n")
        end = text.index("```", start)
        block = text[start:end]
        if block != sol:
            return False, name
    return True, None


def main():
    compile_solution()

    cases = set(range(0, 101))
    for s in range(1, 151):
        sq = s * s
        cases.update({sq - 1, sq, sq + 1})
    for base in (10, 100, 1000, 10_000, 20_000):
        cases.update({base - 1, base, base + 1})
    cases.update({9973, 9991, 12_345, 16_807, 19_999, 20_000})

    rng = random.Random(20260628)
    for _ in range(500):
        r = rng.random()
        if r < 0.20:
            cases.add(rng.randint(0, 50))
        elif r < 0.55:
            cases.add(rng.randint(0, 2_000))
        else:
            cases.add(rng.randint(0, 20_000))

    small_cases = sorted(n for n in cases if n >= 0)
    oracle = trial_division_prefix(max(small_cases))
    mismatches = []

    for n in small_cases:
        got = run_solution(n)
        want = oracle[n]
        if got != want:
            mismatches.append((n, got, want, "trial_division"))

    large_edges = [
        10**12,
        10**12 - 1,
        10**12 - 2,
        999_999 * 999_999 - 1,
        999_999 * 999_999,
        999_999 * 999_999 + 1,
        1_000_000 * 1_000_000 - 1,
        1_000_000 * 1_000_000,
    ]
    for n in large_edges:
        got = run_solution(n)
        want = exact_hyperbola_reference(n)
        if got != want:
            mismatches.append((n, got, want, "exact_isqrt_hyperbola"))

    blocks_ok, bad_name = markdown_cpp_blocks_match_sol()
    print(f"SMALL_CASES={len(small_cases)}")
    print(f"LARGE_EDGE_CASES={len(large_edges)}")
    print(f"MISMATCHES={len(mismatches)}")
    if mismatches:
        for n, got, want, source in mismatches[:20]:
            print(f"MISMATCH n={n} got={got} want={want} oracle={source}")
        raise SystemExit(1)
    if not blocks_ok:
        print(f"CPP_BLOCK_MISMATCH={bad_name}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
