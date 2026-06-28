#!/usr/bin/env python3
from pathlib import Path
import random
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "verify" / "sol.cpp"
TRAIN = ROOT / "train_answer.md"
REASONING = ROOT / "reasoning.md"


def cpp_blocks(path):
    text = path.read_text()
    blocks = []
    pos = 0
    marker = "```cpp"
    while True:
        start = text.find(marker, pos)
        if start == -1:
            break
        start += len(marker)
        if start < len(text) and text[start] == "\n":
            start += 1
        end = text.find("```", start)
        if end == -1:
            raise RuntimeError(f"unterminated cpp block in {path}")
        blocks.append(text[start:end].rstrip("\n"))
        pos = end + 3
    return blocks


def assert_synced():
    sol = SOL.read_text().rstrip("\n")
    for path in (TRAIN, REASONING):
        blocks = cpp_blocks(path)
        if len(blocks) != 1:
            raise AssertionError(f"{path} has {len(blocks)} cpp blocks, expected 1")
        if blocks[0] != sol:
            raise AssertionError(f"{path} cpp block differs from verify/sol.cpp")


def brute_count(n, k, p):
    ans = 0
    for mask in range(1 << n):
        run = 0
        ok = True
        for i in range(n):
            if (mask >> i) & 1:
                run += 1
                if run >= k:
                    ok = False
                    break
            else:
                run = 0
        if ok:
            ans += 1
    return ans % p


def build_cases():
    moduli = [2, 3, 4, 5, 7, 8, 9, 10, 97, 100, 1_000_000_000, 1_000_000_007]
    cases = []

    # Hand-picked edges: empty string, k=1, N<k, N=k, N=k+1, small/composite p.
    for p in moduli:
        for k in range(1, 17):
            for n in sorted({0, 1, max(0, k - 1), k, min(18, k + 1)}):
                if n <= 18:
                    cases.append((n, k, p))
        for n in range(0, 19):
            cases.append((n, 50, p))

    rng = random.Random(20260628)
    while len(cases) < 700:
        # Keep n small enough for literal enumeration, but concentrate around
        # boundaries where the recurrence switches on.
        k = rng.randint(1, 50)
        if rng.random() < 0.65:
            n = rng.choice([0, 1, max(0, k - 2), max(0, k - 1), k, k + 1])
            n = min(n, 18)
        else:
            n = rng.randint(0, 18)
        p = rng.choice(moduli + [rng.randint(2, 10_000)])
        cases.append((n, k, p))

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for case in cases:
        if case not in seen:
            unique.append(case)
            seen.add(case)
    return unique


def compile_solution(exe):
    subprocess.run(
        ["g++", "-std=c++17", "-O2", str(SOL), "-o", str(exe)],
        check=True,
    )


def run_solution(exe, cases):
    payload = [str(len(cases))]
    payload += [f"{n} {k} {p}" for n, k, p in cases]
    proc = subprocess.run(
        [str(exe)],
        input="\n".join(payload) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return [int(x) for x in proc.stdout.split()]


def main():
    assert_synced()
    cases = build_cases()
    with tempfile.TemporaryDirectory() as td:
        exe = Path(td) / "sol"
        compile_solution(exe)
        got = run_solution(exe, cases)

    if len(got) != len(cases):
        raise AssertionError(f"solver returned {len(got)} lines for {len(cases)} cases")

    for idx, ((n, k, p), actual) in enumerate(zip(cases, got), 1):
        expected = brute_count(n, k, p)
        if actual != expected:
            print(
                f"Mismatch at case {idx}: N={n} k={k} p={p}; "
                f"expected {expected}, got {actual}",
                file=sys.stderr,
            )
            return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
