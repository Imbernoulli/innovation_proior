#!/usr/bin/env python3
import itertools
import random
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "verify" / "sol.cpp"
TRAIN = ROOT / "train_answer.md"
REASONING = ROOT / "reasoning.md"


def extract_cpp_block(path):
    text = path.read_text()
    blocks = re.findall(r"```cpp\n(.*?)\n```", text, flags=re.S)
    if not blocks:
        raise AssertionError(f"no cpp block found in {path}")
    return blocks[-1].strip() + "\n"


def check_markdown_blocks():
    sol = SOL.read_text()
    train_cpp = extract_cpp_block(TRAIN)
    reasoning_cpp = extract_cpp_block(REASONING)
    if train_cpp != sol:
        raise AssertionError("train_answer.md cpp block differs from verify/sol.cpp")
    if reasoning_cpp != sol:
        raise AssertionError("reasoning.md cpp block differs from verify/sol.cpp")


def brute(A, B):
    best = None
    n, m = len(A), len(B)
    for k in range(1, min(n, m) + 1):
        for ia in itertools.combinations(range(n), k):
            va = [A[i] for i in ia]
            for ib in itertools.combinations(range(m), k):
                score = sum(x * B[j] for x, j in zip(va, ib))
                if best is None or score > best:
                    best = score
    return best


def run_case(exe, A, B):
    data = f"{len(A)} {len(B)}\n"
    data += " ".join(map(str, A)) + "\n"
    data += " ".join(map(str, B)) + "\n"
    got = subprocess.check_output([str(exe)], input=data.encode(), timeout=2)
    return int(got.decode().strip())


def compile_solution(tmp):
    exe = tmp / "sol"
    subprocess.check_call([
        "g++",
        "-std=c++17",
        "-O2",
        "-pipe",
        str(SOL),
        "-o",
        str(exe),
    ])
    return exe


def edge_cases():
    cases = [
        ([5], [3]),
        ([-5], [3]),
        ([0], [-7]),
        ([2, 1, -2], [3, 0, -1]),
        ([3, -5], [-5, 3]),
        ([-3, -4], [-5, -6]),
        ([1, 2, 3], [-1]),
        ([-1], [3, 2, 1]),
        ([0, 0, 0], [-5, 0, 7]),
        ([1000, -1000, 1000], [-1000, 1000, -1000]),
        ([-1000, -1000, -1000], [-1000, -1000]),
        ([7, -8, 9, -10], [0, -1, 0, 1]),
    ]
    for a in (-3, -1, 0, 1, 3):
        for b in (-3, -1, 0, 1, 3):
            cases.append(([a], [b]))
    return cases


def random_cases(seed=20260628, count=420):
    rng = random.Random(seed)
    cases = []
    modes = ["mixed", "nonnegative", "nonpositive", "zero_heavy", "extreme"]
    for t in range(count):
        n = rng.randint(1, 8)
        m = rng.randint(1, 8)
        mode = modes[t % len(modes)]
        if mode == "mixed":
            vals = list(range(-9, 10))
        elif mode == "nonnegative":
            vals = list(range(0, 10))
        elif mode == "nonpositive":
            vals = list(range(-9, 1))
        elif mode == "zero_heavy":
            vals = [-5, -2, -1, 0, 0, 0, 1, 2, 5]
        else:
            vals = [-1000, -17, -1, 0, 1, 17, 1000]
        A = [rng.choice(vals) for _ in range(n)]
        B = [rng.choice(vals) for _ in range(m)]
        cases.append((A, B))
    return cases


def main():
    check_markdown_blocks()
    cases = edge_cases() + random_cases()
    with tempfile.TemporaryDirectory() as td:
        exe = compile_solution(Path(td))
        for idx, (A, B) in enumerate(cases, 1):
            expected = brute(A, B)
            got = run_case(exe, A, B)
            if got != expected:
                print(f"Mismatch on case {idx}")
                print(f"A={A}")
                print(f"B={B}")
                print(f"expected={expected} got={got}")
                raise SystemExit(1)

        large_cases = [
            ([1000] * 500, [1000] * 500, 500_000_000),
            ([-1000] * 500, [-1000] * 500, 500_000_000),
            ([1000] * 500, [-1000] * 500, -1_000_000),
        ]
        for A, B, expected in large_cases:
            got = run_case(exe, A, B)
            if got != expected:
                print("Mismatch on large case")
                print(f"expected={expected} got={got}")
                raise SystemExit(1)

    print(f"PASS: {len(cases)} brute-force cases plus 3 large sanity cases")


if __name__ == "__main__":
    main()
