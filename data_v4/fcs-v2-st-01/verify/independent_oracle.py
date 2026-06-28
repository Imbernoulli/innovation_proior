#!/usr/bin/env python3
import os
import random
import subprocess
import sys
import tempfile


ROOT = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.join(ROOT, "sol.cpp")


def cmp_substrings(s, l1, len1, l2, len2):
    i = l1 - 1
    j = l2 - 1
    limit = len1 if len1 < len2 else len2
    for k in range(limit):
        a = s[i + k]
        b = s[j + k]
        if a < b:
            return -1
        if a > b:
            return 1
    if len1 < len2:
        return -1
    if len1 > len2:
        return 1
    return 0


def all_queries(n):
    queries = []
    for l1 in range(1, n + 1):
        for len1 in range(1, n - l1 + 2):
            for l2 in range(1, n + 1):
                for len2 in range(1, n - l2 + 2):
                    queries.append((l1, len1, l2, len2))
    return queries


def focused_queries(s, rng):
    n = len(s)
    queries = []

    for l in range(1, n + 1):
        queries.append((l, 1, l, 1))
        queries.append((l, n - l + 1, l, n - l + 1))

    for l1 in range(1, n + 1):
        for l2 in range(1, n + 1):
            max_len = min(n - l1 + 1, n - l2 + 1)
            queries.append((l1, 1, l2, 1))
            queries.append((l1, max_len, l2, max_len))
            if max_len > 1:
                queries.append((l1, max_len - 1, l2, max_len))
                queries.append((l1, max_len, l2, max_len - 1))

    for _ in range(80):
        l1 = rng.randint(1, n)
        l2 = rng.randint(1, n)
        len1 = rng.randint(1, n - l1 + 1)
        len2 = rng.randint(1, n - l2 + 1)
        queries.append((l1, len1, l2, len2))

    return queries


def run_case(exe, s, queries):
    data = [s, str(len(queries))]
    data.extend("%d %d %d %d" % q for q in queries)
    proc = subprocess.run(
        [exe],
        input="\n".join(data) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("solution exited with %d\n%s" % (proc.returncode, proc.stderr))
    got = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(got) != len(queries):
        raise RuntimeError("expected %d output lines, got %d" % (len(queries), len(got)))
    want = [str(cmp_substrings(s, *q)) for q in queries]
    for idx, (g, w) in enumerate(zip(got, want)):
        if g != w:
            return idx, g, w
    return None


def build_cases():
    rng = random.Random(20260628)
    cases = []

    adversarial = [
        "a",
        "aa",
        "ab",
        "ba",
        "aaa",
        "aba",
        "banana",
        "aaaaaa",
        "abababab",
        "abcabcabc",
        "zzzzzzzz",
        "zyxwvuts",
        "aabaaaab",
        "mississippi",
        "abcdabc",
    ]
    cases.extend((s, "adversarial") for s in adversarial)

    alphabet_sets = ["a", "ab", "abc", "abcd", "abcdefghijklmnopqrstuvwxyz"]
    for case_id in range(500):
        n = rng.randint(1, 14)
        alphabet = rng.choice(alphabet_sets)
        s = "".join(rng.choice(alphabet) for _ in range(n))
        cases.append((s, "random-%03d" % case_id))

    return cases


def main():
    exe_arg = sys.argv[1] if len(sys.argv) > 1 else None
    with tempfile.TemporaryDirectory() as td:
        exe = exe_arg or os.path.join(td, "sol")
        if exe_arg is None:
            subprocess.run(
                ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic", SOL, "-o", exe],
                check=True,
            )

        random_count = 0
        query_count = 0
        for s, label in build_cases():
            n = len(s)
            if label.startswith("random"):
                random_count += 1
            queries = all_queries(n) if n <= 6 else focused_queries(s, random.Random(label))
            query_count += len(queries)
            mismatch = run_case(exe, s, queries)
            if mismatch is not None:
                idx, got, want = mismatch
                print("MISMATCH %s" % label)
                print("s=%r" % s)
                print("query=%s" % (queries[idx],))
                print("got=%s want=%s" % (got, want))
                return 1

        print("PASS %d random cases, %d total cases, %d queries" % (
            random_count,
            len(build_cases()),
            query_count,
        ))
        return 0


if __name__ == "__main__":
    sys.exit(main())
