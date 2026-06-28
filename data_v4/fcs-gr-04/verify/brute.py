#!/usr/bin/env python3
"""
Independent oracle for the de Bruijn / Eulerian-path reconstruction problem.

Strategy (intentionally different from the SOTA solution):
  1. Decide INDEPENDENTLY whether *any* valid reconstruction exists. For tiny
     inputs we do this by brute-force search over orderings of the multiset of
     k-mers (overlap-chaining with backtracking) -- no Eulerian theorem reused.
  2. Read the candidate solution's stdout. If it says IMPOSSIBLE, assert that
     no reconstruction exists. Otherwise VALIDATE the printed string: it must
     have length (k-1)+m and its multiset of length-k windows must equal the
     input multiset exactly.

Usage:
  brute.py <input_file> <solution_output_file>
Prints "OK" on agreement, or "MISMATCH: ..." and exits 1.
"""
import sys
from collections import Counter


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    k = int(toks[0]); m = int(toks[1])
    kmers = toks[2:2 + m]
    return k, m, kmers


def exists_by_search(k, m, kmers):
    """Brute force: is there an ordering of the multiset of k-mers such that
    consecutive k-mers overlap by k-1 (suffix of one == prefix of next)?
    Equivalent to a valid reconstruction. Backtracking over remaining counts."""
    if m == 0:
        return True
    cnt = Counter(kmers)
    distinct = list(cnt.keys())
    # index distinct k-mers; group by their (k-1)-prefix for fast extension
    from collections import defaultdict
    by_prefix = defaultdict(list)
    for s in distinct:
        by_prefix[s[:k - 1]].append(s)

    remaining = dict(cnt)

    # Try every possible starting k-mer.
    def dfs(cur_suffix, used):
        if used == m:
            return True
        for nxt in by_prefix.get(cur_suffix, []):
            if remaining.get(nxt, 0) > 0:
                remaining[nxt] -= 1
                if dfs(nxt[1:], used + 1):
                    remaining[nxt] += 1
                    return True
                remaining[nxt] += 1
        return False

    for start in distinct:
        remaining[start] -= 1
        if dfs(start[1:], 1):
            remaining[start] += 1
            return True
        remaining[start] += 1
    return False


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    k, m, kmers = read_input(inp)
    with open(outp) as f:
        out = f.read().strip("\n")

    expect_exists = exists_by_search(k, m, kmers)

    if m == 0:
        # empty list -> empty reconstruction; solution prints empty line.
        if out.strip() == "":
            print("OK"); return
        print(f"MISMATCH: m=0 expected empty, got {out!r}"); sys.exit(1)

    if out.strip() == "IMPOSSIBLE":
        if expect_exists:
            print("MISMATCH: solution said IMPOSSIBLE but a reconstruction exists")
            sys.exit(1)
        print("OK"); return

    # Solution printed a string. It must be valid.
    if not expect_exists:
        print(f"MISMATCH: solution printed {out!r} but no reconstruction exists")
        sys.exit(1)

    s = out
    if len(s) != (k - 1) + m:
        print(f"MISMATCH: length {len(s)} != expected {(k-1)+m} for output {out!r}")
        sys.exit(1)
    windows = Counter(s[i:i + k] for i in range(len(s) - k + 1))
    if windows != Counter(kmers):
        print(f"MISMATCH: window multiset differs. out={out!r}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
