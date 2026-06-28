#!/usr/bin/env python3
"""Independent brute-force oracle for the Josephus survivor problem.

Reads the same stdin format as sol.cpp:
    first line: q (number of queries)
    each query: n k
For each query prints the 1-indexed survivor among n people in a circle,
eliminating every k-th person, counting starting from person 1.

Two independent methods are used and cross-checked against each other so the
oracle itself is trustworthy:
  (A) direct list simulation (true to the physical process) for small n
  (B) the O(n) recurrence J(m) = (J(m-1)+k) % m, 0-indexed, +1 at the end
We default to whichever is feasible; for tiny n we assert they agree.
"""
import sys


def sim_list(n, k):
    """Physical simulation with an explicit circle. O(n*k)-ish. Returns 1-indexed."""
    people = list(range(1, n + 1))
    idx = 0
    while len(people) > 1:
        idx = (idx + k - 1) % len(people)
        people.pop(idx)
    return people[0]


def recurrence(n, k):
    """O(n) recurrence. Returns 1-indexed survivor."""
    r = 0  # J(1) = 0, 0-indexed
    for m in range(2, n + 1):
        r = (r + k) % m
    return r + 1


def solve(n, k):
    if n <= 3000:
        a = sim_list(n, k)
        b = recurrence(n, k)
        assert a == b, f"oracle methods disagree for n={n} k={k}: sim={a} rec={b}"
        return a
    # larger: recurrence only (still O(n), used by generator only for moderate n)
    return recurrence(n, k)


def main():
    data = sys.stdin.read().split()
    pos = 0
    q = int(data[pos]); pos += 1
    out = []
    for _ in range(q):
        n = int(data[pos]); k = int(data[pos + 1]); pos += 2
        out.append(str(solve(n, k)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
