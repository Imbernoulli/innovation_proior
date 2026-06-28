#!/usr/bin/env python3
"""
Independent brute oracle for "balanced bracket sequences of length 2n mod p".

It computes the Catalan numbers by a *different* method than the submitted
solution: the additive convolution recurrence in EXACT Python big integers,

    C[0] = 1
    C[k+1] = sum_{i=0..k} C[i] * C[k-i]

(C[k] is literally the number of balanced bracket sequences of length 2k),
and only reduces mod p at the very end. No factorials, no modular inverses,
so any bug shared with the closed-form solution would have to be a coincidence.

stdin/stdout format matches the problem statement.
"""
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    q = int(data[idx]); idx += 1
    out = []
    # Find the largest n we need, build exact Catalan numbers up to it once.
    queries = []
    max_n = 0
    for _ in range(q):
        n = int(data[idx]); p = int(data[idx + 1]); idx += 2
        queries.append((n, p))
        max_n = max(max_n, n)

    catalan = [0] * (max_n + 1)
    catalan[0] = 1
    for k in range(max_n):
        s = 0
        for i in range(k + 1):
            s += catalan[i] * catalan[k - i]
        catalan[k + 1] = s

    for (n, p) in queries:
        out.append(str(catalan[n] % p))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
