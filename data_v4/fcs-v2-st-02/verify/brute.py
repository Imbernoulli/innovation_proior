#!/usr/bin/env python3
# Independent brute-force oracle for the smallest wildcard-period.
#
# Definition: p is a period iff there exists an assignment of concrete
# letters to every '?' with s[i] == s[i+p] for all 0 <= i < n-p.
# That holds iff every residue class {r, r+p, r+2p, ...} mod p has all
# its non-'?' letters equal.  We scan classes directly: O(n) per p,
# O(n^2) overall -- obviously correct, used only on small n.
import sys

def smallest_period(s):
    n = len(s)
    if n == 0:
        return 0
    for p in range(1, n + 1):
        ok = True
        for r in range(p):
            letter = None
            j = r
            while j < n:
                c = s[j]
                if c != '?':
                    if letter is None:
                        letter = c
                    elif letter != c:
                        ok = False
                        break
                j += p
            if not ok:
                break
        if ok:
            return p
    return n  # p = n always works

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    s = data[0]
    print(smallest_period(s))

if __name__ == "__main__":
    main()
