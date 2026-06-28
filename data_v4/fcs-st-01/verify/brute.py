#!/usr/bin/env python3
# Independent brute force oracle for "count distinct non-empty substrings".
# Method: enumerate every substring s[i:j] and insert into a Python set, then
# report the set size. This is O(n^2) substrings of total length O(n^3) work in
# the worst case (Python copies each slice), so it is only used on small n in
# the differential tester. It is "obviously correct": a set deduplicates by
# value, so its final size is exactly the number of distinct substrings.
import sys

def main():
    data = sys.stdin.read().split()
    s = data[0] if data else ""
    seen = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            seen.add(s[i:j])
    print(len(seen))

if __name__ == "__main__":
    main()
