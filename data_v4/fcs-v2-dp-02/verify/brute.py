#!/usr/bin/env python3
import sys

def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    B = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    items = []
    for _ in range(n):
        items.append(int(data[idx])); idx += 1
    out = []
    for _ in range(q):
        m = int(data[idx]); idx += 1
        # count items that are supersets of m: (item & m) == m
        c = 0
        for it in items:
            if (it & m) == m:
                c += 1
        out.append(str(c))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

if __name__ == "__main__":
    main()
