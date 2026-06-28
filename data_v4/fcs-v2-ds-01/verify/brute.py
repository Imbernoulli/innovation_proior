#!/usr/bin/env python3
# Independent slow oracle: for each query [l,r] (1-based inclusive), compute
# sum over distinct values v in a[l..r] of (count of v)^2, by direct counting.
import sys
from collections import Counter


def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    out = []
    for _ in range(q):
        l = int(next(it))
        r = int(next(it))
        # 1-based inclusive -> python slice
        sub = a[l - 1:r]
        c = Counter(sub)
        s = 0
        for v in c.values():
            s += v * v
        out.append(str(s))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
