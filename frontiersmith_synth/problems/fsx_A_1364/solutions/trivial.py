# TIER: trivial
"""Reproduces the checker's own reference construction: fill Scheme A's classes
greedily in input order while completely ignoring Scheme B, then repair for
Scheme B by dropping (in the same order) whatever now breaks a B-quota."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(data[p])
        p += 1
        return v

    n = nxt()
    K1 = nxt()
    K2 = nxt()
    col1 = [0] * n
    col2 = [0] * n
    for i in range(n):
        col1[i] = nxt() - 1
        col2[i] = nxt() - 1
    cap1 = [nxt() for _ in range(K1)]
    cap2 = [nxt() for _ in range(K2)]

    cnt1 = [0] * K1
    S = []
    for i in range(n):
        c = col1[i]
        if cnt1[c] < cap1[c]:
            cnt1[c] += 1
            S.append(i)

    cnt2 = [0] * K2
    I = []
    for i in S:
        c = col2[i]
        if cnt2[c] < cap2[c]:
            cnt2[c] += 1
            I.append(i)

    print(len(I))
    print(" ".join(str(i + 1) for i in I))
    print(0)


if __name__ == "__main__":
    main()
