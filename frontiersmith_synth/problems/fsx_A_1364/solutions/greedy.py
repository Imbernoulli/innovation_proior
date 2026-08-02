# TIER: greedy
"""The obvious single-pass approach: scan items in input order and add each item
iff it keeps BOTH quota systems feasible right now. Never removes anything it
already picked, so it can get permanently stuck behind an early bridge item that
looked locally fine but forecloses the globally better combination. No exchange,
no certificate."""
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
    cnt2 = [0] * K2
    I = []
    for i in range(n):
        c1, c2 = col1[i], col2[i]
        if cnt1[c1] < cap1[c1] and cnt2[c2] < cap2[c2]:
            cnt1[c1] += 1
            cnt2[c2] += 1
            I.append(i)

    print(len(I))
    print(" ".join(str(i + 1) for i in I))
    print(0)


if __name__ == "__main__":
    main()
