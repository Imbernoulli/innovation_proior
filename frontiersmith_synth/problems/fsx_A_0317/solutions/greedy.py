# TIER: greedy
# Mian-Chowla sum-greedy (B_2^+ packing):
# scan posts 0,1,2,... and keep a post iff it introduces NO repeated pairwise
# sum with the posts already chosen (a self-sum 2*post counts too). This packs
# many distinct sums; if the rail runs out before n boxes are placed, fill the
# remaining boxes with the smallest still-unused posts.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); M = int(data[1])

    A = []
    sums = set()
    for c in range(0, M + 1):
        if len(A) >= n:
            break
        new = set()
        ok = True
        # candidate c must not collide with any existing sum, including 2c and
        # a+c for each existing a.
        for a in A:
            v = a + c
            if v in sums or v in new:
                ok = False
                break
            new.add(v)
        if ok:
            v = c + c
            if v in sums or v in new:
                ok = False
            else:
                new.add(v)
        if ok:
            A.append(c)
            sums |= new

    if len(A) < n:
        used = set(A)
        c = 0
        while len(A) < n and c <= M:
            if c not in used:
                A.append(c)
                used.add(c)
            c += 1

    print(" ".join(str(x) for x in A[:n]))


if __name__ == "__main__":
    main()
