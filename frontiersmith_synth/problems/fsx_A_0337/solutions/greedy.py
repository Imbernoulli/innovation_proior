# TIER: greedy
# Mian-Chowla greedy Sidon packing: repeatedly add the smallest slot that keeps all
# pairwise sums distinct. On the short rail (M = 9n) the greedy set runs out of room
# before reaching n, so we pad with the remaining unused slots. Beats the block, but a
# purely-greedy prefix leaves collisions on the table compared with a global search.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); M = int(toks[1])

    A = [0]
    sums = {0}
    x = 1
    while len(A) < n and x <= M:
        news = set()
        ok = True
        for a in A:
            v = a + x
            if v in sums or v in news:
                ok = False
                break
            news.add(v)
        if ok:
            news.add(2 * x)
            A.append(x)
            sums |= news
        x += 1

    if len(A) < n:
        used = set(A)
        for y in range(M + 1):
            if len(A) >= n:
                break
            if y not in used:
                A.append(y)
                used.add(y)

    A = sorted(A[:n])
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
