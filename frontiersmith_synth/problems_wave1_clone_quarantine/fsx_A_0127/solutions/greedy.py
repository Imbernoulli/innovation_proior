# TIER: greedy
# Mian-Chowla greedy Sidon layout: scan mileposts 0,1,...,V in order and install
# each one iff the resulting set keeps ALL pairwise sums distinct (equivalently a
# B_2 / Sidon set, which also keeps all differences distinct). Stop at k stations.
# Near-optimal when the corridor is loose; stalls early and undersizes when tight.
import sys


def main():
    data = sys.stdin.read().split()
    k = int(data[0])
    V = int(data[1])

    A = []
    sums = set()  # pairwise sums (including doubles) currently realized
    for x in range(V + 1):
        if len(A) >= k:
            break
        # candidate new sums from adding x
        new = []
        ok = True
        cand = set()
        for a in A:
            s = a + x
            if s in sums or s in cand:
                ok = False
                break
            cand.add(s)
        if not ok:
            continue
        d = x + x
        if d in sums or d in cand:
            continue
        cand.add(d)
        # commit
        for s in cand:
            sums.add(s)
        A.append(x)

    if not A:
        A = [0]
    out = [str(len(A))]
    out += [str(x) for x in A]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
