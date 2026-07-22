# TIER: strong
"""Insight: border lengths correspond to periods (p = n - b), and periods of a
word are NOT independently choosable. Committing to target periods p1,p2,...
mechanically FORCES more structure than the targets alone: any multiple of a
period is itself a period, and (Fine and Wilf) two periods p1,p2 with
p1+p2-gcd(p1,p2) <= n force gcd(p1,p2) as a period too. Rather than reasoning
about these rules case by case, we build the ground-truth consequence
directly: union positions i and i+p for every target period p (for all valid
i). The resulting equivalence classes capture EXACTLY what letter-equalities
are mathematically unavoidable once the targets are committed to -- nothing
more. Assigning each class its OWN distinct letter (as long as classes fit in
the K-letter budget) then realizes precisely those unavoidable borders and no
accidental extra ones, while typically using far fewer than K distinct
letters -- collecting the alphabet-saving bonus that naive independent
prefix=suffix stitching leaves on the table (and often breaks besides, since
overlapping stitches can silently overwrite an earlier target)."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    K = int(toks[idx]); idx += 1
    idx += 1  # lam
    idx += 1  # alpha
    m = int(toks[idx]); idx += 1
    targets = []
    for _ in range(m):
        b = int(toks[idx]); idx += 1
        w = int(toks[idx]); idx += 1
        targets.append((b, w))

    periods = sorted(n - b for b, w in targets)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in periods:
        for i in range(n - p):
            union(i, i + p)

    class_id = {}
    for i in range(n):
        r = find(i)
        if r not in class_id:
            class_id[r] = len(class_id)

    W = []
    for i in range(n):
        cid = class_id[find(i)]
        W.append(cid % K)

    sys.stdout.write(" ".join(map(str, W)) + "\n")


if __name__ == "__main__":
    main()
