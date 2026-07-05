# TIER: trivial
# Naive Winograd algorithm: one scalar multiplier per USED product fiber a[i]*b[j].
# For every (i,j) whose output fiber is nonzero, emit the term
#     u = e_i ,  v = e_j ,  w = ( T[0][i][j] .. T[s-1][i][j] )
# so the single product a[i]*b[j] is fanned out to the outputs it feeds.
# R = B, exactly the checker's baseline  -> Ratio ~ 0.1.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it)); q = int(next(it)); s = int(next(it))
    T = [[[0] * q for _ in range(p)] for _ in range(s)]
    for i in range(p):
        for j in range(q):
            for k in range(s):
                T[k][i][j] = int(next(it))

    terms = []
    for i in range(p):
        for j in range(q):
            fiber = [T[k][i][j] for k in range(s)]
            if all(x == 0 for x in fiber):
                continue
            u = [0] * p; u[i] = 1
            v = [0] * q; v[j] = 1
            w = fiber
            terms.append(u + v + w)

    out = [str(len(terms))]
    for t in terms:
        out.append(" ".join(map(str, t)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
