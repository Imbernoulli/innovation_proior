# TIER: greedy
# The "obvious strong-coder" move: textbook recursive mixed-radix Cooley-Tukey
# (decimation in time), splitting off the smallest prime factor at each level and
# recursing on the rest, with an explicit TWIDDLE multiplication between the two
# stages of every split -- exactly as taught for a GENERAL composite length. It
# never notices that our factors are pairwise COPRIME, so it keeps paying twiddles
# it did not need to pay (and it never re-attacks a prime leaf with Rader either).
import sys


class Builder:
    def __init__(self, n, q):
        self.n = n
        self.q = q
        self.instrs = []
        self.next_reg = n

    def add(self, a, b):
        self.instrs.append("A %d %d" % (a, b))
        r = self.next_reg
        self.next_reg += 1
        return r

    def sub(self, a, b):
        self.instrs.append("S %d %d" % (a, b))
        r = self.next_reg
        self.next_reg += 1
        return r

    def mulc(self, a, c):
        c %= self.q
        self.instrs.append("M %d %d" % (a, c))
        r = self.next_reg
        self.next_reg += 1
        return r

    def term(self, c, a):
        c %= self.q
        if c == 1:
            return a
        return self.mulc(a, c)

    def matvec(self, M, xregs):
        out = []
        for row in M:
            terms = [(c % self.q, xregs[j]) for j, c in enumerate(row) if c % self.q != 0]
            if not terms:
                out.append(self.mulc(xregs[0], 0))
                continue
            c0, r0 = terms[0]
            cur = self.term(c0, r0)
            for c, r in terms[1:]:
                t = self.term(c, r)
                cur = self.add(cur, t)
            out.append(cur)
        return out


def smallest_prime_factor(x):
    d = 2
    while d * d <= x:
        if x % d == 0:
            return d
        d += 1
    return x


def ct_transform(b, n, q, w, xregs):
    """Recursive decimation-in-time Cooley-Tukey with explicit twiddles."""
    if n == 1:
        return [xregs[0]]
    p = smallest_prime_factor(n)
    if p == n:
        # prime leaf: naive p x p DFT, no Rader re-attack
        M = [[pow(w, (j * k) % n, q) for j in range(n)] for k in range(n)]
        return b.matvec(M, xregs)

    m = n // p
    wm = pow(w, p, q)  # primitive m-th root
    wp = pow(w, m, q)  # primitive p-th root

    # decimation in time: i = i1 + p*i2, i1 in [0,p), i2 in [0,m)
    stage1 = []
    for i1 in range(p):
        sub_x = [xregs[i1 + p * i2] for i2 in range(m)]
        stage1.append(ct_transform(b, m, q, wm, sub_x))

    # twiddle multiply: U[i1][k2] = w^(i1*k2) * stage1[i1][k2]
    twid = []
    for i1 in range(p):
        row = []
        for k2 in range(m):
            e = (i1 * k2) % n
            c = pow(w, e, q)
            row.append(b.term(c, stage1[i1][k2]))
        twid.append(row)

    out = [None] * n
    for k2 in range(m):
        col = [twid[i1][k2] for i1 in range(p)]
        col_out = ct_transform(b, p, q, wp, col)
        for k1 in range(p):
            out[k1 * m + k2] = col_out[k1]
    return out


def main():
    n, q, w = map(int, sys.stdin.read().split())
    xregs = list(range(n))
    b = Builder(n, q)
    outs = ct_transform(b, n, q, w, xregs)

    L = len(b.instrs)
    out = ["%d %d" % (n + L, L)]
    out.extend(b.instrs)
    out.append("O " + " ".join(map(str, outs)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
