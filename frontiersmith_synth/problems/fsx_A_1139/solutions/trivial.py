# TIER: trivial
# Naive full n x n NTT matrix multiplication: one "M" instruction per nontrivial
# matrix entry, combined by free additions. This exactly reproduces the checker's
# own internal baseline B, so it scores ~0.1.
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


def main():
    n, q, w = map(int, sys.stdin.read().split())
    xregs = list(range(n))
    b = Builder(n, q)
    M = [[pow(w, (j * k) % n, q) for j in range(n)] for k in range(n)]
    outs = b.matvec(M, xregs)

    L = len(b.instrs)
    out = ["%d %d" % (n + L, L)]
    out.extend(b.instrs)
    out.append("O " + " ".join(map(str, outs)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
