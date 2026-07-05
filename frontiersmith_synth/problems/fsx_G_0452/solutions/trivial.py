# TIER: trivial
# Reference binary double-and-add SLP: build each linear form independently, no sharing.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it)); n = int(next(it))
    M = [[int(next(it)) for _ in range(n)] for _ in range(m)]

    regs = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
    instrs = []

    def dbl(a):
        regs.append(tuple(2 * v for v in regs[a]))
        instrs.append("DBL %d" % a)
        return len(regs) - 1

    def add(a, b):
        regs.append(tuple(x + y for x, y in zip(regs[a], regs[b])))
        instrs.append("ADD %d %d" % (a, b))
        return len(regs) - 1

    def sub(a, b):
        regs.append(tuple(x - y for x, y in zip(regs[a], regs[b])))
        instrs.append("SUB %d %d" % (a, b))
        return len(regs) - 1

    def binmul(a, j):
        bits = bin(a)[2:]
        cur = j
        for bit in bits[1:]:
            cur = dbl(cur)
            if bit == '1':
                cur = add(cur, j)
        return cur

    outs = []
    for i in range(m):
        items = [(j, M[i][j]) for j in range(n) if M[i][j] != 0]
        pos = [(j, c) for (j, c) in items if c > 0]
        neg = [(j, c) for (j, c) in items if c < 0]
        order = pos + neg
        running = None
        for j, c in order:
            term = binmul(abs(c), j)
            if running is None:
                running = term
            else:
                running = add(running, term) if c > 0 else sub(running, term)
        outs.append(running)

    lines = [str(len(instrs))]
    lines.extend(instrs)
    lines.append(" ".join(str(o) for o in outs))
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
