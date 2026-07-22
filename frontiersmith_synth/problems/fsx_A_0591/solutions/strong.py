# TIER: strong
# Paterson-Stockmeyer baby-step / giant-step preconditioning.
# Insight: the coefficients are known offline, so any arithmetic on constants is free.
# Precompute x^2..x^k (k-1 nonscalar mults), form each width-k block B_j(x) with only
# free scalar multiplies + adds, then combine blocks by a Horner recurrence in y=x^k
# (m-1 nonscalar mults). Total ~ 2*sqrt(d) online multiplies, far below Horner's d.
import sys


def main():
    data = sys.stdin.read().split()
    d = int(data[0])
    a = [int(x) for x in data[1 : 1 + d + 1]]
    lines = []
    nreg = 1  # r0 = x

    def emit(s):
        nonlocal nreg
        lines.append(s)
        r = nreg
        nreg += 1
        return r

    # block width k = ceil(sqrt(d+1))
    k = 1
    while k * k < d + 1:
        k += 1
    m = (d + 1 + k - 1) // k  # number of blocks

    # precompute powers x^1..x^k ; pw[t] = register holding x^t
    pw = {1: 0}
    for t in range(2, k + 1):
        pw[t] = emit("MUL %d 0" % pw[t - 1])   # x^t = x^{t-1} * x  (nonscalar)
    y = pw[k]  # giant step base y = x^k

    def coeff(idx):
        return a[idx] if 0 <= idx <= d else 0

    def make_block(j):
        # B_j(x) = sum_{t=0}^{k-1} a[j*k + t] * x^t   (free: scalar mults + adds)
        base = j * k
        acc = emit("CON %d" % coeff(base))  # constant term
        for t in range(1, k):
            c = coeff(base + t)
            if c == 0:
                continue
            term = emit("SMUL %d %d" % (pw[t], c))
            acc = emit("ADD %d %d" % (acc, term))
        return acc

    # Horner in y over the blocks, from top block down
    acc = make_block(m - 1)
    for j in range(m - 2, -1, -1):
        mul = emit("MUL %d %d" % (acc, y))  # acc * y  (nonscalar giant step)
        blk = make_block(j)
        acc = emit("ADD %d %d" % (mul, blk))
    lines.append("RET %d" % acc)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
