# TIER: trivial
import sys

# The checker's own baseline construction: same "recompute every economy
# from scratch" habit as greedy, but even less refined -- for every bidder it
# double-checks each fresh welfare value with BOTH a forward pass and a
# (here, unused) backward pass, "just in case", instead of trusting a single
# direction. Same asymptotic idea as greedy, roughly 2x its gate count.

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    # data[1] is T (trial count); the circuit's shape depends only on n --
    # the checker re-evaluates the SAME emitted program once per trial.

    gates = []

    def gate(op, a=None, b=None):
        gates.append((op, a, b))
        return n + len(gates) - 1

    ZERO = gate("CONST", 0)

    def vwire(mask_i, j):
        return ZERO if j == mask_i else (j - 1)

    def forward_dp(mask_i):
        F = [None] * (n + 1)
        F[0] = ZERO
        if n >= 1:
            F[1] = vwire(mask_i, 1)
        for j in range(2, n + 1):
            cand = gate("ADD", F[j - 2], vwire(mask_i, j))
            F[j] = gate("MAX", F[j - 1], cand)
        return F[n]

    def backward_dp(mask_i):
        Bwd = [None] * (n + 2)
        Bwd[n + 1] = ZERO
        if n >= 1:
            Bwd[n] = vwire(mask_i, n)
        for j in range(n - 1, 0, -1):
            cand = gate("ADD", Bwd[j + 2], vwire(mask_i, j))
            Bwd[j] = gate("MAX", Bwd[j + 1], cand)
        return Bwd[1]

    outs = []
    for i in range(1, n + 1):
        V0f = forward_dp(mask_i=i)
        _V0b = backward_dp(mask_i=i)        # redundant double-check, unused
        OPTf = forward_dp(mask_i=None)
        _OPTb = backward_dp(mask_i=None)    # redundant double-check, unused
        x = gate("GT", OPTf, V0f)
        vixi = gate("MUL", i - 1, x)
        t = gate("SUB", V0f, OPTf)
        pay = gate("ADD", t, vixi)
        outs.append(pay)

    lines = [str(len(gates))]
    for op, a, b in gates:
        if op == "CONST":
            lines.append("CONST %d" % a)
        else:
            lines.append("%s %d %d" % (op, a, b))
    lines.append("OUT " + " ".join(str(w) for w in outs))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
