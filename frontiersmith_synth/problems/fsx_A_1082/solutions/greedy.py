# TIER: greedy
import sys

# The "obvious" recipe: for EACH bidder i, treat "the economy without i" and
# "the economy including everyone" as separate problems, and just rerun the
# standard O(n) welfare recurrence (max-weight independent set on a path) for
# each -- correct, and each single rerun uses the textbook efficient
# recurrence, but NOTHING computed for one bidder is ever reused for the
# next. Cost ~4n per bidder, times n bidders = ~4n^2 gates.

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

    def forward_dp(mask_i):
        """Fresh forward welfare DP over ALL n bidders; if mask_i is not None,
        bidder mask_i's value is treated as 0 (so it is never worth taking --
        exactly the welfare of the economy without that bidder)."""
        def vwire(j):  # wire holding v_j under this pass's masking
            return ZERO if j == mask_i else (j - 1)
        F = [None] * (n + 1)
        F[0] = ZERO
        if n >= 1:
            F[1] = vwire(1)
        for j in range(2, n + 1):
            cand = gate("ADD", F[j - 2], vwire(j))
            F[j] = gate("MAX", F[j - 1], cand)
        return F[n]

    outs = []
    for i in range(1, n + 1):
        V0 = forward_dp(mask_i=i)      # fresh DP, redone for this bidder only
        OPTi = forward_dp(mask_i=None)  # fresh DP AGAIN, redone for this bidder only
        x = gate("GT", OPTi, V0)
        vixi = gate("MUL", i - 1, x)
        t = gate("SUB", V0, OPTi)
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
