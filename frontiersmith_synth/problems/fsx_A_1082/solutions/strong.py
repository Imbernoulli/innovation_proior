# TIER: strong
import sys

# Insight: the n leave-one-out counterfactual economies overlap massively.
# Build ONE forward prefix DP table F[0..n] and ONE backward suffix DP table
# Bwd[1..n+1] ONCE (a single shared pass each way), then every bidder's
# leave-one-out welfare and VCG payment is an O(1) stitch of a precomputed
# prefix half and a precomputed suffix half -- no economy is ever recomputed
# from scratch. Total gates ~ 9n (vs ~4n^2 for a fresh recompute per economy).
#
# The circuit is purely STRUCTURAL: it depends only on n (which bidder sits
# on which input wire), never on the concrete bid values -- the same emitted
# program must reproduce every trial's payments, since the checker
# re-evaluates it once per trial with that trial's values on wires 0..n-1.

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    # data[1] is T (trial count); the concrete bid values are never read here
    # -- the circuit's shape depends only on n.

    gates = []

    def gate(op, a=None, b=None):
        gates.append((op, a, b))
        return n + len(gates) - 1

    ZERO = gate("CONST", 0)

    # forward prefix DP: F[j] = best welfare using bidders 1..j
    F = [None] * (n + 1)
    F[0] = ZERO
    if n >= 1:
        F[1] = 0  # input wire 0 = v_1
    for j in range(2, n + 1):
        cand = gate("ADD", F[j - 2], j - 1)      # v_j sits on input wire j-1
        F[j] = gate("MAX", F[j - 1], cand)

    # backward suffix DP: Bwd[j] = best welfare using bidders j..n
    Bwd = [None] * (n + 2)
    Bwd[n + 1] = ZERO
    if n >= 1:
        Bwd[n] = n - 1  # input wire n-1 = v_n
    for j in range(n - 1, 0, -1):
        cand = gate("ADD", Bwd[j + 2], j - 1)
        Bwd[j] = gate("MAX", Bwd[j + 1], cand)

    OPT = F[n]  # global welfare, computed ONCE, reused by every bidder below

    outs = []
    for i in range(1, n + 1):
        V0 = gate("ADD", F[i - 1], Bwd[i + 1])   # OPT(economy without i) -- prefix+suffix stitch
        x = gate("GT", OPT, V0)                  # is i essential to the optimum?
        vixi = gate("MUL", i - 1, x)
        t = gate("SUB", V0, OPT)
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
