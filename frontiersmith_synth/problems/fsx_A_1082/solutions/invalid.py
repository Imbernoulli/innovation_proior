# TIER: invalid
import sys

# Charges every bidder their full declared bid (pay_i = v_i) -- a plausible-
# looking but economically wrong "mechanism" (not the VCG externality), so it
# must be rejected by the exact-equivalence gate on essentially every case.

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])

    gates = []

    def gate(op, a=None, b=None):
        gates.append((op, a, b))
        return n + len(gates) - 1

    ZERO = gate("CONST", 0)
    outs = []
    for i in range(1, n + 1):
        w = gate("ADD", i - 1, ZERO)  # pay_i := v_i
        outs.append(w)

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
