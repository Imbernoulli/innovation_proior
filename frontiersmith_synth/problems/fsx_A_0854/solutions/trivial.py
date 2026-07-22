# TIER: trivial
# Build every target completely independently: chain-XOR its own variables
# one at a time, no sharing at all across rows. Gate count == the checker's
# baseline B == sum(k_i - 1) -> Ratio == 0.1 on every case.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it))
    rows = []
    for _ in range(R):
        k = int(next(it))
        idxs = [int(next(it)) for _ in range(k)]
        rows.append(idxs)

    gates = []          # list of (a, b)
    out_wire = [0] * R

    def new_gate(a, b):
        gates.append((a, b))
        return C + len(gates)

    for r, idxs in enumerate(rows):
        w = idxs[0]                      # wire = the first variable itself
        for v in idxs[1:]:
            w = new_gate(w, v)
        out_wire[r] = w

    lines = [str(len(gates))]
    for a, b in gates:
        lines.append("%d %d" % (a, b))
    for w in out_wire:
        lines.append(str(w))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
