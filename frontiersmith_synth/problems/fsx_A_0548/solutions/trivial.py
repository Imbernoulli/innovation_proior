# TIER: trivial
# Independent per-row XOR-fold: no reuse, no cancellation. Reproduces the checker baseline.
import sys

def main():
    tok = sys.stdin.read().split()
    p = 0
    def nx():
        nonlocal p
        v = int(tok[p]); p += 1; return v
    m = nx(); n = nx()
    rows = []
    for _ in range(m):
        bits = [j for j in range(n) if nx()]
        rows.append(bits)

    gates = []          # list of (a, b)
    def emit(a, b):
        gates.append((a, b))
        return n + len(gates) - 1

    outputs = []
    for bits in rows:
        node = bits[0]
        for b in bits[1:]:
            node = emit(node, b)
        outputs.append(node)

    out = [str(len(gates))]
    for a, b in gates:
        out.append("%d %d" % (a, b))
    out.append(" ".join(map(str, outputs)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
