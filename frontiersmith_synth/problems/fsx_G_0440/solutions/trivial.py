# TIER: trivial
# Naive per-row construction: for each output row, chain-XOR its set bits with no
# sharing across rows.  Uses exactly sum_j (popcount(row_j)-1) gates == checker
# baseline B, so it scores ~0.1.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it)); n = int(next(it))
    rows = []
    for _ in range(m):
        rows.append([int(next(it)) for _ in range(n)])

    gates = []          # each (a,b)
    outs = []
    next_id = n
    for r in rows:
        bits = [i for i in range(n) if r[i]]
        cur = bits[0]
        for i in bits[1:]:
            gates.append((cur, i))
            cur = next_id
            next_id += 1
        outs.append(cur)

    lines = [str(len(gates))]
    for a, b in gates:
        lines.append("%d %d" % (a, b))
    lines.append(" ".join(str(o) for o in outs))
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
