# TIER: strong
"""The insight: reserve tag space proportional to a field's EXPECTED total
future weight, not its snapshot-today frequency. Every field's true cost
contribution is freq * (number of versions it is ever active for), because
that's how many times its tag actually gets paid for across the schema's
whole lifetime. Fields that don't exist yet still have a known lifetime
weight (the evolution script is given up front) and MUST be allowed to
outbid an already-existing low-value field for a cheap tag -- paying a few
extra bytes on some version-1 field today to keep a future high-volume
field cheap forever.

Because the per-tag cost is a non-decreasing step function of tag rank, the
exchange argument makes "sort every field (born or unborn) by lifetime
weight, hand out tags 0..M-1 in that order" the cost-minimizing assignment
for this weight vector -- a genuine reformulation of the problem (weight by
lifetime footprint, not current-version count), not merely "greedy with
more lookahead iterations"."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    V, M, t1cap, t2cap, t2cost, t3cost = (int(next(it)) for _ in range(6))
    fields = []  # (fid, weight)
    for _ in range(M):
        fid = int(next(it)); g = int(next(it)); v0 = int(next(it)); freq = int(next(it))
        weight = freq * (V - v0 + 1)
        fields.append((fid, weight))

    order = sorted(fields, key=lambda f: -f[1])
    tag_of = {}
    for rank, (fid, w) in enumerate(order):
        tag_of[fid] = rank

    out = [f"{fid} {tag_of[fid]}" for fid in range(M)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
