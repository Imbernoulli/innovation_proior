# TIER: greedy
"""The obvious first move: assume the first field is a stable tag (severity /
subsystem label), bucket every line by that single token, then within each
bucket mark a position variable the moment it EVER differs and constant only
if every line in the bucket agrees. One pass, no revisiting the grouping.

This is exactly the trap the family is built around: whenever two distinct
hidden templates happen to share the same first-token tag, they land in the
same bucket and (since they differ almost everywhere past position 0) nearly
every remaining position gets wildcarded to keep the merged bucket feasible
-- one bloated, near-universal template that "matches everything and
explains nothing"."""
import sys
from collections import OrderedDict


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); W = int(next(it))
    lines = [[next(it) for _ in range(W)] for _ in range(N)]

    buckets = OrderedDict()  # tag -> list of line indices, insertion order = deterministic
    for i, row in enumerate(lines):
        buckets.setdefault(row[0], []).append(i)

    templates = []
    assign = [0] * N
    for tag, idxs in buckets.items():
        tid = len(templates)
        tmpl = []
        for p in range(W):
            vals = {lines[i][p] for i in idxs}
            tmpl.append(lines[idxs[0]][p] if len(vals) == 1 else "*")
        templates.append(tmpl)
        for i in idxs:
            assign[i] = tid + 1

    out = [str(len(templates))]
    for tmpl in templates:
        out.append(" ".join(tmpl))
    out.append(" ".join(str(a) for a in assign))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
