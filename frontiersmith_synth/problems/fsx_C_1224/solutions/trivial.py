# TIER: trivial
"""Reproduces the checker's own reference construction: sort ALL fields by
ASCENDING frequency and hand out tags 0..M-1 in that order (the least-used
fields get the cheap tags). A fixed, trivially-constructible anchor that
ignores everything about which fields actually deserve to be cheap --
scores ~0.1 by construction."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    V, M, t1cap, t2cap, t2cost, t3cost = (int(next(it)) for _ in range(6))
    fields = []  # (fid, freq)
    for _ in range(M):
        fid = int(next(it)); g = int(next(it)); v0 = int(next(it)); freq = int(next(it))
        fields.append((fid, freq))

    order = sorted(fields, key=lambda f: f[1])  # ascending freq
    tag_of = {}
    for rank, (fid, freq) in enumerate(order):
        tag_of[fid] = rank

    out = [f"{fid} {tag_of[fid]}" for fid in range(M)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
