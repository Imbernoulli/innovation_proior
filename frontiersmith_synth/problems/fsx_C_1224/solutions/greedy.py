# TIER: greedy
"""Current-frequency-greedy, the obvious first idea: 'give short tags to
frequent fields'. Sorts the version-1 fields by their own frequency and
spends the ENTIRE cheap-tag budget on them right away, since that's all the
"current" data shows. Every field introduced later (version 2, 3, ...) is
just appended, first-come-first-served, in the order it shows up -- this
solution never revisits a tag it already committed, exactly like a team
that ships v1's wire format and only extends it afterward.

Optimal for the version-1 snapshot; breaks once high-volume fields appear in
later versions, because by then the cheap tag space is already spent on
whatever happened to be frequent at v1."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    V, M, t1cap, t2cap, t2cost, t3cost = (int(next(it)) for _ in range(6))
    fields = []  # (fid, group, v0, freq)
    for _ in range(M):
        fid = int(next(it)); g = int(next(it)); v0 = int(next(it)); freq = int(next(it))
        fields.append((fid, g, v0, freq))

    v1_fields = [f for f in fields if f[2] == 1]
    later_fields = [f for f in fields if f[2] != 1]

    v1_sorted = sorted(v1_fields, key=lambda f: -f[3])
    later_sorted = sorted(later_fields, key=lambda f: (f[2], -f[3]))

    order = v1_sorted + later_sorted
    tag_of = {}
    for rank, f in enumerate(order):
        tag_of[f[0]] = rank

    out = [f"{fid} {tag_of[fid]}" for fid in range(M)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
