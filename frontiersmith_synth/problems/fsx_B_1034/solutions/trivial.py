# TIER: trivial
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); D = int(next(it)); R = int(next(it)); cap = int(next(it))
    chain = [int(next(it)) for _ in range(D)]
    free = [int(next(it)) for _ in range(R)]

    out = []
    # chain node i (ascending k) = the first k_i integers -> nested prefixes, trivially
    # satisfies the subset hierarchy, but is maximally *uneven* (all bunched at the start).
    for k in chain:
        out.append(" ".join(str(x) for x in range(k)))
    # each free instrument gets its own consecutive block, parked at an evenly spaced
    # anchor around the cycle (so nothing ever collides), still bunched internally.
    groups = 1 + R
    spacing = N // groups if groups > 0 else N
    for idx, f in enumerate(free):
        anchor = (idx + 1) * spacing
        out.append(" ".join(str((anchor + x) % N) for x in range(f)))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
