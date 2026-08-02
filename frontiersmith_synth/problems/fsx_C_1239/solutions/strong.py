# TIER: strong
"""The insight: a checksum group only fails to notice damage when the number
of ITS bits that got flipped is even. Uniform blocking makes a burst's
"share" of damage land almost entirely inside whichever single block it
started in -- so an even-length burst is invisible to that block, and every
other block sees zero (also even). The fix is not "more blocks" or "bigger
blocks", it is a different ROUTING: interleave message positions across the
K checksum groups cyclically (position i -> group i mod K) instead of
slicing the message into contiguous runs.

Why this wins: a length-L burst starting anywhere now lands roughly L/K bits
in EACH group, split as ceil(L/K) bits in some groups and floor(L/K) in the
rest. Those two counts are consecutive integers, so they can never both be
even -- unless L happens to be an exact multiple of K (then every group gets
the same count, and it only fails when that shared count is itself even).
So interleaving turns "one block silently absorbs the whole burst" into "the
burst is smeared across every checksum at once", and it is defeated only by
the narrow coincidence the channel profile plants on purpose. Scattered
independent-bit errors are essentially unaffected by the routing choice
(each isolated flip still lands in exactly one group and flips its parity),
so this reformulation costs nothing on the warm-up cases while decisively
winning the burst-heavy ones."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    M = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    # SEED and PBURST are not needed: cyclic interleaving is robust to the
    # burst-length distribution by construction, regardless of its exact mix.

    out = [str(i % K) for i in range(M)]
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
