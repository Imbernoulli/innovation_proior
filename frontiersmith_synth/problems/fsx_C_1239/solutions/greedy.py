# TIER: greedy
"""The obvious first idea: uniformly spaced checksums. Cut the message into K
equal, CONTIGUOUS blocks and give each block its own checksum group -- the
textbook "evenly partition the data across your parity bits" recipe. This is
provably optimal against scattered, independent bit errors (each isolated
flip lands in some block and flips that block's parity, full stop). It falls
apart against a contiguous burst that fits inside one block: the block's own
parity only depends on how many of ITS bits got hit, so an even-length burst
sitting entirely inside a single block cancels out and is invisible to every
checksum at once."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    M = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    # SEED, PBURST, and the burst-length table are unused by this tier --
    # uniform spacing does not look at the channel profile at all.

    block = M // K
    out = []
    for i in range(M):
        g = min(i // block, K - 1)
        out.append(str(g))
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
