# TIER: greedy
# The textbook single-modulus trick for "avoid repeated pairwise sums": the classic
# quadratic (Erdos-Turan / Singer-style) B2-sequence construction A[i] = (i + i*i) mod M,
# i = 0..k-1. Over a PRIME modulus this recipe is a genuinely good, well-known way to build
# a low-collision set, and it is a completely natural first attempt here: it directly
# targets the fine-grained echo count E_M, is O(k) to write, and needs no search at all.
#
# It never looks at D or W -- and it cannot "accidentally" get the coarse band right either,
# because i mod D and (i*i) mod D both cycle with a period that divides D, so the residues
# i + i*i mod D repeat with a short, rigid period no matter how large M is. That locks the
# construction into a small, unevenly-loaded set of coarse bands instead of the balanced
# spread the coarse term rewards -- the fine-tuned "textbook" recipe pays for the quotient
# structure it never modeled.
import sys


def main():
    toks = sys.stdin.read().split()
    M, D, k, W = int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3])
    A = [(i + i * i) % M for i in range(k)]
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
