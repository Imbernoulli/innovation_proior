# TIER: greedy
"""The obvious recipe: fixed-period chunking. Count bytes in memory slot m0;
cut (and reset the counter) every W bytes, W chosen from N alone. This is a
perfectly reasonable, well-known chunking strategy -- but its boundaries are
a pure function of absolute position, not of content, so after the corpus is
edited every boundary downstream of the splice point lands on a different
relative offset and nothing dedups against the base layout."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    W = max(4, N // 8)
    prog = [
        f"ADD r12 r0 1",
        f"LT r13 r12 {W}",
        f"SUB r14 1 r13",
        f"SEL r15 r14 0 r12",
        f"RESULT r14 r15 0 0 0 0 0 0 0",
    ]
    print("\n".join(prog))


if __name__ == "__main__":
    main()
