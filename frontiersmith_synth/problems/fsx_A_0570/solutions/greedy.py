# TIER: greedy
# The obvious first improvement: build ONE shared doubling ladder (powers of two
# 1,2,4,...,2^M) a single time, then assemble each target by adding up the powers
# of two at its set bit positions.  This amortizes the doublings across the batch
# -- but it still pays for EVERY set bit of every target, so the shared low chunk
# c is re-summed once per target.  It never notices that the low bits are shared.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    k = int(next(it))
    targets = [int(next(it)) for _ in range(k)]

    values = [1]
    steps = []

    def emit(a, b):
        steps.append((a, b))
        values.append(values[a] + values[b])
        return len(values) - 1

    M = max(n.bit_length() - 1 for n in targets)
    ladder = [0] * (M + 1)      # ladder[i] = index of value 2^i
    ladder[0] = 0
    prev = 0
    for i in range(1, M + 1):
        prev = emit(prev, prev)
        ladder[i] = prev

    for n in targets:
        bits = [i for i in range(n.bit_length()) if (n >> i) & 1]
        acc = ladder[bits[0]]
        for i in bits[1:]:
            acc = emit(acc, ladder[i])
        # acc now holds n

    outp = [str(len(steps))]
    outp.extend("%d %d" % (a, b) for (a, b) in steps)
    sys.stdout.write("\n".join(outp) + "\n")


if __name__ == "__main__":
    main()
