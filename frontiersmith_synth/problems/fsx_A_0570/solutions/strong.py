# TIER: strong
# Insight: the whole batch shares its low s bits (a common sub-exponent c).
# Split each target with meet-in-the-middle:  n_j = (h_j << s) + c.
#   * build the shared doubling ladder once (powers of two);
#   * assemble the shared low chunk C = c exactly ONCE and reuse it;
#   * for each target, assemble the high part (h_j << s) from the ladder and add
#     the single reusable C.
# This pays for c's set bits once instead of once per target -- saving roughly
# (k-1)*(popcount(c)-1) additions over the shared-ladder greedy.
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

    # --- detect the maximal shared low suffix s and its value c ---
    first = targets[0]
    diff = 0
    for n in targets:
        diff |= (n ^ first)
    if diff == 0:
        s = first.bit_length()      # all identical (degenerate)
    else:
        s = (diff & -diff).bit_length() - 1   # lowest differing bit position
    c = first & ((1 << s) - 1)

    # --- shared doubling ladder ---
    M = max(n.bit_length() - 1 for n in targets)
    ladder = [0] * (M + 1)
    prev = 0
    for i in range(1, M + 1):
        prev = emit(prev, prev)
        ladder[i] = prev

    # --- assemble the shared low chunk C = c once ---
    cbits = [i for i in range(s) if (c >> i) & 1]
    Cidx = None
    if cbits:
        Cidx = ladder[cbits[0]]
        for i in cbits[1:]:
            Cidx = emit(Cidx, ladder[i])
        # Cidx now holds value c

    # --- per target: high part + shared C ---
    for n in targets:
        h = n >> s
        if h == 0:
            # n == c, already produced as Cidx
            continue
        hbits = [i for i in range(s, n.bit_length()) if (n >> i) & 1]
        acc = ladder[hbits[0]]
        for i in hbits[1:]:
            acc = emit(acc, ladder[i])
        # acc holds (h << s); add the shared low chunk
        if Cidx is not None:
            acc = emit(acc, Cidx)
        # acc now holds n

    outp = [str(len(steps))]
    outp.extend("%d %d" % (a, b) for (a, b) in steps)
    sys.stdout.write("\n".join(outp) + "\n")


if __name__ == "__main__":
    main()
