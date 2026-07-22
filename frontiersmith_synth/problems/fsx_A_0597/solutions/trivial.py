# TIER: trivial
# Naive INDEPENDENT square-and-multiply per exponent: a fresh doubling chain for
# every target, no sharing at all.  Reproduces the checker's baseline B exactly
# -> Ratio ~ 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    k = int(data[0])
    targets = [int(x) for x in data[1:1 + k]]

    exps = [1]
    pairs = []

    def emit(a, b):
        exps.append(exps[a] + exps[b])
        pairs.append((a, b))
        return len(exps) - 1

    for e in targets:
        top = e.bit_length() - 1
        pw = [0]          # pw[j] = index producing g^{2^j}, built fresh per target
        cur = 0
        for _ in range(top):
            cur = emit(cur, cur)
            pw.append(cur)
        bits = [j for j in range(e.bit_length()) if (e >> j) & 1]
        acc = pw[bits[0]]
        for j in bits[1:]:
            acc = emit(acc, pw[j])

    outp = [str(len(pairs))]
    outp += ["%d %d" % (a, b) for (a, b) in pairs]
    sys.stdout.write("\n".join(outp) + "\n")


if __name__ == "__main__":
    main()
