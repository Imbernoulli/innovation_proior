# TIER: greedy
# The obvious batch improvement: build ONE shared square table g, g^2, g^4, ...
# and assemble every target from its binary digits.  Sees that the exponents
# share the base g, but is BLIND to the hidden modulus m -> still pays
# popcount(e_i) per exponent (which the odd-m scattering keeps large).
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

    maxbit = max(e.bit_length() for e in targets)
    pw = [0]
    cur = 0
    for _ in range(maxbit - 1):          # shared squarings of g
        cur = emit(cur, cur)
        pw.append(cur)

    built = set()
    for e in targets:
        if e in built:
            continue
        built.add(e)
        bits = [j for j in range(e.bit_length()) if (e >> j) & 1]
        acc = pw[bits[0]]
        for j in bits[1:]:
            acc = emit(acc, pw[j])

    outp = [str(len(pairs))]
    outp += ["%d %d" % (a, b) for (a, b) in pairs]
    sys.stdout.write("\n".join(outp) + "\n")


if __name__ == "__main__":
    main()
