# TIER: trivial
# Independent square-and-multiply for each target: a fresh binary addition chain
# per target, starting from value 1.  This reproduces the checker's baseline B,
# so it scores ~0.1.  It shares nothing between targets.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    k = int(next(it))
    targets = [int(next(it)) for _ in range(k)]

    values = [1]          # index 0
    steps = []

    def emit(a, b):
        steps.append((a, b))
        values.append(values[a] + values[b])
        return len(values) - 1

    for n in targets:
        if n == 1:
            continue
        bs = bin(n)[2:]           # MSB..LSB, leading '1'
        cur = 0                   # index holding the value 1 (leading bit)
        for ch in bs[1:]:
            cur = emit(cur, cur)  # double
            if ch == '1':
                cur = emit(cur, 0)  # + 1

    outp = [str(len(steps))]
    outp.extend("%d %d" % (a, b) for (a, b) in steps)
    sys.stdout.write("\n".join(outp) + "\n")


if __name__ == "__main__":
    main()
