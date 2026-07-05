# TIER: greedy
# Share all powers of two once, then compose each target from its set bits with
# fresh partial sums (no cross-target reuse).  length = m + sum(popcount-1).
import sys

def main():
    d = sys.stdin.read().split()
    k = int(d[0]); targets = [int(x) for x in d[1:1 + k]]
    maxt = max(targets)
    m = maxt.bit_length() - 1
    seq = [1]
    cur = 1
    for _ in range(m):
        cur *= 2
        seq.append(cur)                # powers 2..2^m
    for t in targets:
        bits = [i for i in range(t.bit_length()) if (t >> i) & 1]
        acc = 1 << bits[0]             # a shared power (present)
        for b in bits[1:]:
            acc = acc + (1 << b)       # acc(present) + power(present)
            seq.append(acc)            # always append (no reuse)
    sys.stdout.write(' '.join(map(str, seq)) + '\n')

if __name__ == "__main__":
    main()
