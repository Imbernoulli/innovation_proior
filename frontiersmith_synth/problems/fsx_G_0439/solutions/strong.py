# TIER: strong
# Share all powers of two, then compose each target by CANONICAL low-to-high prefix
# sums of its set bits, REUSING any partial sum already in the chain (from another
# target).  Canonical order maximizes cross-target sharing of prefixes, so
# length = m + (#distinct partial-sum values) << greedy.
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
        seq.append(cur)
    present = set(seq)
    # process targets sorted so shared low prefixes are built early and reused
    for t in sorted(targets):
        bits = [i for i in range(t.bit_length()) if (t >> i) & 1]
        acc = 1 << bits[0]             # shared power, present
        for b in bits[1:]:
            acc = acc + (1 << b)       # both operands present
            if acc not in present:
                seq.append(acc)
                present.add(acc)
        # t == acc is now present (or already a power if popcount==1)
    sys.stdout.write(' '.join(map(str, seq)) + '\n')

if __name__ == "__main__":
    main()
