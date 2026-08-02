# TIER: greedy
import sys


def vmod(bits, M):
    v = 0
    for c in bits:
        v = (2 * v + (1 if c == '1' else 0)) % M
    return v


def main():
    toks = sys.stdin.read().split()
    it = 0
    seed = int(toks[it]); it += 1
    step_bound = int(toks[it]); it += 1
    n = int(toks[it]); it += 1
    samples = []
    for _ in range(n):
        bits = toks[it]; it += 1
        label = int(toks[it]); it += 1
        samples.append((bits, label))

    # Obvious first attempt: assume behaviour only depends on a bounded
    # recent-context window (the last 4 bits, i.e. value mod 16) -- the
    # standard n-gram / sliding-window instinct for sequence classification.
    # Build a 16-state table by majority vote per window from the samples.
    buckets = [[] for _ in range(16)]
    for bits, label in samples:
        w = vmod(bits, 16)
        buckets[w].append(label)
    tbl = []
    for w in range(16):
        b = buckets[w]
        if not b:
            tbl.append(0)
        else:
            ones = sum(b)
            tbl.append(1 if 2 * ones >= len(b) else 0)

    lines = ["16"]
    for w in range(16):
        t0 = (2 * w) % 16
        t1 = (2 * w + 1) % 16
        tb = 'A' if tbl[w] == 1 else 'R'
        lines.append("%d %d %s" % (t0, t1, tb))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
