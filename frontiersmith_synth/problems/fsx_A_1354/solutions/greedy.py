# TIER: greedy
"""The obvious first move: plain run-length encoding. Group the crossing sequence into
maximal runs of the same symbol, then chain the runs together: rule0 = run0 (symbol
repeated its run length); rule_i = ref(rule_{i-1})*1 + run_i's symbol * run_i's length.

This is a real, generically-useful compressor (it wins clearly whenever one continued-
fraction term dominates -- long straight run of one crossing type). It does NOT look at
the slope's continued-fraction structure at all, so on inputs whose continued fraction
has a small leading term (let alone the all-ones / Fibonacci case) almost every run has
length 1 or 2 and this buys next to nothing beyond the naive baseline."""
import sys


def true_sequence(P, Q):
    i, j = 1, 1
    out = []
    while i < P or j < Q:
        if i < P and (j >= Q or i * Q < j * P):
            out.append('V')
            i += 1
        else:
            out.append('H')
            j += 1
    return ''.join(out)


def run_length_encode(S):
    runs = []
    for ch in S:
        if runs and runs[-1][0] == ch:
            runs[-1][1] += 1
        else:
            runs.append([ch, 1])
    return runs


def main():
    P, Q = map(int, sys.stdin.read().split())
    S = true_sequence(P, Q)
    runs = run_length_encode(S)

    rules = []
    sym0, cnt0 = runs[0]
    rules.append("0 %s %d H 0" % (sym0, cnt0))
    cur = 0
    for sym, cnt in runs[1:]:
        idx = len(rules)
        rules.append("%d %d 1 %s %d" % (idx, cur, sym, cnt))
        cur = idx

    out = [str(len(rules))] + rules + ["ANSWER %d" % cur]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
