# TIER: trivial
"""Naive per-symbol chain: rule0 = S[0]; rule_i = ref(rule_{i-1})*1 + S[i]*1. This is
exactly the checker's own baseline construction, so it always ties the baseline exactly
(score ~0.1 on every case)."""
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


def main():
    P, Q = map(int, sys.stdin.read().split())
    S = true_sequence(P, Q)
    L = len(S)
    out = [str(L)]
    out.append("0 %s 1 H 0" % S[0])
    for idx in range(1, L):
        out.append("%d %d 1 %s 1" % (idx, idx - 1, S[idx]))
    out.append("ANSWER %d" % (L - 1))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
