# TIER: strong
# RePair-style bottom-up digram merging. Unlike a left-to-right greedy scan, this looks
# at the WHOLE current sequence at once: repeatedly replace the single most frequent
# adjacent pair of symbols (terminals or already-built rules) with one new rule, and
# iterate. Because it ranks by GLOBAL frequency rather than local left-to-right match
# length, short atoms that recur dozens of times across many (mostly-clean) block/section
# copies get factored out first -- unaffected by a few distant point mutations. Once
# atoms become single symbols, frequently-recurring PAIRS OF ATOM-SYMBOLS (i.e. blocks)
# become the next most frequent digrams and get merged next, and so on: the hierarchy
# (atoms -> blocks -> sections) is rediscovered bottom-up, several scales at a time.
import sys
from collections import defaultdict


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0])
    S = data[1]

    seq = list(S)
    rules = []  # rules[k] = [tokA, tokB], the RHS of rule k (creation order)

    while True:
        L = len(seq)
        if L <= 1:
            break
        counts = defaultdict(int)
        first_pos = {}
        for i in range(L - 1):
            p = (seq[i], seq[i + 1])
            counts[p] += 1
            if p not in first_pos:
                first_pos[p] = i

        best = None
        best_count = 1
        for p, c in counts.items():
            if c < 2:
                continue
            if best is None or c > best_count or (c == best_count and first_pos[p] < first_pos[best]):
                best = p
                best_count = c

        if best is None:
            break

        a, b = best
        new_idx = len(rules)
        new_tok = "r%d" % new_idx
        rules.append([a, b])

        new_seq = []
        i = 0
        while i < L:
            if i < L - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(new_tok)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq

    R = len(rules) + 1
    out = [str(R)]
    for rhs in rules:
        out.append(" ".join(rhs))
    out.append(" ".join(seq))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
