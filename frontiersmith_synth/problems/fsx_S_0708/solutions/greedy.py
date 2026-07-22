# TIER: greedy
# Textbook bottom-up grammar compression (RePair-style): repeatedly find the
# most frequent adjacent symbol pair and replace every non-overlapping
# occurrence with a fresh nonterminal, until no pair repeats.  This is "the
# obvious first algorithm" for smallest-grammar problems.  It only ever
# builds concatenation rules -- it has no notion of a reversed occurrence, so
# it cannot share a nonterminal between a block and its mirror image.
import sys


def main():
    t = sys.stdin.readline().rstrip("\n")
    n = len(t)

    rules = []          # list of ("T", ch) or ("C", j, k); index+1 = rule id
    char_id = {}
    seq = []
    for ch in t:
        if ch not in char_id:
            rules.append(("T", ch))
            char_id[ch] = len(rules)
        seq.append(char_id[ch])

    while len(seq) > 1:
        freq = {}
        first_pos = {}
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            freq[pair] = freq.get(pair, 0) + 1
            if pair not in first_pos:
                first_pos[pair] = i

        best_pair, best_count = None, 1
        for pair, cnt in freq.items():
            if cnt > best_count or (cnt == best_count and best_pair is not None
                                     and first_pos[pair] < first_pos[best_pair]):
                best_pair, best_count = pair, cnt
        if best_pair is None or best_count < 2:
            break

        rules.append(("C", best_pair[0], best_pair[1]))
        new_id = len(rules)

        new_seq = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq

    # chain-combine whatever symbols remain into a single start rule
    if len(seq) == 1:
        pass
    else:
        prev = seq[0]
        for s in seq[1:]:
            rules.append(("C", prev, s))
            prev = len(rules)

    out = []
    for r in rules:
        if r[0] == "T":
            out.append("T %s" % r[1])
        else:
            out.append("C %d %d" % (r[1], r[2]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
