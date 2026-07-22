# TIER: greedy
# Textbook "count repeated k-grams and replace the common ones" recipe: in one pass,
# count the GLOBAL frequency of every fixed-width window of length W=5; in a second
# left-to-right pass, whenever the current window's frequency is at least a threshold,
# replace it with a reference to one shared rule (created the first time that exact
# 5-character window is used), otherwise emit the character literally and move on by 1.
# This is hierarchy-blind in a specific way: it only ever looks at ONE fixed granularity.
# It can never assemble a larger reused block by first naming its smaller pieces and then
# reusing THOSE -- even when the true repeating unit is a 3-character atom embedded in a
# 30-character block, a rigid 5-character window frequently straddles two different atoms
# and so never repeats verbatim, while a bottom-up approach discovers whatever length
# actually recurs, at every scale, and only then composes the discovered pieces upward.
import sys

W = 5
THRESH = 6


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0])
    S = data[1]

    freq = {}
    for i in range(n - W + 1):
        w = S[i:i + W]
        freq[w] = freq.get(w, 0) + 1

    content_rule = {}
    rules = []
    start_tokens = []

    i = 0
    while i < n:
        if i + W <= n:
            w = S[i:i + W]
            if freq.get(w, 0) >= THRESH:
                ridx = content_rule.get(w)
                if ridx is None:
                    ridx = len(rules)
                    rules.append(list(w))
                    content_rule[w] = ridx
                start_tokens.append("r%d" % ridx)
                i += W
                continue
        start_tokens.append(S[i])
        i += 1

    R = len(rules) + 1
    out = [str(R)]
    for rhs in rules:
        out.append(" ".join(rhs))
    out.append(" ".join(start_tokens))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
