# TIER: greedy
#
# The "average strong coder's first idea": build the word one letter at a
# time, always trying the smallest available letter first, and stop the
# instant every letter choice would create a palindromic factor longer than
# p. No lookahead, no backtracking, no idea of a substitution/morphism.
import sys


def creates_violation(word, p):
    """word: list[int], already includes the newly appended letter. Returns
    True iff some suffix of `word` (of length > p) is itself a palindrome
    (i.e. appending the last letter just broke the p-bound)."""
    n = len(word)
    for length in range(p + 1, n + 1):
        seg = word[n - length:]
        if seg == seg[::-1]:
            return True
    return False


def main():
    a, p, L = map(int, sys.stdin.read().split()[:3])
    word = []
    for _ in range(L):
        placed = False
        for letter in range(a):
            cand = word + [letter]
            if not creates_violation(cand, p):
                word = cand
                placed = True
                break
        if not placed:
            break  # dead end: every letter would violate the bound
    sys.stdout.write("".join(str(x) for x in word) + "\n")


if __name__ == "__main__":
    main()
