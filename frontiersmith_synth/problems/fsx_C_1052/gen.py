import sys

# Difficulty ladder: (a, p, L) triples, indexed by testId 1..10.
# a      = alphabet size (letters are the decimal digits 0..a-1)
# p      = palindrome-length bound (no factor of the output may be a
#          palindrome of length STRICTLY GREATER than p)
# L      = target length (the participant's word may be at most L long)
#
# Deterministic: purely table-driven by testId, no RNG needed at all.
#
# All 10 cases use the binary alphabet (a=2) on purpose: over alphabets of
# size >= 3 the checker's own naive cyclic baseline (0,1,2,0,1,2,...) never
# contains a palindromic factor at all, so B=L exactly and EVERY valid
# construction -- trivial, greedy, or a tuned morphism -- ties the baseline
# at Ratio=0.1 (verified by direct simulation while authoring this problem).
# Those alphabet sizes carry no discriminative signal, so the whole ladder
# is built in the one alphabet where the greedy dead-end trap is real: over
# a binary alphabet the naive cyclic baseline is itself a nested palindrome
# (dies at length p) and a smallest-letter-first greedy scan dead-ends after
# roughly pi*p characters, while a length-7 substitution table tuned to the
# bound reaches the full target length L (see solutions/strong.py).
CASES = {
    1:  (2, 6,  48),
    2:  (2, 8,  64),
    3:  (2, 10, 80),
    4:  (2, 14, 112),
    5:  (2, 18, 144),
    6:  (2, 22, 176),
    7:  (2, 26, 208),
    8:  (2, 32, 256),
    9:  (2, 40, 320),
    10: (2, 48, 384),
}


def main():
    i = int(sys.argv[1])
    a, p, L = CASES[((i - 1) % len(CASES)) + 1]
    sys.stdout.write("%d %d %d\n" % (a, p, L))


if __name__ == "__main__":
    main()
