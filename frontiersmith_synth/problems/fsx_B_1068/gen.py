#!/usr/bin/env python3
"""gen.py <testId> -- synthetic-lexicon sonnet instance for fsx_B_1068.

Prints:
  line 1: "W BUDGET"
  next W lines: "STRESS RHYME INIT"   (STRESS: 2-char string over {0,1}; RHYME: int class id;
                                       INIT: single uppercase letter, the word's initial phoneme)

Word index = 0-based line order after the header (0..W-1).

Deterministic: all randomness seeded from testId only.
"""
import sys
import random

NUM_CLASSES = 6            # 6 rhyme classes but 7 scheme pairs -> pigeonhole FORCES >=1
                            # class to serve >=2 pairs, for every possible strategy
HOT = "ABCDEF"              # HOT[c] = the "good" (0-inversion) initial phoneme of class c
BAD_LETTER = "J"           # every class's single fallback word uses this cold, unshared letter
JUNK_LETTERS = "ABCDEFGHIJ"

# difficulty ladder: number of "rich" classes (good_count=4) out of 6 total classes.
# the rest are "poor" classes (good_count=1). Smaller num_rich => scarcer trap. At num_rich<=2
# total good-word supply (3*num_rich+6) falls below the 14-word demand (7 pairs x 2 lines),
# forcing real reuse / fallback-word decisions, not just the baseline pigeonhole collision.
NUM_RICH_SCHEDULE = {
    1: 6, 2: 5, 3: 4, 4: 4, 5: 2, 6: 2, 7: 1, 8: 1, 9: 3, 10: 2,
}


def build_instance(test_id: int):
    rng = random.Random(20000 + 97 * test_id)
    num_rich = NUM_RICH_SCHEDULE.get(test_id, 4)
    rich_classes = set(range(num_rich))  # classes 0..num_rich-1 are rich (solver must discover
    # this from the data itself -- richness is NOT implied by class id, since junk words below
    # add same-class extra entries with unrelated properties, and solutions must recompute
    # per-class good/bad counts directly from the lexicon, not assume an ordering).

    words = []  # each: (stress, rhyme_class, init_letter)
    for c in range(NUM_CLASSES):
        good_count = 4 if c in rich_classes else 1
        for _ in range(good_count):
            words.append(("01", c, HOT[c]))
        # one costly fallback word per class (max hamming distance to "01")
        words.append(("10", c, BAD_LETTER))

    # noise: extra decoy words scattered across random classes / stresses / letters, scaling
    # mildly with test_id so later cases require more careful filtering of the real structure.
    junk_n = 3 + 2 * test_id
    stress_opts = ["00", "01", "10", "11"]
    for _ in range(junk_n):
        c = rng.randrange(NUM_CLASSES)
        st = rng.choice(stress_opts)
        letter = rng.choice(JUNK_LETTERS)
        words.append((st, c, letter))

    rng.shuffle(words)

    # inversion budget: generous enough to always be satisfiable (0 inversions is always
    # achievable, e.g. by repeating a single "01" word), tight enough to bind real strategies
    # that spend budget for lexical diversity / alliteration / entropy gain.
    budget = 8 + 2 * (NUM_CLASSES - num_rich)

    return words, budget


def main():
    test_id = int(sys.argv[1])
    words, budget = build_instance(test_id)
    out = [f"{len(words)} {budget}"]
    for st, c, letter in words:
        out.append(f"{st} {c} {letter}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
