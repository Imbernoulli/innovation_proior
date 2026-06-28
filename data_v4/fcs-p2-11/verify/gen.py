#!/usr/bin/env python3
"""Random + adversarial test generator for the word-break problem.

Usage: gen.py <seed>
Prints one test case to stdout in the judge's input format:

  n
  w_1 w_2 ... w_n
  s

The generator mixes several regimes so the differential test sees:
  * tiny alphabets (so accidental segmentations are common -> exercises DP),
  * cases built to fool greedy longest-match and greedy shortest-match,
  * fully random dictionaries/strings,
  * empty / degenerate inputs.
"""
import random
import sys


def emit(words, s):
    words = list(words)
    out = [str(len(words))]
    if words:
        out.append(" ".join(words))
    out.append(s)
    sys.stdout.write("\n".join(out) + "\n")


def gen_random(rng):
    alpha_size = rng.choice([1, 2, 2, 3, 3, 4])
    alphabet = "abcdefghij"[:alpha_size]
    n = rng.randint(0, 8)
    words = set()
    for _ in range(n):
        L = rng.randint(1, 4)
        words.add("".join(rng.choice(alphabet) for _ in range(L)))
    if not words and rng.random() < 0.5:
        words.add("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 3))))
    slen = rng.randint(0, 12)
    s = "".join(rng.choice(alphabet) for _ in range(slen))
    return words, s


def gen_built_from_words(rng):
    # Build s by concatenating dictionary words sometimes, so YES is reachable,
    # but also include distractor words to mislead greedy.
    alphabet = "ab"
    base = set()
    n = rng.randint(2, 6)
    for _ in range(n):
        L = rng.randint(1, 4)
        base.add("".join(rng.choice(alphabet) for _ in range(L)))
    base = list(base)
    parts = rng.randint(0, 5)
    s = "".join(rng.choice(base) for _ in range(parts)) if base else ""
    # maybe corrupt the string a touch
    if s and rng.random() < 0.3:
        i = rng.randrange(len(s))
        s = s[:i] + rng.choice(alphabet) + s[i + 1:]
    return set(base), s


def gen_greedy_trap(rng):
    # Classic longest-match trap families. The intended segmentation needs a
    # *shorter* word early so a later boundary lines up; greedy-longest grabs
    # the long word and gets stuck.
    families = [
        # dict, string
        (["a", "aa", "aaa"], "a" * rng.randint(1, 9)),
        (["ab", "abc", "cd", "d"], "abcd"),
        (["aaaa", "aaa", "a"], "a" * rng.randint(1, 11)),
        (["leet", "code", "leetcode", "le", "etcode"], "leetcode"),
        (["go", "goo", "good", "d", "dog"], "good" + rng.choice(["dog", "og", "d"])),
        (["x", "xx", "xxx", "xxxx"], "x" * rng.randint(1, 12)),
        # shortest-match trap: shortest word leads to dead end
        (["a", "ab", "b"], "ab"),
        (["a", "aab", "ab", "b"], "aab"),
    ]
    d, s = rng.choice(families)
    return set(d), s


def gen_degenerate(rng):
    choice = rng.randint(0, 4)
    if choice == 0:
        return set(), ""                       # no words, empty string -> YES
    if choice == 1:
        return set(), "abc"                     # no words, non-empty -> NO
    if choice == 2:
        return {"a"}, ""                        # words but empty s -> YES
    if choice == 3:
        return {"abc"}, "abc"                   # single exact match -> YES
    return {"a", "b"}, "c"                       # unmatchable letter -> NO


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    kind = rng.randint(0, 9)
    if kind <= 3:
        words, s = gen_random(rng)
    elif kind <= 5:
        words, s = gen_built_from_words(rng)
    elif kind <= 7:
        words, s = gen_greedy_trap(rng)
    else:
        words, s = gen_degenerate(rng)
    emit(words, s)


if __name__ == "__main__":
    main()
