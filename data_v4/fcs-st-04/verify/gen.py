#!/usr/bin/env python3
# Random small-case generator for minimum palindromic factorization.
# Usage: gen.py <seed>
# Prints a single string token (lowercase letters) to stdout.
#
# Mixes regimes to stress palindrome structure:
#   - tiny alphabets (lots of palindromes, long series chains)
#   - palindromic seeds and mirrored constructions
#   - fully random strings

import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    regime = rng.randint(0, 5)

    if regime == 0:
        # very small alphabet, short length
        L = rng.randint(0, 30)
        alpha = rng.randint(1, 2)
        s = "".join(chr(ord('a') + rng.randrange(alpha)) for _ in range(L))
    elif regime == 1:
        # small alphabet, medium length
        L = rng.randint(1, 60)
        alpha = rng.randint(1, 3)
        s = "".join(chr(ord('a') + rng.randrange(alpha)) for _ in range(L))
    elif regime == 2:
        # build by concatenating random palindromes
        parts = []
        total = 0
        alpha = rng.randint(1, 3)
        while total < rng.randint(1, 40):
            half = "".join(chr(ord('a') + rng.randrange(alpha))
                            for _ in range(rng.randint(0, 4)))
            mid = "" if rng.random() < 0.5 else chr(ord('a') + rng.randrange(alpha))
            pal = half + mid + half[::-1]
            if pal:
                parts.append(pal)
                total += len(pal)
        s = "".join(parts)
    elif regime == 3:
        # mirror a random core to force long palindromes
        alpha = rng.randint(1, 4)
        core = "".join(chr(ord('a') + rng.randrange(alpha))
                       for _ in range(rng.randint(0, 20)))
        if rng.random() < 0.5:
            s = core + core[::-1]
        else:
            s = core + (chr(ord('a') + rng.randrange(alpha))) + core[::-1]
    elif regime == 4:
        # larger alphabet random (few palindromes -> many singletons)
        L = rng.randint(1, 50)
        alpha = rng.randint(4, 8)
        s = "".join(chr(ord('a') + rng.randrange(alpha)) for _ in range(L))
    else:
        # possibly empty / single char
        L = rng.randint(0, 3)
        s = "".join(chr(ord('a') + rng.randrange(2)) for _ in range(L))

    sys.stdout.write(s)


if __name__ == "__main__":
    main()
