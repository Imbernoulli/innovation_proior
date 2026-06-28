#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
# Produces a single line: a lowercase string. Small alphabet sizes are chosen
# often so that the suffix-automaton clone/split path is exercised heavily
# (repeats and overlaps create the split cases). Occasionally emits an empty
# string to test the degenerate input.
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # ~2% of the time, emit an empty string (no token on the line).
    if rng.random() < 0.02:
        print("")
        return

    n = rng.randint(1, 40)
    # Bias toward tiny alphabets to maximize repeated structure / SAM splits.
    alpha = rng.choice([1, 1, 2, 2, 2, 3, 3, 4, 6, 26])
    letters = "abcdefghijklmnopqrstuvwxyz"[:alpha]
    s = "".join(rng.choice(letters) for _ in range(n))
    print(s)

if __name__ == "__main__":
    main()
