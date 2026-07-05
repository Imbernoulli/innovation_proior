import sys
import random

# gen.py <testId>  ->  prints ONE grid-scheduling instance to stdout.
# testId 1..10 is a small->large difficulty ladder; all randomness is seeded by testId.
#
# Instance = an N x N reservation grid (rows = time slots, cols = tracks) with a set of
# BLOCKED cells (slots already committed / under maintenance) that may not be used.

def build(test_id):
    N = 8 + 2 * test_id                     # 10, 12, ..., 28
    rng = random.Random(90000 + test_id)    # deterministic per test
    p = 0.10 + 0.005 * test_id              # blocked fraction grows with difficulty
    n_block = round(N * N * p)
    blocked = set()
    while len(blocked) < n_block:
        blocked.add((rng.randrange(N), rng.randrange(N)))
    return N, sorted(blocked)


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(2)
    t = int(sys.argv[1])
    N, blocked = build(t)
    out = [f"{N} {len(blocked)}"]
    for (r, c) in blocked:
        out.append(f"{r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
