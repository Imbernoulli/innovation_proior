import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mostly generate well-formed small expressions; occasionally emit a
    # malformed / edge string to exercise the validation path.
    roll = rng.random()
    if roll < 0.08:
        # malformed: random characters / wrong shape
        choices = ['', 'T&', '&T', 'TT', 'TX', 'X', '&', 'T&&F', 'T|F|', 'TF']
        print(rng.choice(choices))
        return

    m = rng.randint(1, 7)              # number of literals (tiny: brute enumerates all)
    parts = []
    for i in range(m):
        parts.append(rng.choice(['T', 'F']))
        if i != m - 1:
            parts.append(rng.choice(['&', '|', '^']))
    print(''.join(parts))

main()
