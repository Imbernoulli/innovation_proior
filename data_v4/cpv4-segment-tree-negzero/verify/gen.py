import sys
import random

def main():
    seed = int(sys.argv[1])
    rng = random.Random(seed)

    n = rng.randint(1, 8)
    q = rng.randint(1, 12)

    # Value pool biased toward negatives and zeros to stress sign handling.
    def rand_val():
        roll = rng.random()
        if roll < 0.40:
            return rng.randint(-9, -1)   # negative
        elif roll < 0.65:
            return 0                     # zero
        else:
            return rng.randint(1, 9)     # positive

    lines = []
    lines.append(f"{n} {q}")
    a = [rand_val() for _ in range(n)]
    lines.append(" ".join(map(str, a)))

    for _ in range(q):
        if rng.random() < 0.5:
            # update
            i = rng.randint(0, n - 1)
            x = rand_val()
            lines.append(f"1 {i} {x}")
        else:
            # query l <= r
            l = rng.randint(0, n - 1)
            r = rng.randint(l, n - 1)
            lines.append(f"2 {l} {r}")

    sys.stdout.write("\n".join(lines) + "\n")

main()
