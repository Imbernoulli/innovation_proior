import random
import sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    # Tiny cases so brute (O(n) per op) is fine. Bias toward negatives/zeros and
    # toward the empty / all-negative corners that drive the pitfall.
    mode = random.randint(0, 6)
    if mode == 0:
        n = 0            # empty corner
    elif mode == 1:
        n = 1            # single player: never forms a multi-member guild
    else:
        n = random.randint(2, 6)

    q = random.randint(0, 8)

    lines = []
    lines.append(f"{n} {q}")

    if n > 0:
        # value range chosen to make negatives, zeros and small positives common
        lo, hi = -5, 5
        # occasionally force all-negative balances
        if random.random() < 0.35:
            lo, hi = -6, -1
        vals = [str(random.randint(lo, hi)) for _ in range(n)]
        lines.append(" ".join(vals))

    for _ in range(q):
        if n >= 2 and random.random() < 0.6:
            u = random.randint(1, n)
            v = random.randint(1, n)
            lines.append(f"1 {u} {v}")
        else:
            lines.append("2")

    sys.stdout.write("\n".join(lines) + "\n")

main()
