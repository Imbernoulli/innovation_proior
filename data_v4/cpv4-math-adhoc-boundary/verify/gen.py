import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    q = rng.randint(1, 8)
    lines = [str(q)]
    for _ in range(q):
        n = rng.randint(1, 60)
        # Sometimes pick a/b near the actual quotient values to hit boundaries,
        # sometimes pick freely (including degenerate / out-of-range).
        mode = rng.randint(0, 3)
        if mode == 0:
            a = rng.randint(0, n + 2)
            b = rng.randint(0, n + 2)
        elif mode == 1:
            # tight around a real quotient
            x = rng.randint(1, n)
            v = n // x
            a = v
            b = v
        elif mode == 2:
            a = rng.randint(0, n + 1)
            b = a  # single quotient value
        else:
            a = rng.randint(0, n + 3)
            b = rng.randint(0, n + 3)
        if a > b:
            a, b = b, a
        # occasionally make a==0 or b small to exercise corners
        lines.append(f"{n} {a} {b}")
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
