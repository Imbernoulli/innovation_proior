import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Mix of regimes to stress squares: small alphabet (many squares), tiny n,
    # and occasionally a slightly larger alphabet.
    mode = rng.randint(0, 3)
    if mode == 0:
        n = rng.randint(0, 12)
        alpha = "ab"
    elif mode == 1:
        n = rng.randint(0, 14)
        alpha = "ab"
    elif mode == 2:
        n = rng.randint(0, 12)
        alpha = "abc"
    else:
        n = rng.randint(0, 16)
        alpha = "a"  # all-equal: every even-length window is a square
    s = "".join(rng.choice(alpha) for _ in range(n))
    out = [str(n)]
    if n > 0:
        out.append(s)
    print("\n".join(out))

if __name__ == "__main__":
    main()
