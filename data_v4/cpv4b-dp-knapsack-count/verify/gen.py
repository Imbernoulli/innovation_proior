import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the brute force (enumerate all copy-tuples) stays cheap.
    n = rng.randint(1, 5)
    S = rng.randint(0, 18)

    # Mix of moduli: a small prime to force collisions, a non-prime, and a big
    # modulus that behaves like "no reduction" on these small counts.
    MOD = rng.choice([2, 3, 5, 7, 10, 13, 1000000007])

    lines = []
    lines.append(f"{n} {S} {MOD}")
    for _ in range(n):
        v = rng.randint(1, 8)
        c = rng.randint(0, 4)   # bounded supply, including 0 copies allowed
        lines.append(f"{v} {c}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
