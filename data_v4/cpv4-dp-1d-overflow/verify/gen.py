import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # tiny cases so the brute force (2^n enumeration) stays feasible
    n = rng.randint(0, 14)
    s = "".join(rng.choice("..x") for _ in range(n))  # bias toward working lamps
    out = [str(n)]
    if n > 0:
        out.append(s)
    print("\n".join(out))

main()
