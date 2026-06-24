import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(1, 3)
    # small periods so brute's heap merge stays cheap
    p = [rng.randint(1, 12) for _ in range(n)]
    # K small enough that brute pops a bounded number of pulses
    K = rng.randint(1, 40)
    out = [str(n), " ".join(map(str, p)), str(K)]
    sys.stdout.write("\n".join(out) + "\n")

main()
