import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(0, 12)
    # Keep value range small so windows frequently straddle the threshold,
    # which stresses the inclusive/exclusive boundary logic.
    vmax = random.choice([1, 2, 3, 5, 8])
    a = [random.randint(-vmax, vmax) for _ in range(n)]
    # D ranges over 0 .. (span+2) so we hit D=0, exact-equality, and large-D cases.
    span = (max(a) - min(a)) if a else 0
    D = random.randint(0, span + 2)

    out = [f"{n} {D}"]
    if n > 0:
        out.append(" ".join(map(str, a)))
    print("\n".join(out))

if __name__ == "__main__":
    main()
