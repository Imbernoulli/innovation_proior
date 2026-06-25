import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 12)
    # small value range so many subarrays collide on the same sum (stresses counting)
    vlo, vhi = -4, 4
    a = [rng.randint(vlo, vhi) for _ in range(n)]

    # pick S sometimes as a real achievable subarray sum, sometimes random
    mode = rng.randint(0, 2)
    if mode == 0 and n > 0:
        i = rng.randint(0, n - 1)
        j = rng.randint(i, n - 1)
        S = sum(a[i:j + 1])
    elif mode == 1:
        S = 0
    else:
        S = rng.randint(-20, 20)

    out = [f"{n} {S}"]
    out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
