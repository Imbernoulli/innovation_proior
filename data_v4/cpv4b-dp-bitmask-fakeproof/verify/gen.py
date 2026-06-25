import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(0, 6)
    FULL = (1<<n) - 1
    # number of crews
    m = rng.randint(0, 12)
    crews = []
    for _ in range(m):
        if n == 0:
            # no valid mask in [1, 0]; but contract says n>=1 typically; still emit something
            # we just skip crews when n==0 (mask must be >=1). Generate mask=1 is impossible.
            # To keep it well-formed, set m effectively 0 by breaking.
            break
        mk = rng.randint(1, FULL) if FULL >= 1 else 0
        c = rng.randint(0, 20)
        crews.append((mk, c))
    m = len(crews)
    out = []
    out.append(f"{n} {m}")
    for mk, c in crews:
        out.append(f"{mk} {c}")
    sys.stdout.write("\n".join(out) + "\n")

main()
