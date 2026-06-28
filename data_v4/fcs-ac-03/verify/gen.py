import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Keep #inserts small so brute's 2^m subset enumeration is cheap.
    max_inserts = rng.randint(0, 14)
    # Sometimes use a small bit width so memberships hit YES and values collide.
    bitw = rng.choice([1, 2, 3, 4, 5, 6, 8, 12, 20, 40, 60])
    hi = (1 << bitw) - 1
    if hi == 0:
        hi = 1  # bitw could be tiny; keep at least value 1 reachable

    inserts_done = 0
    ops = []

    # Total queries: a handful.
    q = rng.randint(1, 30)
    for _ in range(q):
        # Bias toward inserts early, but allow queries any time.
        choices = [2, 3, 4]
        if inserts_done < max_inserts:
            choices = [1, 1, 1, 2, 3, 4]  # bias toward inserts
        t = rng.choice(choices)
        if t == 1:
            x = rng.randint(0, hi)
            ops.append(f"1 {x}")
            inserts_done += 1
        elif t == 2:
            ops.append("2")
        elif t == 3:
            # Mix in-span-likely and random x.
            x = rng.randint(0, hi if rng.random() < 0.7 else (1 << 60) - 1)
            ops.append(f"3 {x}")
        else:  # t == 4
            # k chosen to sometimes exceed 2^rank (=> -1) and sometimes be valid.
            kmax = (1 << max(0, inserts_done)) + 2
            k = rng.randint(0, kmax)  # 0 -> out of range -> -1
            ops.append(f"4 {k}")

    print(len(ops))
    print("\n".join(ops))

main()
