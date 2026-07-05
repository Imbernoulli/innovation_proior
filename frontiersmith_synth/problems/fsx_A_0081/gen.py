import sys, random, itertools

def diag_strings(n):
    d = ["0" * n]
    for i in range(n):
        s = ["0"] * n
        s[i] = "1"
        d.append("".join(s))
    return d

def main():
    i = int(sys.argv[1])
    rng = random.Random(7000 + i)
    # difficulty ladder: n grows 3 -> 5, flooded fraction varies
    if i <= 3:
        n = 3
    elif i <= 6:
        n = 4
    else:
        n = 5
    frac = 0.10 + 0.12 * ((i - 1) % 3)   # 0.10, 0.22, 0.34 cycling

    total = 3 ** n
    diag = set(diag_strings(n))
    all_cells = ["".join(map(str, c)) for c in itertools.product("012", repeat=n)]
    pool = [c for c in all_cells if c not in diag]        # never flood the baseline
    b = int(frac * total)
    b = min(b, len(pool))
    flooded = rng.sample(pool, b)
    flooded.sort()

    out = [str(n), str(len(flooded))]
    out.extend(flooded)
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
