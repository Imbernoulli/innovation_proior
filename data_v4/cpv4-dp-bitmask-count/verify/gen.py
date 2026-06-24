import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 7)
    full = (1 << n) - 1 if n > 0 else 0

    # number of candidate squads
    m = rng.randint(0, 14)
    masks = []
    for _ in range(m):
        r = rng.random()
        if r < 0.12:
            x = 0  # empty squad (should be ignored)
        elif r < 0.22 and n > 0:
            # occasionally include an out-of-range bit to stress masking
            x = rng.randint(1, full) | (1 << (n + rng.randint(0, 2)))
        elif n == 0:
            x = 0
        else:
            x = rng.randint(1, full)
        masks.append(x)

    # occasionally duplicate some masks to stress dedup
    if masks and rng.random() < 0.5:
        dup = rng.randint(0, len(masks) - 1)
        masks.append(masks[dup])

    m = len(masks)
    out = [f"{n} {m}"]
    if masks:
        out.append(" ".join(str(x) for x in masks))
    else:
        out.append("")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
