import sys, random

# testId 1..10 -> difficulty ladder over n (recycling sorting-stage count)
# and a growing list of blocked routes, seeded deterministically by testId.
NS = [4, 4, 5, 5, 6, 6, 7, 7, 7, 8]

def main():
    t = int(sys.argv[1])
    n = NS[(t - 1) % len(NS)]
    rng = random.Random(1000 + t)
    space = 3 ** n
    # number of blocked routes grows with difficulty but stays modest
    nb = min(space - 1, 2 * t)
    blocked = set()
    while len(blocked) < nb:
        blocked.add(rng.randrange(space))
    out = [f"{n} {len(blocked)}"]
    for e in sorted(blocked):
        digs = []
        x = e
        for _ in range(n):
            digs.append(x % 3)
            x //= 3
        out.append(' '.join(map(str, digs)))
    sys.stdout.write('\n'.join(out) + '\n')

main()
