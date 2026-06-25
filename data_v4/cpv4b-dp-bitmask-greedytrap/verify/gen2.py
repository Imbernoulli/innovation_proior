import sys, random

# Denser generator: smaller costs, larger bursts, more overlap, so optimal
# covers frequently differ from the greedy ratio heuristic.
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    m = rng.randint(1, 5)
    k = rng.randint(1, 13)
    lines = [f"{m} {k}"]
    for _ in range(k):
        c = rng.randint(1, 30)
        # bias toward larger bursts
        t = rng.randint(1, m)
        chans = rng.sample(range(m), t)
        lines.append(" ".join([str(c), str(t)] + [str(x) for x in chans]))
    sys.stdout.write("\n".join(lines) + "\n")

main()
