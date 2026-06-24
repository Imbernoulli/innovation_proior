import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(1, 7)
    # number of edges
    max_e = n * (n + 1) // 2 + 3
    m = random.randint(0, min(12, max_e))
    # small set of line labels so collisions (same line) happen often -> greedy traps appear
    num_lines = random.randint(1, 4)
    # allow non-contiguous labels to exercise compression
    line_pool = random.sample(range(0, 20), num_lines)

    lines = [f"{n} {m}"]
    for _ in range(m):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        # allow self loops occasionally? keep simple: allow u==v sometimes
        if random.random() < 0.1:
            v = u
        c = random.choice(line_pool)
        lines.append(f"{u} {v} {c}")

    sys.stdout.write("\n".join(lines) + "\n")

main()
