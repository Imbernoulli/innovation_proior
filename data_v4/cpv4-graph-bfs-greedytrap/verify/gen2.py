import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(1, 12)
    max_e = 25
    m = random.randint(0, max_e)
    num_lines = random.randint(1, 6)
    line_pool = random.sample(range(-5, 1000000000, 7), num_lines) if num_lines <= 5 else list(range(num_lines))
    line_pool = random.sample(range(0, 30), num_lines)

    lines = [f"{n} {m}"]
    for _ in range(m):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if random.random() < 0.15:
            v = u
        c = random.choice(line_pool)
        lines.append(f"{u} {v} {c}")

    sys.stdout.write("\n".join(lines) + "\n")

main()
