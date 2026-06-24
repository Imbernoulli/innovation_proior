import sys
import random

# Adversarial generator: few distinct line labels reused across disconnected pairs,
# so the "ride the whole line for free" mistake is exposed (a line splits into components).
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    n = random.randint(2, 9)
    m = random.randint(0, 18)
    num_lines = random.randint(1, 3)   # very few labels -> heavy reuse / disconnected segments
    line_pool = random.sample(range(0, 8), num_lines)
    lines = [f"{n} {m}"]
    for _ in range(m):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        c = random.choice(line_pool)
        lines.append(f"{u} {v} {c}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
