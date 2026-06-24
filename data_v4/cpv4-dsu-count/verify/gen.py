import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(1, 8)
    m = random.randint(0, 14)
    if n == 1:
        m = 0  # with a single router there are no valid (distinct-endpoint) cables
    lines = ["%d %d" % (n, m)]
    for _ in range(m):
        u = random.randint(1, n)
        v = random.randint(1, n)
        while v == u:
            v = random.randint(1, n)
        # duplicate edges on purpose to stress redundancy counting
        lines.append("%d %d" % (u, v))
    sys.stdout.write("\n".join(lines) + "\n")

main()
