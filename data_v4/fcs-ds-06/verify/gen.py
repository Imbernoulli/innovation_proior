import sys
import random

# Generator for fcs-ds-06.
# Produces a self-consistent input where query rectangle coordinates are
# XOR-encoded with the previous answer (forced-online). To encode correctly we
# must simulate the true answers ourselves while generating.

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rnd = random.Random(seed)

    n = rnd.randint(1, 12)
    q = rnd.randint(1, 18)

    # small coordinate / value ranges so brute is trivially correct
    COORD = rnd.choice([3, 5, 8, 12, 40])
    WMAX = rnd.choice([1, 3, 10, 50])

    px = [rnd.randint(0, COORD) for _ in range(n)]
    py = [rnd.randint(0, COORD) for _ in range(n)]
    w0 = [rnd.randint(-WMAX, WMAX) for _ in range(n)]

    lines = []
    lines.append(f"{n} {q}")
    for i in range(n):
        lines.append(f"{px[i]} {py[i]} {w0[i]}")

    # live weights for simulation
    w = w0[:]
    last = 0

    for _ in range(q):
        # bias toward queries so the online chain is exercised, but allow updates
        is_update = (rnd.random() < 0.35) and (n > 0)
        if is_update:
            idx = rnd.randint(0, n - 1)
            d = rnd.randint(-WMAX, WMAX)
            w[idx] += d
            lines.append(f"1 {idx} {d}")
        else:
            X1 = rnd.randint(0, COORD); X2 = rnd.randint(0, COORD)
            Y1 = rnd.randint(0, COORD); Y2 = rnd.randint(0, COORD)
            if X1 > X2: X1, X2 = X2, X1
            if Y1 > Y2: Y1, Y2 = Y2, Y1
            # occasionally produce an empty rectangle (X1>X2) to test the guard
            if rnd.random() < 0.1:
                X1, X2 = X2 + 1, X2
            # compute true answer to keep the encoding chain valid
            s = 0
            if X1 <= X2 and Y1 <= Y2:
                for i in range(n):
                    if X1 <= px[i] <= X2 and Y1 <= py[i] <= Y2:
                        s += w[i]
            # encode the four coordinates with the PREVIOUS answer
            a = X1 ^ last; b = Y1 ^ last; c = X2 ^ last; e = Y2 ^ last
            lines.append(f"2 {a} {b} {c} {e}")
            last = s

    sys.stdout.write("\n".join(lines) + "\n")

main()
