import sys
import random

# Wider-spread generator variant (more points, larger coord range) used to stress
# the BIT-of-BIT x-rank routing and inner-y compression. Same online encoding.

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rnd = random.Random(seed * 2654435761 + 12345)

    n = rnd.randint(1, 40)
    q = rnd.randint(1, 60)
    COORD = rnd.choice([6, 15, 30, 60, 100])
    WMAX = rnd.choice([1, 5, 25, 100, 1000])

    px = [rnd.randint(-COORD, COORD) for _ in range(n)]
    py = [rnd.randint(-COORD, COORD) for _ in range(n)]
    w0 = [rnd.randint(-WMAX, WMAX) for _ in range(n)]

    lines = [f"{n} {q}"]
    for i in range(n):
        lines.append(f"{px[i]} {py[i]} {w0[i]}")

    w = w0[:]
    last = 0
    for _ in range(q):
        if rnd.random() < 0.4:
            idx = rnd.randint(0, n - 1)
            d = rnd.randint(-WMAX, WMAX)
            w[idx] += d
            lines.append(f"1 {idx} {d}")
        else:
            X1 = rnd.randint(-COORD, COORD); X2 = rnd.randint(-COORD, COORD)
            Y1 = rnd.randint(-COORD, COORD); Y2 = rnd.randint(-COORD, COORD)
            if X1 > X2: X1, X2 = X2, X1
            if Y1 > Y2: Y1, Y2 = Y2, Y1
            s = 0
            if X1 <= X2 and Y1 <= Y2:
                for i in range(n):
                    if X1 <= px[i] <= X2 and Y1 <= py[i] <= Y2:
                        s += w[i]
            a = X1 ^ last; b = Y1 ^ last; c = X2 ^ last; e = Y2 ^ last
            lines.append(f"2 {a} {b} {c} {e}")
            last = s

    sys.stdout.write("\n".join(lines) + "\n")

main()
