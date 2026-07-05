# TIER: trivial
# Ship only the prefilled slots -- reproduces the checker's baseline B (~0.1).
import sys


def main():
    data = [ln.split() for ln in sys.stdin.read().splitlines()]
    data = [t for t in data if t]
    N = int(data[0][0])
    P = [[int(x) for x in data[1 + i]] for i in range(N)]
    out = "\n".join(" ".join(str(P[r][c]) for c in range(N)) for r in range(N))
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
