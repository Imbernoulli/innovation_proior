# TIER: trivial
"""Single flat fill: pick the one palette value closest to the overall mean density
implied by the given sinogram, and print an N x N grid of that constant value.
This reproduces the checker's own baseline construction exactly."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); _test_id = int(next(it)); R = int(next(it))
    P = int(next(it))
    palette = [int(next(it)) for _ in range(P)]
    K = int(next(it))
    _angles = [int(next(it)) for _ in range(K)]
    sino = []
    for _ in range(K):
        sino.append([int(next(it)) for _ in range(R)])

    tot = sum(sum(row) for row in sino)
    avg = tot / (K * N)
    best = min(palette, key=lambda p: abs(p - avg))

    out = []
    line = " ".join(str(best) for _ in range(N))
    for _ in range(N):
        out.append(line)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
