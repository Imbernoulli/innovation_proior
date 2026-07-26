# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split("\n")
    N, C, TMAX = (int(x) for x in data[0].split())
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        a_i, d_i = data[i].split()
        A[i] = int(a_i)

    out = []
    for i in range(1, N + 1):
        out.append(f"{A[i]} 1 {i}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
