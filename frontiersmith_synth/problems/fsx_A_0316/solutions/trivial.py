# TIER: trivial
# Reproduces the checker's internal baseline exactly (lightly-perturbed Legendre circulant) -> ~0.1.
import sys


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def baseline(p):
    M = [[1 if i == j else (legendre((j - i) % p, p) or 1) for j in range(p)] for i in range(p)]
    for t in range(2):
        i = (2 * t + 1) % p
        j = (5 * t + 3) % p
        M[i][j] = -M[i][j]
    return M


def main():
    N = int(sys.stdin.read().split()[0])
    M = baseline(N)
    out = "\n".join(" ".join(str(x) for x in row) for row in M)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
