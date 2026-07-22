# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split("\n")
    p, k, r = map(int, data[0].split())
    R = 2 * r + 3
    lines = []
    for i in range(k):
        s = pow(R, i, p)
        lines.append(f"1 {s % p} 0 1")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
