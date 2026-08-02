# TIER: trivial
"""Never retry: max_attempts=1 for every request. Reproduces the checker's
own baseline construction exactly, so this scores ~0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[1])
    for _ in range(N):
        print("1")


if __name__ == "__main__":
    main()
