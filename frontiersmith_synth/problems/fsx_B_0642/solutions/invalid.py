# TIER: invalid
# Emits a word that starts "00" -- an immediate square factor (period 1) -- so the
# checker's feasibility gate must reject it: scores 0.
import sys


def main():
    data = sys.stdin.read().split()
    L = int(data[0])
    N = min(L, 40) if L > 0 else 0
    if N < 2:
        N = min(L, 2)
    word = "00" + "1" * (N - 2) if N >= 2 else "0" * N
    print(N)
    print(word)


if __name__ == "__main__":
    main()
