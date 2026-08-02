# TIER: greedy
"""The obvious textbook fix for priority inversion: apply priority
INHERITANCE uniformly to every shared lock. This is exactly what most
engineers reach for first (it's the standard OS-course answer to "the
low-priority task holds a lock and a high-priority task is starving").
It never even looks at each lock's individual contention pattern, so it
does not know some locks are better served by a priority-ceiling
protocol."""
import sys


def main():
    data = sys.stdin.read().split()
    L = int(data[0])
    print(" ".join(["1"] * L))  # 1 = inherit, on every lock


if __name__ == "__main__":
    main()
