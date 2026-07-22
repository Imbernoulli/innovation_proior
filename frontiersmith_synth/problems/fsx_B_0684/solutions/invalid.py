# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); K = int(next(it)); C = int(next(it)); R = int(next(it))
    target = [int(next(it)) for _ in range(n)]
    tools = []
    for _ in range(m):
        mx = int(next(it)); mn = int(next(it))
        tools.append((mx, mn))

    # A single "pass" that removes far more depth than tool 1 legally allows --
    # blatantly outside [minDepth, maxDepth] and also undercuts the target.
    bad_depth = tools[0][0] + R + 1000
    print(1)
    print(f"1 1 {bad_depth}")


if __name__ == "__main__":
    main()
