# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it)); M = int(next(it)); K = int(next(it))
    P = int(next(it)); W = int(next(it)); D = int(next(it))
    ref = next(it)
    positions = [int(next(it)) for _ in range(M)]

    C = max(2, M // 4)
    lines = [str(C)]
    for i in range(C):
        lines.append(f"{positions[i]} F")
    sys.stdout.write("\n".join(lines) + "\n")


main()
