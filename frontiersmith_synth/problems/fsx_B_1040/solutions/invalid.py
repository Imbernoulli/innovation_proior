# TIER: invalid
# Emits a garbage artifact: duplicate anchor positions (all A anchors piled
# on the very first open cell). Must be rejected by the checker's
# distinctness check -> Ratio: 0.0.
import sys


def main():
    data = sys.stdin.read().splitlines()
    W, H, A, K, T = map(int, data[0].split())
    grid = data[1:1 + H]
    r0 = c0 = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '.':
                r0, c0 = r, c
                break
        if r0 is not None:
            break
    out = []
    for _ in range(A):
        out.append(f"{r0} {c0}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
