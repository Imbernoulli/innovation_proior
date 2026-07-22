# TIER: invalid
"""Deliberately infeasible: claims bookmark ids far past F (and a target
position past N), which violates the 1<=i<=F and 1<=q<=N range checks.
Must score 0."""
import sys


def main():
    data = sys.stdin.buffer.read().split()
    N = int(data[0]); M = int(data[1]); F = int(data[2])

    K = 3
    out = [str(K)]
    out.append(f"1 {F + 50} {N + 100}")   # finger id AND position both out of range
    out.append(f"2 {F + 50} {N + 100}")
    out.append(f"1 {F + 50} {N + 100}")   # also a duplicate (t,i) pair on top of it
    print("\n".join(out))


if __name__ == "__main__":
    main()
