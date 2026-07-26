# TIER: invalid
"""
Emits a garbage schedule: an out-of-range node id and a negative time.
Must be rejected by the checker (Ratio: 0.0).
"""
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]) if toks else 0
    out = []
    out.append("2")
    out.append(f"{n + 999} 0")   # node out of range
    out.append("0 -5")           # time out of range
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
