# TIER: trivial
"""
Do the least possible: keep the rack in the jars' input-given numbering
(identity order) and never spend a single swap. This is exactly the
checker's own internal baseline construction.
"""
import sys


def main():
    data = sys.stdin.read().split()
    ptr = 0
    n = int(data[ptr]); ptr += 1
    m = int(data[ptr]); ptr += 1
    K = int(data[ptr]); ptr += 1
    P = int(data[ptr]); ptr += 1
    ptr += 2 * P
    ptr += m  # sequence unused by trivial

    out = []
    out.append(" ".join(str(x) for x in range(1, n + 1)))
    out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
