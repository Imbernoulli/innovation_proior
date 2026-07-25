# TIER: invalid
# Emits schema-valid lines whose expressions reference unknown names -> the
# checker's strict validator rejects the program and prints Ratio: 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    K = int(toks[1]) if len(toks) > 1 else 3
    out = ["W%d = qwark ( rho ) + zibble" % (c + 1) for c in range(K)]
    print("\n".join(out))


if __name__ == "__main__":
    main()
