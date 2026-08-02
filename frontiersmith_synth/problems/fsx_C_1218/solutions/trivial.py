# TIER: trivial
# "Do nothing": for every OLD message type, emit zero NEW messages (L=0). This is exactly
# the checker's own internal baseline construction, so it reproduces B and scores ~0.1.
import sys


def main():
    toks = sys.stdin.read().split()
    M_OLD = int(toks[2])
    out = [str(M_OLD)]
    for _ in range(M_OLD):
        out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
