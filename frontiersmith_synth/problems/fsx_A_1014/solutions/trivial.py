# TIER: trivial
# Leave every valve fully open: x=0 on every pipe. This is exactly the
# checker's own internal baseline B, so it always scores ~0.1.
import sys


def main():
    tok = sys.stdin.read().split()
    n_hub = int(tok[0]); K = int(tok[1]); n_edges = int(tok[2])
    print(" ".join(["0.0"] * n_edges))


if __name__ == "__main__":
    main()
