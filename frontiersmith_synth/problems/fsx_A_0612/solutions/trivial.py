# TIER: trivial
# Do nothing: emit the all-zero source field. Reproduces the checker baseline B = ||y||^2.
import sys

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    row = " ".join(["0"] * N)
    sys.stdout.write("\n".join([row] * N) + "\n")

if __name__ == "__main__":
    main()
