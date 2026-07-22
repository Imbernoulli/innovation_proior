# TIER: trivial
# Reproduce the checker baseline: spam move 1 at increasing (capped) lengths.
import sys

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); m = int(d[1]); k = int(d[2]); L = int(d[3])
    out = []
    for i in range(k):
        ln = min(i, L)
        out.append(" ".join(["1"] * ln) if ln > 0 else "0")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
