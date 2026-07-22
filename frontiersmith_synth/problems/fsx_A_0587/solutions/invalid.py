# TIER: invalid
# Emits a garbage layout (all boats moored at berth 0 -> duplicate berths and
# reach-window violations).  Must score 0.
import sys

def main():
    toks = open(sys.argv[1]).read().split() if len(sys.argv) > 1 else sys.stdin.read().split()
    it = iter(toks)
    m = int(next(it)); n = int(next(it)); R = int(next(it))
    sys.stdout.write(" ".join("0" for _ in range(n)) + "\n")

if __name__ == "__main__":
    main()
