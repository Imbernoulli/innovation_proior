# TIER: invalid
# Emits an all-zero ledger: correct shape but entries are not in {-1,+1} -> checker scores 0.
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it))
    row = " ".join("0" for _ in range(N))
    sys.stdout.write("\n".join(row for _ in range(N)) + "\n")

main()
