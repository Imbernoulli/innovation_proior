# TIER: invalid
# Emits relays far outside the unit cube -> feasibility gate rejects -> score 0.
import sys

def main():
    tok = sys.stdin.read().split()
    m = int(tok[0])
    out = []
    for i in range(m):
        out.append("5.0 5.0 5.0")
    sys.stdout.write("\n".join(out) + "\n")

main()
