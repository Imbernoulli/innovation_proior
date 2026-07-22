# TIER: invalid
# Emits an out-of-range plan (intensity index way beyond the grid) -> the
# checker's feasibility gate must reject it -> Ratio 0.0.
import sys

def main():
    d = sys.stdin.read().split()
    L = int(d[0])
    out = []
    for _ in range(L):
        out.append("999999 7")  # bad index + bad pit flag
    sys.stdout.write("\n".join(out) + "\n")

main()
