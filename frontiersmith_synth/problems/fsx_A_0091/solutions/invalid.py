# TIER: invalid
import sys

def main():
    d = sys.stdin.read().split()
    m = int(d[0])
    # emit an explicit spoilage cascade: (0,0),(1,0),(0,1) is a corner with d=1
    cells = [(0, 0), (1, 0), (0, 1)]
    out = [str(len(cells))]
    out.extend("%d %d" % (w, t) for (w, t) in cells)
    sys.stdout.write("\n".join(out) + "\n")

main()
