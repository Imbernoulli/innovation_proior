# TIER: trivial
# Plain weave (tabby): the checker's internal baseline construction. Scores ~0.1.
import sys

def main():
    tok = sys.stdin.read().split()
    N = int(tok[0]); S = int(tok[1]); T = int(tok[2])
    threading = [1 + (j % 2) for j in range(N)]
    treadling = [1 + (i % 2) for i in range(N)]
    out = []
    out.append(" ".join(map(str, threading)))
    out.append(" ".join(map(str, treadling)))
    for t in range(T):
        out.append(" ".join(str((s + t) % 2) for s in range(S)))
    sys.stdout.write("\n".join(out) + "\n")

main()
