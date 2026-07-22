# TIER: invalid
# Emits out-of-range threading (value S+1) -> checker rejects -> 0.
import sys

def main():
    tok = sys.stdin.read().split()
    N = int(tok[0]); S = int(tok[1]); T = int(tok[2])
    threading = [S + 1] * N          # out of range
    treadling = [1] * N
    out = []
    out.append(" ".join(map(str, threading)))
    out.append(" ".join(map(str, treadling)))
    for t in range(T):
        out.append(" ".join("1" for _ in range(S)))
    sys.stdout.write("\n".join(out) + "\n")

main()
