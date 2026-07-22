# TIER: trivial
# Reproduce the checker's baseline: post the given cheap reference price on every ride.
import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    M = int(next(it)); N = int(next(it)); LAM = int(next(it)); T = int(next(it))
    ref = []
    for j in range(M):
        s_reg = int(next(it)); s_fast = int(next(it)); r = int(next(it))
        ref.append(r)
    sys.stdout.write("\n".join(str(r) for r in ref) + "\n")

main()
