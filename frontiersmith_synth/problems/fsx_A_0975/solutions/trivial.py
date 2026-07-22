# TIER: trivial
# Cut every required edge as its own individual pierce-cut-retract, visiting
# edges in the exact order they are listed in the input. This is the
# "do nothing clever" reference construction (matches the checker's internal
# baseline B exactly).
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    def rdi(): return int(next(it))
    n = rdi(); m = rdi(); P = rdi()
    for _ in range(n):
        rdi(); rdi()
    edges = []
    for _ in range(m):
        u = rdi(); v = rdi()
        edges.append((u, v))
    K = rdi()
    for _ in range(K):
        rdi(); rdi(); rdi(); rdi()

    out = [str(m)]
    for (u, v) in edges:
        out.append(f"1 {u} {v}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
