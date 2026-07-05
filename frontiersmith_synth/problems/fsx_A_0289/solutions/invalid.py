# TIER: invalid
# Emits coordinates outside [0,1] -> feasibility gate must reject -> score 0.
import sys

def main():
    inp = sys.stdin.read().split()
    d = int(inp[0]); M = int(inp[1])
    out = []
    for i in range(M):
        out.append(" ".join("2.5" for _ in range(d)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
