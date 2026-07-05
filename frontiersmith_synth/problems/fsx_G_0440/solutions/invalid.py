# TIER: invalid
# Emits a syntactically valid program whose outputs are wrong (every output is
# tied to input 0), so the exact-equivalence gate fails -> score 0.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it)); n = int(next(it))
    for _ in range(m * n):
        next(it)
    lines = ["0"]                       # no gates
    lines.append(" ".join("0" for _ in range(m)))   # all outputs = input 0
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
