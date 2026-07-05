# TIER: invalid
# Emits a syntactically valid but WRONG program: zero instructions, outputs point at the
# raw input registers (unit vectors), which never equal the dense target rows -> Ratio 0.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it)); n = int(next(it))
    # consume matrix (not used)
    for _ in range(m * n):
        next(it)
    outs = [min(i, n - 1) for i in range(m)]
    lines = ["0", " ".join(str(o) for o in outs)]
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
