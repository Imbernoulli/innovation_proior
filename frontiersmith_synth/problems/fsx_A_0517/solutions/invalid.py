# TIER: invalid
# Emits a well-formed but WRONG circuit: every target is reported as x_0.
# Passes schema parsing but fails the exact-equivalence gate -> Ratio 0.0.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it)); n = int(next(it)); k = int(next(it))
    # zero instructions; point every output at input node 0
    sys.stdout.write("0\n")
    sys.stdout.write("out " + " ".join("0" for _ in range(k)) + "\n")

if __name__ == "__main__":
    main()
