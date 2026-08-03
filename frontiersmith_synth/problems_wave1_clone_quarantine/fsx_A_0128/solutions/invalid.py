# TIER: invalid
# Emits a single bogus channel that does not reconstruct the tensor -> checker
# reconstruction gate fails -> Ratio 0.0.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    outp = ["1", " ".join(["1"] * (a + b + c))]
    sys.stdout.write("\n".join(outp) + "\n")

if __name__ == "__main__":
    main()
