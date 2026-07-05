# TIER: invalid
# Emits a single all-zero bilinear term: correct token count but reconstructs the
# zero tensor, which never equals a dense structure tensor -> checker scores 0.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it)); q = int(next(it)); s = int(next(it))
    zeros = [0] * (p + q + s)
    sys.stdout.write("1\n" + " ".join(map(str, zeros)) + "\n")

if __name__ == "__main__":
    main()
