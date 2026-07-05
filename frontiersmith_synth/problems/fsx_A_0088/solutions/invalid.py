# TIER: invalid
# Emits one all-zero stage: correct token count but reconstructs the zero tensor,
# which never equals a nonzero gain tensor -> checker scores 0.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    zeros = [0] * (a + b + c)
    sys.stdout.write("1\n" + " ".join(map(str, zeros)) + "\n")

if __name__ == "__main__":
    main()
