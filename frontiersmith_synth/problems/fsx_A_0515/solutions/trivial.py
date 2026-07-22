# TIER: trivial
# Reproduce the checker's baseline: post the constant reference price p0 every day.
import sys

def main():
    tok = sys.stdin.read().split()
    T = int(tok[0]); p0 = int(tok[5])
    sys.stdout.write(" ".join([str(p0)] * T) + "\n")

if __name__ == "__main__":
    main()
