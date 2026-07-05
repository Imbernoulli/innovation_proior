# TIER: invalid
# Activate a full hazard line {0,1,2} (indices 0,1,2 -> vectors differing only in
# coordinate 0: 0+1+2 == 0 mod 3, other coords 0) -> line-free check fails -> 0.0.
import sys

def main():
    _ = sys.stdin.read()
    sys.stdout.write("3\n0 1 2\n")

if __name__ == "__main__":
    main()
