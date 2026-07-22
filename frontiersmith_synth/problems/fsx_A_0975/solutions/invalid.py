# TIER: invalid
# Deliberately infeasible: claims to cut a self-loop edge that does not
# exist in the input graph. Must score 0.
import sys

def main():
    sys.stdin.read()
    sys.stdout.write("1\n1 1 1\n")

if __name__ == "__main__":
    main()
