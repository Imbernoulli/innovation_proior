# TIER: trivial
# Submits "nothing changes": r=0, fee=0, theta=0. Reproduces the checker's
# own flat-rollout baseline exactly.
import sys

def main():
    sys.stdin.read()  # consume input (unused)
    print("0 1 0 0")

if __name__ == "__main__":
    main()
