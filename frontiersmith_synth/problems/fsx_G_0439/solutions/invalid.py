# TIER: invalid
# Emits a broken sequence: 5 is not a sum of two earlier elements ({1,2}) -> score 0.
import sys

def main():
    sys.stdin.read()
    sys.stdout.write("1 2 5 999\n")

if __name__ == "__main__":
    main()
