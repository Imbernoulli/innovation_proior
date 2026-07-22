# TIER: invalid
# Emit an out-of-range token -> checker rejects as infeasible -> Ratio 0.0
import sys

def main():
    sys.stdin.read()
    sys.stdout.write("999999999\n")

if __name__ == "__main__":
    main()
