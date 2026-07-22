# TIER: invalid
# Emits frequencies far outside the forbidden-low-band constraint -> checker rejects.
import sys

def main():
    d = sys.stdin.read().split()
    # ignore instance; produce an out-of-band, garbage triple
    print("0 9.999999 0.0")

main()
