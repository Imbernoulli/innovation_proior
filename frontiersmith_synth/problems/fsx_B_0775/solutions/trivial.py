# TIER: trivial
# Do-nothing baseline: no mixing stage, reduce = key mod M directly.
import sys

def main():
    sys.stdin.read()  # consume the instance (ignored)
    print("0 MODM")

if __name__ == "__main__":
    main()
