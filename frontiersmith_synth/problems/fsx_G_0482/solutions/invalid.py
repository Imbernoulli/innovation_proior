# TIER: invalid
# Emits a redundant (arithmetic) array: separation 1 repeats many times, so the
# non-redundancy check fails -> Ratio 0.0.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    print(" ".join(str(i) for i in range(n)))

if __name__ == "__main__":
    main()
