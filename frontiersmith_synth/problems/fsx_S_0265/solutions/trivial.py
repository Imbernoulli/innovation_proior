# TIER: trivial
# Do-nothing interdictor: remove no links. Residual s-t path == original,
# so F == B and the score sits exactly at the calibrated baseline (~0.1).
import sys

def main():
    sys.stdin.read()  # instance ignored
    sys.stdout.write("0\n")

if __name__ == "__main__":
    main()
