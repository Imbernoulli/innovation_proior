# TIER: trivial
# Do nothing: upgrade zero cells. Reproduces the checker's do-nothing baseline -> Ratio ~= 0.1.
import sys

def main():
    sys.stdin.read()          # consume the instance
    sys.stdout.write("0\n")   # zero upgraded cells

if __name__ == "__main__":
    main()
