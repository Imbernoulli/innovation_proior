# TIER: trivial
# Do nothing: submit the empty graph (no diffusion at all, A = identity).
# This reproduces the checker's own no-diffusion baseline exactly.
import sys

def main():
    sys.stdin.read()  # consume input (unused)
    print(0)

if __name__ == "__main__":
    main()
