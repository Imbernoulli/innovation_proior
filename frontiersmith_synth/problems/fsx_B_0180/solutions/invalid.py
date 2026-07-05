# TIER: invalid
# Emits an expression that references an undefined name (not in the allowed set),
# so the checker's strict validation rejects it -> score 0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("reservoir_temperature * 999\n")


if __name__ == "__main__":
    main()
