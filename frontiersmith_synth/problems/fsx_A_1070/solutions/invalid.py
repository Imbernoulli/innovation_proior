# TIER: invalid
# Emits a single, clearly-insufficient firing (minimum power, minimum duration,
# fired at the very start) -- never reaches the required threshold H at any
# real test in this problem's ladder. Must score 0 on every case.
import sys


def main():
    sys.stdin.read()  # consume input (not used)
    sys.stdout.write("1\n0 0 1 1\n")


if __name__ == "__main__":
    main()
