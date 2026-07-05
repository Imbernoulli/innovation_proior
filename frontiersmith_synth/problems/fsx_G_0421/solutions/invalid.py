# TIER: invalid
# Emits a garbage, non-parseable artifact -> must score 0.
import sys


def main():
    sys.stdout.write("this is not a valid loss law !!!\n")


if __name__ == "__main__":
    main()
