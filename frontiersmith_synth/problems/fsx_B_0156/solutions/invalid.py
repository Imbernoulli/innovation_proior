# TIER: invalid
# Emits a non-expression (contains a disallowed name / bad syntax) -> Ratio 0.0.
import sys


def main():
    sys.stdin.read()
    print("this is definitely not @@@ a valid formula")


if __name__ == "__main__":
    main()
