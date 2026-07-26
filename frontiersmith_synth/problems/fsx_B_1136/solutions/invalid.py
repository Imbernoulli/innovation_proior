# TIER: invalid
# Emits an infeasible cascade: a basin whose length is way over the per-basin cap.
import sys


def main():
    sys.stdin.read()
    print(1)
    print(1, 999999999, 1)


if __name__ == "__main__":
    main()
