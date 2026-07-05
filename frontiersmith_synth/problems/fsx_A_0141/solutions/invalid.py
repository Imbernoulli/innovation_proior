# TIER: invalid
# Emit three collinear watchtowers 0..0 / 10..0 / 20..0 (they sum to 0 in every
# coordinate = a raiding sightline). Feasibility gate must score this 0.0.
import sys


def main():
    raw = sys.stdin.read().splitlines()
    n = int(raw[0].split()[0])
    a = '0' * n
    b = '1' + '0' * (n - 1)
    c = '2' + '0' * (n - 1)
    sys.stdout.write('\n'.join([a, b, c]) + '\n')


if __name__ == "__main__":
    main()
