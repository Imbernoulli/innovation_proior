# TIER: trivial
"""Do nothing: reprint the original heightfield unedited. Always feasible
(zero edit cost, satisfies the slope constraint because the generator only
ever emits an already-feasible original), and reproduces the checker's own
baseline exactly -> Ratio ~= 0.1."""
import sys


def main():
    data = sys.stdin.read().split('\n')
    idx = 0
    R, C = map(int, data[idx].split()); idx += 1
    idx += 1  # S B (unused)
    out = []
    for _ in range(R):
        out.append(data[idx]); idx += 1
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
