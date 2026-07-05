# TIER: invalid
# Emits an infeasible layout: a milepost strictly outside [0,V] -> checker fails -> 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    k = int(data[0])
    V = int(data[1])
    # header says 2 stations, but one milepost is out of range
    sys.stdout.write("2\n0\n%d\n" % (V + 100))


if __name__ == "__main__":
    main()
