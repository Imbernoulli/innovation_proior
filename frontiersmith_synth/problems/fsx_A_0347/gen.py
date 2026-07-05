import sys

# fsx_A_0347 -- Drone Delivery Swarm: Maximum Rendezvous Diversity
# Difficulty ladder testId 1..10: number of drones k grows small -> large.
# Schedule horizon M = 9*k (chosen so that NO perfect Sidon set of size k fits
# in [0,M] -- max |A+A| is genuinely open -- while the interval bound 2M+1 keeps
# the achievable sumset well below the saturation point of the score).
LADDER = {
    1: 8,
    2: 10,
    3: 12,
    4: 16,
    5: 20,
    6: 24,
    7: 32,
    8: 40,
    9: 48,
    10: 56,
}


def main():
    i = int(sys.argv[1])
    if i < 1:
        i = 1
    if i > 10:
        i = 10
    k = LADDER[i]
    M = 9 * k
    sys.stdout.write("%d %d\n" % (k, M))


if __name__ == "__main__":
    main()
