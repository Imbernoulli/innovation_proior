import sys

# Difficulty ladder testId 1..10 (deterministic table -- no RNG needed since
# every instance parameter is fixed by testId). S values are chosen with
# rich divisor structure (12,20,30,60,84,90,120,72,210) so that a solver
# who does not reason about gcd(R,S) has a good chance of picking curves
# that alias/collapse under the fixed sampling budget S -- this is the
# planted trap. M always leaves a generous gap above r_out so an S-coprime
# R can always be found within [C_i+2, M].
#
# columns: r_in r_out K Q S M
TABLE = {
    1:  (10,  50,  70, 4,  21,  90),
    2:  (20,  80,  90, 5,  56, 130),
    3:  (15,  95, 110, 5,  30, 150),
    4:  (30, 150, 110, 5,  78, 220),
    5:  (40, 200, 140, 6,  66, 280),
    6:  (50, 250, 160, 6, 112, 340),
    7:  (60, 280, 160, 6,  95, 370),
    8:  (25, 300, 140, 5,  75, 450),
    9:  (100,350, 170, 6,  90, 500),
    10: (80, 400, 190, 7, 128, 600),
}


def instance_for(i):
    if i in TABLE:
        return TABLE[i]
    # defensive fallback for any testId beyond the shipped ladder: scale the
    # largest case up deterministically by i (never randomness).
    base = TABLE[10]
    scale = 1 + (i - 10)
    r_in, r_out, K, Q, S, M = base
    r_in2 = r_in
    r_out2 = min(500, r_out + 10 * scale)
    M2 = min(700, M + 10 * scale)
    return (r_in2, r_out2, K, Q, S, M2)


def main():
    i = int(sys.argv[1])
    r_in, r_out, K, Q, S, M = instance_for(i)
    print(f"{r_in} {r_out} {K} {Q} {S} {M}")


if __name__ == "__main__":
    main()
