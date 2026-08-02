# TIER: greedy
# The "textbook" recipe: every epoch, find whichever station currently has the
# highest base cycle time (today's theoretical bottleneck) and dump the ENTIRE
# boost pool onto it. This ignores (a) whether the station even benefits much
# from extra boost, (b) the changeover cost of moving boost units again, and
# (c) where the bottleneck is headed next -- it always chases the CURRENT
# reading. Buffers are split evenly (no anticipation of where migration will
# stress the line).
import sys, json


def main():
    inst = json.load(sys.stdin)
    K, T = inst["K"], inst["T"]
    P = inst["P"]
    base_cycle = inst["base_cycle"]
    budget = inst["buffer_budget"]

    alloc = []
    for t in range(T):
        row = [0] * K
        b = max(range(K), key=lambda i: base_cycle[t][i])
        row[b] = P
        alloc.append(row)

    n = K - 1
    base = budget // n
    rem = budget % n
    buffers = [base + (1 if j < rem else 0) for j in range(n)]

    print(json.dumps({"alloc": alloc, "buffers": buffers}))


if __name__ == "__main__":
    main()
