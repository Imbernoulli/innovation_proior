# TIER: trivial
# "Do nothing": keep the given initial boost allocation for the entire horizon
# (so zero changeovers are ever paid) and split the buffer budget evenly across
# the gaps. This is the textbook do-nothing reference point.
import sys, json


def main():
    inst = json.load(sys.stdin)
    K, T = inst["K"], inst["T"]
    initial = inst["initial_alloc"]
    budget = inst["buffer_budget"]

    alloc = [list(initial) for _ in range(T)]

    n = K - 1
    base = budget // n
    rem = budget % n
    buffers = [base + (1 if j < rem else 0) for j in range(n)]

    print(json.dumps({"alloc": alloc, "buffers": buffers}))


if __name__ == "__main__":
    main()
