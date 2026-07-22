# TIER: greedy
"""
The obvious first attempt: process tasks in the given order; for each task, try its desks
from most valuable to least valuable and commit to the first one that still has room. This
is a completely reasonable single-pass heuristic -- and it is exactly the trap: a task's
locally-best desk can be the SAME desk a later task badly needs, and once it is spent there
is no going back.
"""
import sys, json


def main():
    inst = json.load(sys.stdin)
    N, M = inst["n_tasks"], inst["n_agents"]
    weight, value, capacity = inst["weight"], inst["value"], inst["capacity"]
    rem = list(capacity)
    assign = [None] * N
    for i in range(N):
        order = sorted(range(M), key=lambda k: (-value[i][k], k))
        for k in order:
            if weight[i][k] <= rem[k]:
                assign[i] = k
                rem[k] -= weight[i][k]
                break
        if assign[i] is None:
            # every desk (including overflow) should always have room somewhere; if not,
            # dump on the last desk anyway so we still emit a well-formed answer.
            assign[i] = M - 1
            rem[M - 1] -= weight[i][M - 1]
    print(json.dumps({"assign": [int(a) for a in assign]}))


if __name__ == "__main__":
    main()
