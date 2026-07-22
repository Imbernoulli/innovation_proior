# TIER: greedy
# The "obvious first draft": allocate reservoir slots proportional to how often
# each class has been SEEN in the burn-in so far. Sounds reasonable ("sample more
# where there's more data") but ignores variance entirely and trusts small/zero
# counts at face value -- a rare-but-erratic class that showed up 0-2 times gets
# starved of reservoir slots even though its true variance may dominate the
# network's estimator error.
import sys, json


def main():
    inst = json.load(sys.stdin)
    S = inst["S"]; K = inst["K"]; burnin = inst["burnin"]
    counts = [0] * S
    for rec in burnin:
        s = rec["s"]
        if 0 <= s < S:
            counts[s] += 1
    total = sum(counts)
    if total == 0:
        alloc = [K / S] * S
    else:
        alloc = [K * c / total for c in counts]
    print(json.dumps({"alloc": alloc}))


if __name__ == "__main__":
    main()
