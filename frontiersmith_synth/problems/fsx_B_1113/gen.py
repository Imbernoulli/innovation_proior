import sys
import random

CLASSES = "HML"

# Fixed physical constants shared by every test case (still emitted in the input,
# a solver must read and exploit them -- only the *instance* r_i/class stream varies).
D = 2        # fix-crossing -> landing transit ticks (same for everyone)
FIXSEP = 2   # minimum gap between ANY two consecutive fix crossings (shared merge point)
# S[from][to]: minimum landing-time gap on the SAME runway when a `from`-class aircraft
# lands immediately before a `to`-class aircraft. Heavy-leads-Light is by far the
# costliest case (S[H][L] >> S[H][H]) -- wake vortex lingers for the light trailing
# aircraft, while two same-class (or light-leading) landings can follow closely.
S = [
    [3, 5, 20],  # from H: to H, to M, to L
    [3, 3, 6],   # from M
    [3, 3, 3],   # from L
]
FW = [3, 2, 1]   # fuel burn per tick held in the stack, by class H, M, L

# difficulty ladder: (N, pattern, gap). gap < FIXSEP for the trap patterns so that
# arrivals genuinely OUTPACE the shared fix -- a backlog of several ready-and-waiting
# aircraft of each class builds up, and reordering the crossing sequence within that
# backlog (not just picking a runway) is what a strategy can actually exploit.
#
# "maxmix" is the key trap: a THREE-class sequence built so that (as far as possible)
# NO two consecutive arrivals share a class. A naive per-item best-runway greedy that
# never reorders is defeated no matter how it splits the two runways, because with
# three interleaved classes a simple parity/period-2 runway split cannot separate the
# expensive H->L pairs the way it accidentally could with only two classes.
LADDER = {
    1:  (6,  "mixed",  4),
    2:  (10, "maxmix", 1),
    3:  (10, "clustered", 3),
    4:  (16, "roundrobin", 1),
    5:  (18, "maxmix", 1),
    6:  (20, "mixed",  3),
    7:  (24, "maxmix", 1),
    8:  (28, "roundrobin", 1),
    9:  (32, "maxmix", 1),
    10: (40, "maxmix", 1),
}


def build_instance(testId):
    N, pattern, gap = LADDER.get(testId, LADDER[1])
    rng = random.Random(1000003 * testId + 17)

    r = [0] * N
    cls = [0] * N

    if pattern == "maxmix":
        # trap: a maximally-scattered 3-class sequence (greedy "no two adjacent
        # equal" construction over near-equal H/M/L counts). Every FCFS-adjacent
        # pair differs in class, so a non-reordering strategy is forced to eat a
        # transition on essentially every landing; a strategy that reorders into
        # class runs can batch this into a handful of transitions total.
        bias = testId % 3
        base = N // 3
        counts = [base, base, base]
        counts[bias] += N - 3 * base
        last = -1
        for i in range(N):
            order = sorted(range(3), key=lambda c: (-counts[c], c))
            pick = None
            for c in order:
                if counts[c] > 0 and c != last:
                    pick = c
                    break
            if pick is None:
                pick = order[0]
            counts[pick] -= 1
            cls[i] = pick
            last = pick
        # burst arrivals: WAVE aircraft become ready SIMULTANEOUSLY every WAVEGAP
        # ticks. This is what actually creates a holding stack: several classes
        # are genuinely available at once, so a strategy has real freedom to pick
        # which one to send through the fix next (batching becomes a real choice,
        # not just a relabeling of the arrival order).
        WAVE, WAVEGAP = 5, 3
        for i in range(N):
            r[i] = (i // WAVE) * WAVEGAP
    elif pattern == "roundrobin":
        # trap: H, M, L cycling, arriving in simultaneous bursts (see maxmix) so
        # FCFS almost never repeats a class AND a real backlog exists to batch.
        WAVE, WAVEGAP = 5, 3
        for i in range(N):
            cls[i] = i % 3
            r[i] = (i // WAVE) * WAVEGAP
    elif pattern == "clustered":
        # already batched by class in arrival order -> greedy is not badly trapped here
        # (kept in the ladder to show the ladder isn't ALWAYS adversarial).
        thirds = [0] * (N // 3) + [1] * (N // 3) + [2] * (N - 2 * (N // 3))
        t = 0
        for i in range(N):
            cls[i] = thirds[i]
            r[i] = t
            t += gap
    else:  # "mixed": seeded pseudo-random arrivals/classes
        weights = [0.3, 0.3, 0.4]
        t = 0
        for i in range(N):
            u = rng.random()
            cls[i] = 0 if u < weights[0] else (1 if u < weights[0] + weights[1] else 2)
            t += rng.randint(1, gap)
            r[i] = t

    return N, D, FIXSEP, S, FW, r, cls


def main():
    testId = int(sys.argv[1])
    N, d, fixsep, S, fw, r, cls = build_instance(testId)
    out = []
    out.append("%d" % N)
    out.append("%d %d" % (d, fixsep))
    for row in S:
        out.append("%d %d %d" % tuple(row))
    out.append("%d %d %d" % tuple(fw))
    for i in range(N):
        out.append("%d %s" % (r[i], CLASSES[cls[i]]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
