# TIER: strong
import sys
from math import gcd


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); D = int(next(it)); R = int(next(it)); cap = int(next(it))
    chain = [int(next(it)) for _ in range(D)]
    free = [int(next(it)) for _ in range(R)]

    result = []

    # Chain: canonical phase-0 maximally-even construction for every nested layer.
    # 0 is common to every layer's arithmetic progression, so the subset hierarchy holds
    # for free -- no cleverness needed here, this part is forced by the structure.
    for k in chain:
        step = N // k
        result.append([(t * step) % N for t in range(k)])

    # The coarsest chain layer (fewest onsets) occupies the residue class 0 (mod s1);
    # because the layers are nested, EVERY chain onset that a free instrument could ever
    # hit without blowing the cap lives outside that coarsest grid once phase choice is
    # done right -- and, since cap == the chain depth in the tight cases, a free instrument
    # must never land on the coarsest grid at all.
    s1 = N // chain[0] if D >= 1 else N
    slack = cap - D  # how much room is left once the mandatory chain pileup is paid for

    placed_steps_phases = []  # (step, phase) of free instruments placed so far

    # The core insight: a maximally-even rhythm's evenness is ROTATION INVARIANT (it only
    # depends on the multiset of gaps, not on where the pattern starts). So instead of
    # trading evenness for collision-avoidance (the greedy jitter), we pick the PHASE of
    # each free instrument's arithmetic progression -- for free, at zero evenness cost --
    # using the CRT fact that two residue classes r (mod s) and r' (mod s') are disjoint
    # iff r != r' (mod gcd(s, s')). We scan for the first phase that lands in a distinct
    # residue, modulo every relevant shared factor, from the chain's coarsest grid and
    # from every previously placed free instrument.
    for idx, f in enumerate(free):
        step = N // f
        if step <= 1:
            phase = 0
        else:
            # Among the CRT-safe candidates we don't just take the first one: we search
            # outward from a preferred anchor spread across [1, step) (roughly
            # idx-th of (R+1) equal slices) so that, among many equally collision-free
            # phases, the one chosen also tends to disperse this layer's onsets away from
            # the other layers' -- picking up some of the entropy bonus too, without
            # spending any evenness (every candidate here still gives a perfect AP).
            anchor = max(1, (step * (idx + 1)) // (R + 1))
            order = [anchor]
            for d in range(1, step):
                if anchor + d < step:
                    order.append(anchor + d)
                if anchor - d >= 1:
                    order.append(anchor - d)
            phase = None
            for cand in order:
                ok = True
                if D >= 1 and slack < 1:
                    # cap == chain depth exactly: the coarsest chain grid is already at
                    # cap, so a free onset landing there would overflow it -- must avoid.
                    g = gcd(step, s1)
                    if g > 0 and cand % g == 0:
                        ok = False
                if ok:
                    for (s2, p2) in placed_steps_phases:
                        g = gcd(step, s2)
                        if g > 0 and (cand - p2) % g == 0:
                            ok = False
                            break
                if ok:
                    phase = cand
                    break
            if phase is None:
                phase = 0  # should not happen for the generated instances
        placed_steps_phases.append((step, phase))
        result.append([(phase + t * step) % N for t in range(f)])

    out = [" ".join(str(x) for x in ons) for ons in result]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
