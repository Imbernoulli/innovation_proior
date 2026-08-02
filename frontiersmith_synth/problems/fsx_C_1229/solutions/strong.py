# TIER: strong
import sys, math

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    V = int(next(it)); T = int(next(it))
    variants = []
    for _ in range(V):
        fleet = int(next(it))
        p = float(next(it))
        reward = float(next(it))
        penalty = float(next(it))
        D = int(next(it))
        W = float(next(it))
        variants.append((fleet, p, reward, penalty, D, W))

    out = []
    for (fleet, p, reward, penalty, D, W) in variants:
        ev = (1.0 - p) * reward - p * penalty
        stages = []
        if ev > 0 and p > 0:
            # THE INSIGHT: size every stage against the SIGNAL LATENCY D, not against
            # the previous stage's apparent success. A per-step cohort of W/(D*p)
            # devices keeps exactly D such cohorts "in flight" at steady state, whose
            # summed risk sits right at the rollback-window cap W -- never over it,
            # never wastefully under it -- so nothing is ever rejected and the fleet
            # is refreshed at the fastest rate the latency allows.
            per_step = W / (D * p)
            c_step = max(1, math.floor(per_step))
            used = 0
            for t in range(T):
                c = min(c_step, fleet - used)
                if c <= 0:
                    break
                stages.append((t, c))
                used += c
        # ev <= 0: this variant is not worth updating at all -- skip it entirely
        # rather than burn risk budget (and rollback-window slots) on a variant whose
        # expected value per device is negative.
        out.append(str(len(stages)))
        for (tt, cc) in stages:
            out.append("%d %d" % (tt, cc))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
