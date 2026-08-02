# TIER: greedy
# The obvious "control-theory" recipe: react to the herbivore/prey level.
# Cull the predator hard while prey is below a target fraction of its
# carrying capacity, ease off once it recovers. Since every coefficient of
# the (stated, fixed) recurrence is public, this program can self-simulate
# forward exactly as the evaluator will replay it, and derive its own
# causal reactive schedule offline before printing the whole array.
#
# THE TRAP: predator relief is felt by prey IMMEDIATELY (predation drops the
# moment culling starts), so prey overshoots past what current vegetation
# regrowth can support long before the controller sees any problem -- and
# the resulting vegetation crash only reaches prey's OWN growth/mortality
# term tauV steps later, by which point the controller has already relaxed
# (prey looked fine) and cannot prevent the delayed starvation collapse
# through the population floor. Longer tauV -> worse.
import sys, json

TARGET = 0.5
KP = 2.5


def simulate_reactive(inst):
    T = inst["T"]
    rV, gH = inst["rV"], inst["gH"]
    rH, aPred, hHalf, mH = inst["rH"], inst["aPred"], inst["hHalf"], inst["mH"]
    rPr = inst["rPr"]
    tauV, tauH = inst["tauV"], inst["tauH"]
    fV = inst["floor"]["V"]; fH = inst["floor"]["H"]; fPr = inst["floor"]["Pr"]
    cull_max = inst["cull_max"]

    Vh = [inst["V0"]]; Hh = [inst["H0"]]; Prh = [inst["Pr0"]]
    cull = []
    for t in range(T):
        # react to the CURRENT simulated prey level -- textbook proportional
        # feedback control, no knowledge of the future.
        c = KP * (TARGET - Hh[t])
        c = max(0.0, min(cull_max, c))
        cull.append(c)

        Vd = Vh[t - tauV] if t - tauV >= 0 else Vh[0]
        Hd = Hh[t - tauH] if t - tauH >= 0 else Hh[0]
        Vt, Ht, Prt = Vh[t], Hh[t], Prh[t]

        graze = gH * Ht * Vt
        Vn = Vt + rV * Vt * (1 - Vt) - graze

        denom = Ht + hHalf
        predation = aPred * Prt * Ht / denom if denom > 1e-12 else 0.0
        Hn = Ht + rH * Ht * Vd * (1 - Ht) - mH * Ht * (1 - Vd) - predation

        Prn = Prt + rPr * Prt * Hd * (1 - Prt) - c * Prt

        Vn = max(0.0, min(2.0, Vn))
        Hn = max(0.0, min(2.0, Hn))
        Prn = max(0.0, min(2.0, Prn))
        if Vn < fV: Vn = 0.0
        if Hn < fH: Hn = 0.0
        if Prn < fPr: Prn = 0.0

        Vh.append(Vn); Hh.append(Hn); Prh.append(Prn)
    return cull


def main():
    inst = json.load(sys.stdin)
    cull = simulate_reactive(inst)
    print(json.dumps({"cull": cull}))


if __name__ == "__main__":
    main()
