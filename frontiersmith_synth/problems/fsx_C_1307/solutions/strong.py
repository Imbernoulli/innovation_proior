# TIER: strong
# THE INSIGHT: never chase the trajectory. Because prey's own growth and
# starvation-mortality only feel vegetation from tauV steps back, any
# controller that reacts to *today's* prey level is reacting on the wrong
# clock -- by the time it sees a problem (or sees prey "recovering"), the
# vegetation damage that will decide prey's fate tauV steps later is already
# locked in. So commit, open-loop, to ONE small constant cull rate for the
# whole horizon: gentle enough that vegetation regrowth (rV) always keeps
# pace with the extra grazing pressure a partially-released prey population
# adds, so the overshoot-then-delayed-starvation cascade never triggers.
#
# The calibration reads the instance's own delay length tauV (the dominant
# driver of how much headroom a sustained cull has before it destabilizes
# the cascade -- a longer maturation/gestation lag means more time for a
# too-large release to compound before its consequence is even felt) and
# scales gently by the predator's own regeneration rate (rPr) and prey's
# starting abundance (H0): a faster-regenerating predator population, or a
# prey population starting further from its own carrying capacity, can
# absorb a slightly larger constant cull without tipping into the same
# overshoot dynamics that punish reactive correction.
import sys, json

# reference point the tauV-indexed magnitudes below were calibrated against
_REF_RPR = 0.35
_REF_H0 = 0.20


def calibrated_cull(inst):
    rPr, H0 = inst["rPr"], inst["H0"]
    tauV = inst["tauV"]
    cull_max = inst["cull_max"]

    if tauV <= 2:
        base = 0.10
    elif tauV == 3:
        base = 0.16
    else:
        # each additional unit of delay beyond the short-delay regime
        # shrinks the safe sustained magnitude further
        base = 0.065 * (3.0 / (tauV - 1))

    scale = (rPr / _REF_RPR) * (H0 / _REF_H0)
    c = base * scale
    lo, hi = 0.02, 0.45 * cull_max
    return max(lo, min(hi, c))


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    c = calibrated_cull(inst)
    print(json.dumps({"cull": [c] * T}))


if __name__ == "__main__":
    main()
