# TIER: greedy
"""The obvious first pass: standard maximum-information adaptive testing. At
every step, administer whichever remaining item has the highest Fisher
information at the current ability estimate (info_weight=1, exposure_weight=0
-- exposure never enters the score at all). This is the best-known recipe for
a SINGLE examinee in isolation, and it is what an average strong coder writes
first for a CAT problem. But across a whole season it repeatedly reaches for
the same locally-best handful of items for every examinee whose ability lands
near theirs -- so whenever true ability is concentrated (not spread out like a
generic reference population), those items blow through their exposure cap
early and get compromised, silently corrupting ability estimates for the rest
of the season. This is the trap the problem is built to expose."""
import sys, json


def main():
    json.load(sys.stdin)  # public instance unused: pure max-information, no exposure signal
    ans = {"info_weight": 1.0, "exposure_weight": 0.0,
           "exposure_shape": 1.0, "hint_trust": 0.0}
    print(json.dumps(ans))


if __name__ == "__main__":
    main()
