#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE Claim Ring Audit instance to stdout (format C).
Deterministic in testId only (see ring_truth.build). Hidden fraud/ring labels
are NEVER printed here -- they live only inside verify.py's re-derivation.

Output format:
  line 1: N M NC NP NA testId
  next N lines (one per claim, index = its position, 0-based):
    claimant provider adjuster amount plausibility cost
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ring_truth


def main():
    test_id = int(sys.argv[1])
    d = ring_truth.build(test_id)
    out = ["%d %d %d %d %d %d" % (d["N"], d["budget"], d["NC"], d["NP"], d["NA"], test_id)]
    for c in d["claims"]:
        out.append("%d %d %d %.2f %.4f %d" % (
            c["claimant"], c["provider"], c["adjuster"], c["amount"],
            c["plausibility"], c["cost"]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
