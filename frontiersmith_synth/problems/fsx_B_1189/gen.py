#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE Echo Room instance to stdout (format C).
Deterministic in testId only (see roomgeo.build). Held-out microphones are
NEVER printed here -- they live only inside verify.py's re-derivation.

Output format:
  line 1: W K testId
  line 2: Sx Sy
  next K lines: mx my L t1 t2 ... tL   (L >= W; L>W lines carry decoys)
testId is an opaque ladder index (1..10) re-used by verify.py to
deterministically re-derive the SAME ground truth (including the two
held-out microphones this file never prints) -- it carries no information
about the room's actual geometry.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roomgeo


def main():
    test_id = int(sys.argv[1])
    d = roomgeo.build(test_id)
    S, W, K, given_mics, obs = d["S"], d["W"], d["K"], d["given_mics"], d["obs"]

    out = []
    out.append("%d %d %d" % (W, K, test_id))
    out.append("%.9f %.9f" % (S[0], S[1]))
    for i in range(K):
        mx, my = given_mics[i]
        readings = obs[i]
        toks = ["%.9f" % mx, "%.9f" % my, str(len(readings))]
        toks += ["%.9f" % t for t in readings]
        out.append(" ".join(toks))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
