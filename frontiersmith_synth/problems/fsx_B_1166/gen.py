#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE plume-source-inversion instance to stdout.

Deterministic in testId only (see plumelib.build_instance). Prints the VISIBLE half of
the instance: grid/physics parameters, the visible monitoring wells and their noisy
readings, and the mass budget. The TRUE source cells/rates and the held-out wells are
NEVER printed -- verify.py regenerates them from the testId embedded in line 1.

Format (stdout):
  line 1: testId N K MT S_MAX
  line 2: D vx vy               (diffusivity, wind x/y components)
  line 3: t_1 t_2 ... t_MT      (observation snapshot times)
  line 4: B_mass                (mass budget: upper bound on total release rate)
  next K lines: "row col r_1 r_2 ... r_MT"  (visible well grid cell + its noisy
                readings at each of the MT times)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plumelib import build_instance, MT, S_MAX  # noqa: E402


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    inst = build_instance(t)
    out = []
    out.append("%d %d %d %d %d" % (inst["test_id"], inst["N"], inst["K"], MT, S_MAX))
    out.append("%.6f %.6f %.6f" % (inst["D"], inst["vx"], inst["vy"]))
    out.append(" ".join("%.6f" % x for x in inst["times"]))
    out.append("%.6f" % inst["B_mass"])
    for (i, j), readings in zip(inst["visible_wells"], inst["vis_readings"]):
        row = ["%d" % i, "%d" % j] + ["%.6f" % v for v in readings]
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
