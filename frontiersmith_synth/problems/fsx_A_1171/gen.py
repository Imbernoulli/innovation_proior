#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE public instance to stdout. Deterministic in testId only.

Format:
  testId
  n_terminal n_edges
  n_edges lines: u v R_nominal
  n_shown
  n_shown lines: s g Q V_s V_g      (measured terminal voltages under this excitation)

Everything printed here is PUBLIC: the netlist (with nominal resistances) and the shown
excitation patterns' measured readings. The true (possibly drifted) resistances and the
held-out excitation patterns are NEVER printed -- they live only inside common_gen.py,
recomputed independently (and identically) by verify.py from testId.
"""
import sys
from common_gen import build_instance, make_readings


def main():
    test_id = int(sys.argv[1])
    inst = build_instance(test_id)
    shown, _held = make_readings(inst)

    out = []
    out.append(str(test_id))
    out.append(f"{inst['n_terminal']} {inst['n_edges']}")
    for (u, v, R) in inst['edges']:
        out.append(f"{u} {v} {R}")
    out.append(str(len(shown)))
    for (s, g, Q, Vs, Vg) in shown:
        out.append(f"{s} {g} {Q} {Vs:.9f} {Vg:.9f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
