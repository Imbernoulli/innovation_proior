# TIER: greedy
# The OBVIOUS approach an average strong coder writes first: read the nominal
# gradient, note that the stripe boundaries sit at the 25% and 75% positions, look
# up the ABSOLUTE morphogen concentration there, and threshold every future cell at
# those two concentrations.  Perfect on the nominal embryo -- but the network reads
# ABSOLUTE concentration, so once the next embryo rescales amplitude / offset /
# slope the fixed thresholds land in the wrong place and the stripes drift or
# collapse.  This is the trap: point-calibrated absolute readout.
import sys, json

inst = json.load(sys.stdin)
field = inst["field"]
L = inst["L"]
q1 = L // 4
q3 = (3 * L) // 4
c1 = field[q1]
c2 = field[q3]
print(json.dumps({"feature": "absolute", "smooth": 0, "cuts": [c1, c2]}))
