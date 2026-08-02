# TIER: trivial
# Do-nothing baseline: predict zero efficacy and zero toxicity everywhere.
# Both curves are flat and equal, so the grader's own argmax over the dosing
# grid ties everywhere and (by first-touched-wins) resolves to dose 0 -- i.e.
# this reproduces the "give nothing" reference point exactly, the checker's
# calibrated floor.
import sys

sys.stdin.read()
print("EFFICACY 0")
print("TOXICITY 0")
