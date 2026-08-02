# TIER: greedy
"""The obvious first idea: heat as fast as the burner allows, the whole way, in one
segment.  Minimizes total minutes (and hence fuel) and is perfectly fine on a kiln load of
uniformly thin pieces -- but on a mixed-thickness load it drives the THICKEST piece's core
so far behind the surface while crossing the quartz/cristobalite inversion bands that it
cracks (loses most or all of its value).  This is exactly the trap the problem statement
describes: fastest-possible ramp saves fuel time but cracks the ware at the quartz
inversion."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    start = float(inst["start_temp"])
    target = float(inst["target_temp"])
    max_rate = float(inst["max_rate"])
    minutes = (target - start) / max_rate
    print(json.dumps([{"to_temp": target, "minutes": minutes}]))


main()
