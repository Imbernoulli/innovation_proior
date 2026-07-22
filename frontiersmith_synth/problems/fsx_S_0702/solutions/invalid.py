# TIER: invalid
"""Deliberately malformed answer: repeats a facility index (violates the
'K DISTINCT site indices' contract), so the evaluator must reject it -> 0.0
on every instance."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    k = inst["k"]
    facilities = [0] * k              # duplicate indices -> invalid
    assign = [0] * inst["n_demand"]
    print(json.dumps({"facilities": facilities, "assign": assign}))


if __name__ == "__main__":
    main()
