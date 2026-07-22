# TIER: trivial
# Grind-the-weakest: drill only the skill that starts with the lowest proficiency,
# for the entire montage. This is exactly the evaluator's own weak-baseline
# construction, so it scores ~0.1. Every other skill sits untouched by its own
# gain curve but still gets multiplicatively decayed T times by the interference
# from the one skill being hammered, so most skills end up near zero.
import sys, json


def main():
    inst = json.load(sys.stdin)
    p0 = inst["p0"]
    T = inst["T"]
    j0 = min(range(len(p0)), key=lambda i: p0[i])
    print(json.dumps({"sequence": [j0] * T}))


if __name__ == "__main__":
    main()
