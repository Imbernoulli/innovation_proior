# TIER: invalid
"""Deliberately broken candidate: emits a harvest matrix with the wrong number
of rows AND negative / NaN values mixed in, and also tries to over-harvest
massively. Must score 0.0 on every instance via strict shape/type/finiteness
validation in the evaluator."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    Z = inst["n_zones"]
    # wrong number of rows (T - 1 instead of T) -- fails the shape check outright
    harvest = []
    for t in range(max(0, T - 1)):
        row = []
        for zi in range(Z):
            if zi % 2 == 0:
                row.append(-1000.0)                # negative -> invalid
            else:
                row.append(float("nan"))            # NaN -> invalid
        harvest.append(row)
    print(json.dumps({"harvest": harvest}))


if __name__ == "__main__":
    main()
