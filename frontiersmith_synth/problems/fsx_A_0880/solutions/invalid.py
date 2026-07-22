# TIER: invalid
"""Broken candidate: emits a schedule whose times run BACKWARDS (arrival before
departure), which must be rejected by the checker for every move."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    out_moves = []
    for mv in inst["moves"]:
        out_moves.append({
            "id": mv["id"],
            "path": [mv["src"], mv["dst"]],
            "times": [mv["release"], mv["release"] - 5],
        })
    print(json.dumps({"moves": out_moves}))


main()
