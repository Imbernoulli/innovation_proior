# TIER: trivial
"""Naive constant-rate release: open every gate to a fixed fraction of its
own max rate for the whole horizon, ignoring storage, forecast, routing
delay and every other dam entirely. The simplest thing anyone would try."""
import sys, json

def main():
    inst = json.load(sys.stdin)
    T = inst["t_steps"]
    FRAC = 0.72

    def const(dam):
        r = FRAC * dam["release_max"]
        return [r] * T

    ans = {
        "release1": const(inst["dam1"]),
        "release2": const(inst["dam2"]),
        "release3": const(inst["dam3"]),
    }
    print(json.dumps(ans))

if __name__ == "__main__":
    main()
