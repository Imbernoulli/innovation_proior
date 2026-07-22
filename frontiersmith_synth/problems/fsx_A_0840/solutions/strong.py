# TIER: strong
# The insight: don't guess the hidden reward scale, DISCOVER it. Start modest,
# and every time a stagnation window shows the searcher has both stopped
# improving AND stopped accepting moves, escalate the temperature multiplicatively
# from wherever it currently is (so repeated failures compound into a genuine
# upward search for the scale that's actually needed), jump back to the best
# state found so far so escalation can't destroy progress, and try again. This
# is scale-free: it self-calibrates to whatever the hidden per-instance slope
# turns out to be, and gives every trap block many independent attempts across
# the run instead of the one attempt a fixed monotone schedule gets.
import sys, json

def main():
    inst = json.load(sys.stdin)
    steps = inst["steps"]
    max_events_cap = inst["max_events_cap"]

    window = 300
    max_events = min(max_events_cap, steps // window)

    policy = {
        "T0": 4.0,
        "alpha": 0.85,
        "window": window,
        "stagnation_window": window,
        "accept_floor": 0.99,       # essentially: reheat whenever genuinely stagnant
        "reheat_factor": 1.6,       # compounds across repeated escalations
        "restart_mode": "restart_best",
        "max_events": max_events,
    }
    print(json.dumps(policy))

if __name__ == "__main__":
    main()
