# TIER: greedy
"""Follow the average tempo of all players. At every beat, cue the (smoothed)
average of everyone's own inclination reading -- the natural first idea:
treat all players as equally informative and just react to what you hear
right now. This is stable on the metronomic warm-ups, but on a rubato
passage it drowns the one player who is actually carrying the phrase in
n-1 other players' noise (damping the swell), and it never compensates for
the players' own reaction latency (so what tracking it does achieve arrives
late)."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = int(inst["T"])
    obs = inst["observed"]           # T x n
    n = int(inst["n_players"])

    raw = [sum(row) / n for row in obs]

    # simple exponential smoothing to denoise -- still purely reactive
    alpha = 0.3
    cue = [0.0] * T
    cue[0] = raw[0]
    for t in range(1, T):
        cue[t] = alpha * raw[t] + (1 - alpha) * cue[t - 1]

    print(json.dumps(cue))


main()
