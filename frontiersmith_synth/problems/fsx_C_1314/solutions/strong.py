# TIER: strong
"""Weight players by structural role, and anticipate rather than follow.

Instead of averaging every player's reading uniformly, trust the ONE
channel the score says is carrying the phrase at each beat (the role
weights are exactly the score's melody annotation) -- a role-weighted
combination throws away the n-1 noisy, non-phrase-aware readings instead of
averaging them in. Lightly smooth that cleaner estimate to tame the
phrase-carrier's own sensor noise.

Because the conductor has already read through the whole passage before
performing it (the observed track spans the whole piece, not just "so
far"), the true insight is to STOP being reactive: don't cue beat t with
the estimate of beat t (every player will only realize it `latency` beats
later, so the ensemble will always lag the swell). Instead, cue beat t with
the (already known) estimate of a beat AHEAD, shifted forward by roughly
the ensemble's typical reaction latency, so that by the time the cue has
propagated through each player's own latency and inertia, the ensemble
actually arrives at the right tempo at the right beat."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = int(inst["T"])
    n = int(inst["n_players"])
    role = inst["role_weight"]        # T x n
    obs = inst["observed"]            # T x n
    latency = inst["latency"]         # length n

    # 1) role-weighted phrase estimate: trust the phrase-carrier's channel
    lead_est = [sum(role[t][i] * obs[t][i] for i in range(n)) for t in range(T)]

    # 2) light centered smoothing (we have the whole track, so this is safe)
    smoothed = [0.0] * T
    for t in range(T):
        lo, hi = max(0, t - 1), min(T - 1, t + 1)
        window = lead_est[lo:hi + 1]
        smoothed[t] = sum(window) / len(window)

    # 3) anticipate: shift the known curve forward by the mean reaction
    #    latency so the cue "leads" rather than "follows" it
    L = max(1, round(sum(latency) / n))

    cue = [smoothed[min(t + L, T - 1)] for t in range(T)]

    print(json.dumps(cue))


main()
