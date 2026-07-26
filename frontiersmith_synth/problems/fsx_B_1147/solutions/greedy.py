# TIER: greedy
"""The obvious first-write approach: replace, week by week, exactly the
water a field is expected to lose on average -- deficit[f][t] =
max(0, cons[f] - avg_rain[t]) using the CROSS-SCENARIO AVERAGE rainfall,
never any individual scenario. Scale to fit the budget/weekly caps and
reinforce any leftover budget back onto the same weeks the average model
already flagged as needing water. This is a textbook "replace the expected
loss" recipe and is perfectly reasonable in normal weeks.  But when other
scenarios are deliberately wetter in exactly the weeks a drought scenario
runs dry (the planted trap), the cross-scenario average during that window
looks WET, not dry -- deficit[f][t] comes out to 0 for every field in every
one of those weeks -- so this planner allocates nothing at all right when
the drought scenario needs it most, and never reinforces those weeks with
leftover budget either (it has no signal telling it to)."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0

    def nxt():
        nonlocal idx
        v = data[idx]
        idx += 1
        return v

    T = int(nxt()); F = int(nxt()); K = int(nxt())
    total_budget = int(nxt()); weekly_cap = int(nxt())
    fields = []
    for _ in range(F):
        m0 = int(nxt()); cap = int(nxt()); cons = int(nxt())
        thresh = int(nxt()); rate = int(nxt())
        fields.append((m0, cap, cons, thresh, rate))
    rain = []
    for _ in range(K):
        rain.append([int(nxt()) for _ in range(T)])

    avg_rain = []
    for t in range(T):
        avg_rain.append(sum(rain[k][t] for k in range(K)) / K)

    # Step 1: per-week "replace the expected loss" against the AVERAGE
    # weather only -- no cumulative/threshold reasoning, no scenario
    # awareness.
    X = [[0.0] * T for _ in range(F)]
    for f in range(F):
        m0, cap, cons, thresh, rate = fields[f]
        for t in range(T):
            d = cons - avg_rain[t]
            if d > 0:
                X[f][t] = d

    # Step 2: scale down uniformly to respect the total budget.
    total_desired = sum(sum(row) for row in X)
    if total_desired > total_budget and total_desired > 0:
        scale = total_budget / total_desired
        for f in range(F):
            for t in range(T):
                X[f][t] = X[f][t] * scale

    # Step 3: clamp each week to the weekly cap (excess simply dropped --
    # a naive planner does not redistribute it to other weeks).
    for t in range(T):
        wk = sum(X[f][t] for f in range(F))
        if wk > weekly_cap and wk > 0:
            scale = weekly_cap / wk
            for f in range(F):
                X[f][t] = X[f][t] * scale

    Xi = [[int(X[f][t]) for t in range(T)] for f in range(F)]

    # Step 4: reinforce any obviously leftover budget round-robin -- but
    # only onto (field, week) slots the average-weather pass already
    # flagged as needing water. A textbook planner reinforces the weeks it
    # already believes are thirsty; it has no reason to add water to a
    # week its own average-deficit model said was already fine (which is
    # exactly the masked drought window: the cross-scenario average there
    # looks wet).
    used = sum(sum(row) for row in Xi)
    weekly_used = [sum(Xi[f][t] for f in range(F)) for t in range(T)]
    leftover = total_budget - used
    all_slots = [(f, t) for t in range(T) for f in range(F)]
    needy_slots = [(f, t) for (f, t) in all_slots if Xi[f][t] > 0]

    # Sub-step 4a: reinforce the weeks the average-deficit model already
    # flagged, up to a few multiples of their own original allocation --
    # a planner tops up what it already believes is thirsty, but does not
    # pour unlimited water into the same handful of weeks forever.
    NEEDY_ROUNDS = 3
    if leftover > 0 and needy_slots:
        i = 0
        guard = 0
        while leftover > 0 and guard < NEEDY_ROUNDS * len(needy_slots) + 20:
            f, t = needy_slots[i % len(needy_slots)]
            if weekly_used[t] < weekly_cap:
                Xi[f][t] += 1
                weekly_used[t] += 1
                leftover -= 1
            i += 1
            guard += 1
            if i >= NEEDY_ROUNDS * len(needy_slots):
                break

    # Sub-step 4b: only genuinely spare change (needy weeks already
    # reinforced several times over) gets swept across every remaining
    # week as basic due diligence -- a small, bounded leak, not a rescue.
    if leftover > 0 and all_slots:
        i = 0
        guard = 0
        while leftover > 0 and guard < 3 * len(all_slots) + 20:
            f, t = all_slots[i % len(all_slots)]
            if weekly_used[t] < weekly_cap:
                Xi[f][t] += 1
                weekly_used[t] += 1
                leftover -= 1
            i += 1
            guard += 1

    print(F, T)
    for f in range(F):
        print(" ".join(str(x) for x in Xi[f]))


if __name__ == "__main__":
    main()
