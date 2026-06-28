#!/usr/bin/env python3
"""Instance generator for "Soft-Constraint Assignment" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    n m
    cap_0 cap_1 ... cap_{m-1}
    pref_0_0  pref_0_1  ... pref_0_{m-1}
    pref_1_0  pref_1_1  ... pref_1_{m-1}
    ...
    pref_{n-1}_0 ... pref_{n-1}_{m-1}
    C
    t_0 a_0 b_0 w_0
    t_1 a_1 b_1 w_1
    ...
    t_{C-1} a_{C-1} b_{C-1} w_{C-1}

where:
  * n agents (chosen from the seed in [200, 600]); each agent is assigned to exactly
    one of the m slots.
  * m slots (chosen in [8, 40]).
  * cap_j is the capacity of slot j (a non-negative integer); sum(cap) >= n is
    GUARANTEED so that a feasible assignment always exists.
  * pref[i][j] is an integer preference score in [0, 1000] for putting agent i in
    slot j (higher is better).
  * C soft constraints. Each is a 4-tuple `t a b w`:
        t = 0  ("DIFFER"):  agents a and b SHOULD be in DIFFERENT slots;
                            penalty w is charged iff assign[a] == assign[b].
        t = 1  ("SAME"):    agents a and b SHOULD be in the SAME slot;
                            penalty w is charged iff assign[a] != assign[b].
    a != b, w is a positive integer.

Design intent: the objective is  (sum of chosen preferences) - (sum of violated soft
penalties).  We make the soft penalties LOAD-BEARING by setting their scale relative
to the preference spread: a single violated constraint costs on the order of a
"typical" preference value, so the assignment that maximizes raw preference (ignoring
constraints) is NOT optimal, and a constraint-aware local search must trade a little
preference for fewer violations.  Constraints are placed with locality (each agent
participates in a few), so the constraint->agent incidence lists that drive the O(1)
swap delta are non-trivial but sparse -- exactly the regime the innovation targets.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x50FACADE ^ (seed * 2654435761 & 0xFFFFFFFF))

    n = rng.randint(200, 600)
    m = rng.randint(8, 40)

    # --- capacities: sum(cap) >= n guaranteed ---
    # Start each slot with a base capacity, then distribute slack so the total
    # comfortably exceeds n (slack ratio in [1.1, 1.6]). Tight-ish but always feasible.
    total_cap_target = int(n * rng.uniform(1.10, 1.60)) + m
    # random partition of total_cap_target into m non-negative parts
    cuts = sorted(rng.randint(0, total_cap_target) for _ in range(m - 1))
    cap = []
    prev = 0
    for c in cuts:
        cap.append(c - prev)
        prev = c
    cap.append(total_cap_target - prev)
    # guard: ensure each cap >= 1 and sum stays >= n (move slack around if needed)
    for j in range(m):
        if cap[j] < 1:
            cap[j] = 1
    s = sum(cap)
    # if rounding pushed us below n (shouldn't, but be safe), pad the biggest slot
    if s < n + 1:
        cap[max(range(m), key=lambda j: cap[j])] += (n + 1 - s)
    assert sum(cap) >= n

    # --- preferences ---
    # Each agent has a "favourite" slot it scores high in, plus background noise, so
    # the pure-preference optimum is a real assignment (not all-uniform), and tilting
    # an agent away from its favourite to satisfy a constraint costs something.
    pref = []
    pref_scale = 1000
    for i in range(n):
        fav = rng.randrange(m)
        row = []
        for j in range(m):
            base = rng.randint(0, 400)
            if j == fav:
                base += rng.randint(400, 600)   # favourite spikes high
            row.append(min(pref_scale, base))
        pref.append(row)

    # typical preference magnitude -> sets the penalty scale so constraints matter
    flat = [pref[i][j] for i in range(n) for j in range(m)]
    typical = sorted(flat)[len(flat) // 2]  # median preference
    typical = max(50, typical)

    # --- soft constraints ---
    # Number of constraints scales with n; each agent participates in a few. Penalty
    # weights are on the order of a typical preference value (so one violation ~ losing
    # one agent's favourite slot), making the trade-off genuine.
    C = rng.randint(int(0.8 * n), int(2.0 * n))
    cons = []
    seen_pairs = set()
    attempts = 0
    while len(cons) < C and attempts < 40 * C:
        attempts += 1
        a = rng.randrange(n)
        # bias b to be "nearby" in index so incidence has structure, but allow far links
        if rng.random() < 0.6:
            b = (a + rng.randint(1, 8)) % n
        else:
            b = rng.randrange(n)
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        if (lo, hi) in seen_pairs:
            continue
        seen_pairs.add((lo, hi))
        t = 0 if rng.random() < 0.55 else 1   # slight DIFFER majority
        w = rng.randint(int(0.5 * typical), int(2.0 * typical))
        if w < 1:
            w = 1
        cons.append((t, lo, hi, w))
    C = len(cons)

    out = []
    out.append(f"{n} {m}")
    out.append(" ".join(str(c) for c in cap))
    for i in range(n):
        out.append(" ".join(str(v) for v in pref[i]))
    out.append(str(C))
    for (t, a, b, w) in cons:
        out.append(f"{t} {a} {b} {w}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
