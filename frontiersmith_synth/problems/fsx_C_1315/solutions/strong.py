# TIER: strong
# The insight: restrict the search to TIME-CONSISTENT policies (one exit per
# zone, held for the whole horizon) -- a directive that never contradicts
# itself never pays the credibility tax, so under a constant policy every
# zone's compliance sits at its full nominal rate for the entire simulation.
# That collapses the search space to "one exit choice per zone" and lets us
# directly hill-climb the REAL objective (using the exact, honestly-documented
# simulator formula from the statement -- everything it needs is public),
# correctly weighting only the compliant fraction as steerable and the rest
# as a sunk default flow. This beats both (a) doing nothing (which ignores
# that some zones' shared default exit is oversubscribed) and (b) reactive
# full-compliance rerouting (which mis-sizes the controllable flow AND pays
# the credibility tax every time its picture of "least loaded" drifts).
import sys, json


def simulate(inst, choice):
    """Evacuated total under a TIME-INVARIANT guidance choice (one exit per
    zone). Because the choice never changes, credibility never decays, so
    every zone's effective compliance is simply its nominal base_compliance."""
    Z, E, T = inst["n_zones"], inst["n_exits"], inst["T"]
    pop = inst["population"]; cap = inst["capacity"]; egress = inst["egress_cap"]
    base_c = inst["base_compliance"]; default_exit = inst["default_exit"]
    beta = inst["congestion_beta"]

    remaining = list(pop)
    pending = [[0.0] * E for _ in range(Z)]
    evac = 0.0
    for _t in range(T):
        for i in range(Z):
            depart = remaining[i] if remaining[i] < egress[i] else egress[i]
            remaining[i] -= depart
            guided = depart * base_c[i]
            default_amt = depart - guided
            pending[i][choice[i]] += guided
            pending[i][default_exit[i]] += default_amt
        for e in range(E):
            tot = sum(pending[i][e] for i in range(Z))
            if tot <= 1e-12:
                continue
            ratio = tot / cap[e]
            mult = 1.0 if ratio <= 1.0 else 1.0 / (1.0 + beta[e] * (ratio - 1.0))
            cap_eff = cap[e] * mult
            served = tot if tot < cap_eff else cap_eff
            frac = served / tot
            for i in range(Z):
                s = pending[i][e] * frac
                pending[i][e] -= s
                evac += s
    return evac


def main():
    inst = json.load(sys.stdin)
    Z, E, T = inst["n_zones"], inst["n_exits"], inst["T"]
    reach = inst["reachable"]; default_exit = inst["default_exit"]

    choice = list(default_exit)              # seed: nobody redirected
    best_val = simulate(inst, choice)
    for _sweep in range(6):
        improved = False
        for i in range(Z):
            cur = choice[i]
            best_e, best_e_val = cur, best_val
            for e in range(E):
                if not reach[i][e] or e == cur:
                    continue
                choice[i] = e
                v = simulate(inst, choice)
                if v > best_e_val + 1e-9:
                    best_e_val, best_e = v, e
                choice[i] = cur
            if best_e != cur:
                choice[i] = best_e
                best_val = best_e_val
                improved = True
        if not improved:
            break

    guidance = [[choice[i]] * T for i in range(Z)]
    print(json.dumps({"guidance": guidance}))


main()
