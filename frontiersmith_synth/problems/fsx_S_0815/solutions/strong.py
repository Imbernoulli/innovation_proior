# TIER: strong
# Two-way looking search: the statement gives the exact ring dynamics, so this
# solution replicates them internally (a faithful copy of the evaluator's
# simulate()) and grid-searches a linear holding law
#   u_i(k) = clip(gain_back*(Ht-H_i(k)) + gain_fwd*(Ht-H_{i+1}(k)), 0, cap_frac*max_hold)
# over BOTH gain_back and gain_fwd (not just gain_back), per instance. Reacting
# also to the headway of the bus immediately behind you (gain_fwd) is what lets
# a policy damp the ring's slowest mode -- the alternating, every-other-bus
# oscillation -- instead of only chasing your own personal schedule (which
# structurally cannot fix that mode, and easily overcorrects it). The search
# also tunes cap_frac and target_frac, so on high-capacity-weight instances it
# can trade a little correction speed for a lot less total hold spent.
import sys, json

inst = json.load(sys.stdin)
n = inst["n_buses"]
beta = inst["beta"]
k_rounds = inst["rounds"]
hnom = inst["nominal_headway"]
max_hold = inst["max_hold"]
gamma = inst["capacity_weight"]
H0 = inst["initial_headways"]


def simulate(gb, gf, tf, cf):
    ht = tf * hnom
    cap = cf * max_hold
    H = list(H0)
    wait_acc = 0.0
    hold_acc = 0.0
    for _k in range(k_rounds):
        u = [0.0] * n
        for i in range(n):
            j = (i + 1) % n
            raw = gb * (ht - H[i]) + gf * (ht - H[j])
            if raw < 0.0:
                raw = 0.0
            elif raw > cap:
                raw = cap
            u[i] = raw
        for i in range(n):
            d = H[i] - hnom
            wait_acc += d * d
            hold_acc += u[i]
        newH = [0.0] * n
        for i in range(n):
            p = (i - 1) % n
            v = (1 + beta) * H[i] - beta * H[p] + u[i] - u[p]
            if v != v or v > 1e9:
                v = 1e9
            elif v < -1e9:
                v = -1e9
            newH[i] = v
        H = newH
    denom = float(k_rounds * n)
    return (wait_acc / denom) + gamma * (hold_acc / denom)


best = None
best_params = (0.0, 0.0, 1.0, 1.0)
gb_grid = [-0.4 + 0.2 * i for i in range(11)]      # -0.4 .. 1.6
gf_grid = [-0.8 + 0.2 * i for i in range(11)]      # -0.8 .. 1.2
tf_grid = [0.85, 1.0, 1.15]
cf_grid = [0.4, 0.7, 1.0]

for tf in tf_grid:
    for cf in cf_grid:
        for gb in gb_grid:
            for gf in gf_grid:
                obj = simulate(gb, gf, tf, cf)
                if best is None or obj < best:
                    best = obj
                    best_params = (gb, gf, tf, cf)

gb, gf, tf, cf = best_params
print(json.dumps({"gain_back": gb, "gain_fwd": gf, "target_frac": tf, "cap_frac": cf}))
