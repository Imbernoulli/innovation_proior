# TIER: strong
"""Own-occupancy-peak reserve (the intended insight).

Instead of a fixed reserve (trivial: always full; greedy: always zero), size
ward w's reserve on day t from how much MORE of its own capacity it is
about to need, using a light forward shadow-simulation of ward w's home
occupancy under the public forecast (home admissions only, continuous
discharge at rate 1/mean_los[w] -- ignoring outlier traffic, a first-order
approximation of "what would happen if I only ever served my own
patients"):

     occ_home_est[w][t]  = shadow-simulated home occupancy at day t
     peak[w][t]          = max(occ_home_est[w][t : t+W])   (near-term peak,
                            lookahead window W)
     reserve[w][t]        = clamp(peak[w][t] - occ_home_est[w][t], 0, cap[w])

This says: protect exactly the ADDITIONAL beds ward w's own trajectory
says it will need between now and its near-term peak -- no more. In a
quiet, steady-state ward, occupancy barely moves within the window, so the
peak is close to today's level and the reserve collapses to ~0 (full
lending, same as greedy) -- exactly the low-load regime, where holding
capacity back would only waste it. When a ward's own demand is about to
ramp up faster than turnover can keep pace, occ_home_est ramps ahead of
time in the shadow simulation, the reserve rises BEFORE the ramp hits, and
those beds are not there to be borrowed away by another ward's outlier
patients right when this ward needs them back."""
import sys, json

WINDOW = 8


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    wards = inst["wards"]
    K = len(wards)
    capacity = inst["capacity"]
    mean_los = inst["mean_los"]
    forecast = inst["arrival_forecast"]
    init_occ = inst["init_occ_home"]

    # forward shadow-simulation of home-only occupancy under the forecast, to
    # know occ_home_est[w] at every day t (used for the turnover estimate).
    occ_home_est = [[0.0] * T for _ in range(K)]
    occ = list(init_occ)
    for t in range(T):
        for w in range(K):
            occ_home_est[w][t] = occ[w]
        for w in range(K):
            free_w = max(0.0, capacity[w] - occ[w])
            admit = min(forecast[w][t], free_w)
            occ[w] += admit
            occ[w] -= occ[w] / mean_los[w]
            if occ[w] < 0.0:
                occ[w] = 0.0

    reserve = [[0.0] * T for _ in range(K)]
    for w in range(K):
        cap_w = capacity[w]
        for t in range(T):
            hi = min(T, t + WINDOW)
            occ_now = occ_home_est[w][t]
            peak = max(occ_home_est[w][t:hi])
            need = peak - occ_now
            if need < 0.0:
                need = 0.0
            if need > cap_w:
                need = cap_w
            reserve[w][t] = need

    print(json.dumps({"reserve": reserve}))


if __name__ == "__main__":
    main()
