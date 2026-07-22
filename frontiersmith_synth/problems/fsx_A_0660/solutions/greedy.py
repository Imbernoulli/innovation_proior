# TIER: greedy
"""The obvious first approach: a per-dam reactive threshold controller. Each
dam looks ONLY at its own current storage fraction and opens its gate more as
it fills up -- exactly the textbook "release when nearly full" rule. It never
looks at the forecast ahead, never anticipates the routed pulse from
upstream, and never coordinates with the other two dams. Because the
candidate must submit its whole release plan at once (no real per-tick
feedback from the evaluator), it self-simulates forward using the SAME
physics the evaluator uses, but the *decision rule* at each tick still only
consults that dam's own storage level at that tick -- it is blind to what is
coming."""
import sys, json

def per_dam_release(cap, rmax, storage_frac):
    if storage_frac > 0.80:
        return rmax
    elif storage_frac > 0.55:
        return 0.35 * rmax
    return 0.0

def main():
    inst = json.load(sys.stdin)
    T = inst["t_steps"]
    d1, d2, d3 = inst["dam1"], inst["dam2"], inst["dam3"]
    delay12, delay23 = inst["delay12"], inst["delay23"]

    s1, s2, s3 = d1["storage0"], d2["storage0"], d3["storage0"]
    cap1, cap2, cap3 = d1["capacity"], d2["capacity"], d3["capacity"]
    rmax1, rmax2, rmax3 = d1["release_max"], d2["release_max"], d3["release_max"]

    rel1 = [0.0] * T
    rel2 = [0.0] * T
    rel3 = [0.0] * T
    act1 = [0.0] * T
    act2 = [0.0] * T
    act3 = [0.0] * T

    for t in range(T):
        routed2 = act1[t - delay12] if t - delay12 >= 0 else 0.0
        avail1 = s1 + d1["inflow"][t]
        req1 = per_dam_release(cap1, rmax1, s1 / cap1)
        a1 = min(req1, avail1)
        left1 = avail1 - a1
        if left1 > cap1:
            a1 += left1 - cap1; s1 = cap1
        else:
            s1 = left1
        rel1[t] = req1; act1[t] = a1

        avail2 = s2 + d2["inflow"][t] + routed2
        req2 = per_dam_release(cap2, rmax2, s2 / cap2)
        a2 = min(req2, avail2)
        left2 = avail2 - a2
        if left2 > cap2:
            a2 += left2 - cap2; s2 = cap2
        else:
            s2 = left2
        rel2[t] = req2; act2[t] = a2

        routed3 = act2[t - delay23] if t - delay23 >= 0 else 0.0
        avail3 = s3 + d3["inflow"][t] + routed3
        req3 = per_dam_release(cap3, rmax3, s3 / cap3)
        a3 = min(req3, avail3)
        left3 = avail3 - a3
        if left3 > cap3:
            a3 += left3 - cap3; s3 = cap3
        else:
            s3 = left3
        rel3[t] = req3; act3[t] = a3

    print(json.dumps({"release1": rel1, "release2": rel2, "release3": rel3}))

if __name__ == "__main__":
    main()
