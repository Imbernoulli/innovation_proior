#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE market-garden instance (JSON) to stdout.
Deterministic: all randomness seeded from testId only.
"""
import sys, json, random, copy

T = 8          # seasons (fixed: "eight seasons" theme)
K = 3          # nutrient dimensions: 0=N, 1=P, 2=Kx (potassium)
CAP = [100.0, 100.0, 100.0]
INIT_SOIL = [40.0, 50.0, 50.0]   # identical starting vector for every plot

PEST_GROW = 0.5
PEST_DECAY = 0.5
PEST_CAP = 1.0
PEST_COEFF = 1.0   # pest multiplier = 1 - PEST_COEFF*pest_p

# core template crops (the complementary A/B pair + the greedy "cash" bait)
BASE_CROPS = [
    dict(name="Legume",     family=0, req=[10.0, 30.0, 10.0], depletion=[0.0, 18.0, 0.0],
         replenish=[40.0, 0.0, 0.0], base_yield=70.0, price=9.0),
    dict(name="GrainFeeder", family=1, req=[40.0, 15.0, 10.0], depletion=[40.0, 2.0, 0.0],
         replenish=[0.0, 20.0, 0.0], base_yield=90.0, price=11.0),
    dict(name="CashCrop",   family=2, req=[30.0, 10.0, 50.0], depletion=[10.0, 2.0, 45.0],
         replenish=[0.0, 0.0, 0.0], base_yield=130.0, price=16.0),
]

DECOY_TEMPLATES = [
    dict(name="Millet",  family=3, req=[20.0, 20.0, 20.0], depletion=[8.0, 8.0, 8.0],
         replenish=[0.0, 0.0, 0.0], base_yield=55.0, price=6.5),
    dict(name="Squash",  family=4, req=[15.0, 25.0, 15.0], depletion=[6.0, 10.0, 4.0],
         replenish=[2.0, 0.0, 0.0], base_yield=48.0, price=7.0),
    dict(name="Buckwheat", family=5, req=[12.0, 18.0, 12.0], depletion=[4.0, 6.0, 3.0],
         replenish=[0.0, 3.0, 0.0], base_yield=42.0, price=6.0),
]

# (P, n_decoys, jitter_pct) per testId 1..10 -- last two are larger held-out cases
CASE_PARAMS = {
    1:  (4,  0, 0.00),
    2:  (5,  1, 0.03),
    3:  (6,  1, 0.04),
    4:  (7,  2, 0.05),
    5:  (6,  0, 0.05),
    6:  (8,  2, 0.06),
    7:  (9,  1, 0.06),
    8:  (10, 2, 0.07),
    9:  (12, 3, 0.05),   # larger, held-out
    10: (16, 3, 0.05),   # largest, held-out
}


def yield_norm(soil, req):
    r = 1.0
    for k in range(K):
        if req[k] > 1e-9:
            r = min(r, soil[k] / req[k])
    return max(0.0, min(1.0, r))


def simulate_pair_cycle(crop_a, crop_b, init):
    """Verify A,B,A,B,...  (T seasons) never hits a nutrient bottleneck (y_norm==1
    every season) from `init`, i.e. it really is a sustained productive cycle."""
    soil = list(init)
    for t in range(T):
        c = crop_a if t % 2 == 0 else crop_b
        yn = yield_norm(soil, c["req"])
        if yn < 1.0 - 1e-9:
            return False
        for k in range(K):
            soil[k] = max(0.0, min(CAP[k], soil[k] - c["depletion"][k] * yn + c["replenish"][k]))
    return True


def value_at(crop, soil):
    return crop["price"] * crop["base_yield"] * yield_norm(soil, crop["req"])


def build_crops(rng, n_decoys, jitter):
    crops = []
    for tpl in BASE_CROPS:
        c = copy.deepcopy(tpl)
        for key in ("req", "depletion", "replenish"):
            c[key] = [max(0.0, v * (1.0 + rng.uniform(-jitter, jitter))) for v in c[key]]
        c["base_yield"] *= (1.0 + rng.uniform(-jitter, jitter))
        c["price"] *= (1.0 + rng.uniform(-jitter, jitter))
        crops.append(c)
    for i in range(n_decoys):
        tpl = DECOY_TEMPLATES[i % len(DECOY_TEMPLATES)]
        c = copy.deepcopy(tpl)
        c["family"] = 3 + i
        for key in ("req", "depletion", "replenish"):
            c[key] = [max(0.0, v * (1.0 + rng.uniform(-jitter, jitter))) for v in c[key]]
        c["base_yield"] *= (1.0 + rng.uniform(-jitter, jitter))
        c["price"] *= (1.0 + rng.uniform(-jitter, jitter))
        crops.append(c)
    return crops


def make_instance(test_id):
    P, n_decoys, jitter = CASE_PARAMS[test_id]
    attempt = 0
    while True:
        rng = random.Random(test_id * 100003 + attempt * 97 + 0x5eed)
        crops = build_crops(rng, n_decoys, jitter)
        A, B, C = crops[0], crops[1], crops[2]
        # (1) greedy bait: at the initial state, CashCrop strictly beats GrainFeeder
        #     strictly beats Legume, so a myopic 1-step planner always bites on CashCrop.
        vA, vB, vC = value_at(A, INIT_SOIL), value_at(B, INIT_SOIL), value_at(C, INIT_SOIL)
        ok = (vC > vB * 1.15) and (vB > vA * 1.15)
        # decoys must never look better than the A/B baseline value at init
        for d in crops[3:]:
            if value_at(d, INIT_SOIL) > min(vA, vB) * 0.9:
                ok = False
        # (2) A,B alternation from INIT_SOIL is a genuinely sustained cycle (never
        #     nutrient-bottlenecked across all T seasons)
        if ok and not simulate_pair_cycle(A, B, INIT_SOIL):
            ok = False
        if ok:
            break
        attempt += 1
        if attempt > 200:
            raise RuntimeError("could not build a valid instance")

    threshold = max(1, -(-P // 2))  # ceil(P/2)
    inst = {
        "name": f"garden_{test_id}",
        "P": P, "T": T, "K": K,
        "cap": CAP,
        "crops": crops,
        "pest_grow": PEST_GROW, "pest_decay": PEST_DECAY,
        "pest_cap": PEST_CAP, "pest_coeff": PEST_COEFF,
        "glut_threshold": threshold,
        "init_soil": [list(INIT_SOIL) for _ in range(P)],
    }
    return inst


def main():
    test_id = int(sys.argv[1])
    inst = make_instance(test_id)
    print(json.dumps(inst))


if __name__ == "__main__":
    main()
