"""
common.py -- shared PRIVATE world-generation + physics for the pollution-attribution-wind
problem. Imported by gen.py (to synthesize the printed TRAIN log) and verify.py (to
regenerate the hidden true emission rates + the held-out day set). NEVER importable by
a submitted solution: the harness sandboxes solutions with the whole synth tree (incl.
this problem directory) hidden behind a tmpfs, so a solution process cannot read this
file -- it must re-derive the (fully STATED) kernel/background formulas from scratch,
using only the per-instance constants printed in its stdin.

Bearing convention: for a source at (sx, sy) and the receptor fixed at the origin,
`bearing_deg` returns the compass bearing FROM the source TOWARD the receptor (the
direction the wind must blow FOR the source's plume to reach the receptor). `theta`
(the receptor->source placement angle used only inside world-building) is exactly
180 degrees away from that bearing; the two are kept distinct to make the placement
code below read naturally.
"""
import math, random

# ---------------- fixed physical constants (same value for every test id; still
# echoed into the printed instance so a solution never has to hardcode them) ----------
SIGMA_MAX_DEG = 48.0   # angular plume spread at (near-)zero wind speed -- broad/diffuse
ALPHA = 1.15           # how fast the spread narrows as wind speed grows
L0 = 42.0              # distance decay length scale
BETA = 0.9             # dilution-with-speed rate constant

SECRET_BASE = 4_211_000     # world (sources / true rates / background law) RNG offset
HOLDOUT_BASE = 8_622_000    # held-out day-set RNG offset
UPPER_BOUND = 1.0e6         # feasibility cap on a submitted emission rate

# per test_id: (K, cluster sizes, trap flag). D_train is DERIVED (see build_world)
# from K/clusters/trap so every source always gets at least one clean "anchor" day.
CONFIGS = {
    1:  dict(K=6,  clusters=[2],        trap=0),
    2:  dict(K=6,  clusters=[2],        trap=0),
    3:  dict(K=7,  clusters=[3],        trap=1),
    4:  dict(K=7,  clusters=[2, 2],     trap=0),
    5:  dict(K=8,  clusters=[2, 2],     trap=1),
    6:  dict(K=8,  clusters=[3],        trap=0),
    7:  dict(K=9,  clusters=[3, 2],     trap=1),
    8:  dict(K=9,  clusters=[2, 2],     trap=0),
    9:  dict(K=9,  clusters=[3, 2, 2],  trap=1),
    10: dict(K=10, clusters=[3, 3],     trap=1),
}


def wind_sigma_deg(speed):
    return SIGMA_MAX_DEG / (1.0 + ALPHA * speed)


def dilution(speed):
    return 1.0 / (1.0 + BETA * speed)


def bearing_deg(sx, sy):
    return math.degrees(math.atan2(-sy, -sx)) % 360.0


def ang_diff_deg(a, b):
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def kernel(sx, sy, wind_dir_deg, wind_speed):
    r = math.hypot(sx, sy)
    brg = bearing_deg(sx, sy)
    delta = ang_diff_deg(wind_dir_deg, brg)
    sigma = wind_sigma_deg(wind_speed)
    ang_k = math.exp(-(delta * delta) / (2.0 * sigma * sigma))
    dist_k = math.exp(-r / L0)
    return ang_k * dist_k


def background(day_id, A0, A1, P):
    return A0 + A1 * math.sin(2.0 * math.pi * day_id / P)


def build_world(test_id):
    """Deterministic hidden world for this test id: candidate source coordinates,
    the SECRET true emission rate per source, the per-instance background-trend
    constants, and bearing info needed to bias the TRAIN day sample (gen.py only).
    Bit-identical whenever called with the same test_id (only depends on test_id)."""
    cfg = CONFIGS[test_id]
    K = cfg["K"]
    clusters = list(cfg["clusters"])
    rng = random.Random(SECRET_BASE + 97 * test_id)

    standalone_n = K - sum(clusters)
    group_sizes = clusters + [1] * standalone_n
    order = list(range(len(group_sizes)))
    rng.shuffle(order)
    group_sizes = [group_sizes[i] for i in order]
    is_cluster = [gs > 1 for gs in group_sizes]

    n_groups = len(group_sizes)
    base_angles = [(360.0 * i) / n_groups for i in range(n_groups)]
    rng.shuffle(base_angles)

    sources = []             # (sx, sy) in placement order == source index 0..K-1
    all_bearings = []        # per-source kernel bearing, index-aligned with `sources`
    cluster_member_idx = []  # list of lists of source indices, one list per cluster group
    trap_group_bearing = None
    trap_group_size = -1
    for gi, gsize in enumerate(group_sizes):
        theta_center = (base_angles[gi] + rng.uniform(-8.0, 8.0)) % 360.0
        spacing = rng.uniform(17.0, 24.0)  # degrees between adjacent cluster members
        members_brg = []
        member_idx = []
        for m in range(gsize):
            if gsize == 1:
                theta = theta_center
            else:
                theta = (theta_center + (m - (gsize - 1) / 2.0) * spacing) % 360.0
            r = rng.uniform(15.0, 55.0)
            sx = r * math.cos(math.radians(theta))
            sy = r * math.sin(math.radians(theta))
            member_idx.append(len(sources))
            sources.append((round(sx, 3), round(sy, 3)))
            brg = bearing_deg(sx, sy)
            all_bearings.append(brg)
            members_brg.append(brg)
        if gsize > 1:
            cluster_member_idx.append(member_idx)
            if gsize > trap_group_size:
                trap_group_size = gsize
                trap_group_bearing = sum(members_brg) / len(members_brg)

    E_true = [round(rng.uniform(4.0, 22.0), 4) for _ in range(K)]

    A0 = round(rng.uniform(3.0, 8.0), 3)
    A1 = round(rng.uniform(2.0, 6.0), 3)
    P = rng.choice([6, 7, 8, 9])

    return dict(K=K, trap=cfg["trap"], sources=sources, all_bearings=all_bearings,
                cluster_member_idx=cluster_member_idx, E_true=E_true,
                A0=A0, A1=A1, P=P, trap_group_bearing=trap_group_bearing)


def sample_day(rng, kind, world, target_bearing=None):
    """One (wind_dir, wind_speed) draw for the TRAIN log.
    kind='prevail'      -> low speed (broad, diluted-little/high-magnitude), aimed at the
                            instance's dominant (trap) cluster bearing.
    kind='anchor'       -> high speed (sharp), aimed squarely at `target_bearing`.
    kind='info_broad'   -> moderate speed, direction uniformly random (general diversity).
    """
    if kind == "prevail":
        base = world["trap_group_bearing"] if world["trap_group_bearing"] is not None \
            else rng.uniform(0.0, 360.0)
        wind_dir = (base + rng.uniform(-3.0, 3.0)) % 360.0
        wind_speed = rng.uniform(0.10, 0.35)
    elif kind == "anchor":
        wind_dir = (target_bearing + rng.uniform(-1.5, 1.5)) % 360.0
        wind_speed = rng.uniform(3.0, 4.0)
    else:  # 'info_broad'
        wind_dir = rng.uniform(0.0, 360.0)
        wind_speed = rng.uniform(1.0, 3.0)
    return wind_dir, wind_speed


def concentration(world, day_id, wind_dir, wind_speed, noise_rng, sigma_noise=0.5):
    total = 0.0
    for (sx, sy), e in zip(world["sources"], world["E_true"]):
        total += kernel(sx, sy, wind_dir, wind_speed) * e
    y = dilution(wind_speed) * total + background(day_id, world["A0"], world["A1"], world["P"])
    y += noise_rng.gauss(0.0, sigma_noise)
    return y


def holdout_days(test_id, world, n_holdout=60):
    """A FAIR, direction-diverse held-out day set (never trap-biased) -- the real
    test of whether the submitted rates generalize across the wind rose, regenerated
    only from test_id (never printed anywhere)."""
    rng = random.Random(HOLDOUT_BASE + 173 * test_id)
    noise_rng = random.Random(HOLDOUT_BASE + 911 * test_id)
    days = []
    for h in range(n_holdout):
        day_id = 10_000 + h  # disjoint id-space from any train day id
        wind_dir = rng.uniform(0.0, 360.0)
        wind_speed = rng.uniform(0.4, 3.6)
        y = concentration(world, day_id, wind_dir, wind_speed, noise_rng)
        days.append((day_id, wind_dir, wind_speed, y))
    return days
