"""
roomgeo.py -- shared deterministic construction for the "Echo Room" problem
(fsx_B_1189). Imported by BOTH gen.py (prints the visible instance) and
verify.py (recomputes ground truth, including the two held-out microphones
that are never written to stdout). Nothing here reads argv or does I/O.

Model (image-source method, first-order only):
  A room has W flat walls. Each wall k is an infinite line at perpendicular
  distance d_k > 0 from the fixed source S, in direction (unit normal) n_k.
  Reflecting S across wall k's line gives the "image source"
      I_k = S + 2*d_k*n_k.
  For ANY receiver point R, the length of the source->wall_k->R path
  (a first-order / single-bounce echo) equals |I_k - R| exactly. This is
  the image-source-forward mechanism.

  A microphone observes only the UNLABELED multiset of the W first-order
  echo travel times (echo-labeling-combinatorics: which number came from
  which wall is not given). A handful of instances also splice in decoy
  numbers that are close to a real echo time but are not the distance from
  any consistent point to that microphone (first-order-reflection-prior:
  the solver must trust only readings that are globally consistent with a
  single fixed point across every microphone, not every number it hears).
"""
import math
import random

# ladder: W = wall count, K = given (visible) microphones, trap = engineered
# difficulty. 'swap'  -> one pair of walls has its NEAREST-mic rank flipped
# between mic0 and mic1 (defeats "same sorted order at every mic").
# 'swap2' -> two independent such flipped pairs. 'decoyN' -> N decoy echo
# readings spliced into mic0 and mic1's lists.
LADDER = {
    1: dict(W=4, K=4, trap=()),
    2: dict(W=4, K=4, trap=()),
    3: dict(W=5, K=5, trap=("swap",)),
    4: dict(W=5, K=4, trap=()),
    5: dict(W=5, K=5, trap=("swap",)),
    6: dict(W=6, K=5, trap=("swap", "decoy1")),
    7: dict(W=6, K=4, trap=("swap2",)),
    8: dict(W=6, K=5, trap=("swap", "decoy2")),
    9: dict(W=7, K=5, trap=("swap2", "decoy2")),
    10: dict(W=7, K=4, trap=("swap2", "decoy2")),
}


def _unit(theta):
    return (math.cos(theta), math.sin(theta))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _scale(a, s):
    return (a[0] * s, a[1] * s)


def build(test_id):
    """Deterministic full ground truth for a given testId (1..10, clamped).
    Returns a dict with S, W, K, walls=[(nx,ny,d)]*W, image_pts=[(x,y)]*W,
    given_mics=[(x,y)]*K, held_mics=[(x,y)]*2, obs=[[floats]]*K (the
    shuffled, possibly decoy-spliced, per-mic reading lists)."""
    tid = min(max(int(test_id), 1), 10)
    cfg = LADDER[tid]
    W, K, trap = cfg["W"], cfg["K"], cfg["trap"]
    rng = random.Random(1_000_003 * tid + 20260726)

    S = (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6))

    # ---- base walls: evenly spread angles + jitter, random distances ----
    walls = []
    for k in range(W):
        theta = (2 * math.pi * k / W) + rng.uniform(-0.35, 0.35)
        d = rng.uniform(2.0, 4.2)
        nx, ny = _unit(theta)
        walls.append((nx, ny, d))

    # ---- planted rank-swap trap(s): force wall p to be nearer than wall q
    # at mic0 but FARTHER than wall q at mic1 (defeats a same-order-at-every
    # -mic assumption). We overwrite two wall directions/distances outright
    # so the property holds by construction, not by chance. ----
    swap_pairs = []
    if "swap" in trap or "swap2" in trap:
        swap_pairs.append((0, 1))
    if "swap2" in trap:
        swap_pairs.append((2, 3 if W > 3 else 2))

    def image_of(nx, ny, d):
        return _add(S, _scale((nx, ny), 2 * d))

    # placeholders for mic0/mic1 -- filled in below once walls are final
    forced_mic0 = None
    forced_mic1 = None
    for (p, q) in swap_pairs:
        if p >= W or q >= W or p == q:
            continue
        thp = rng.uniform(0, 2 * math.pi)
        thq = thp + rng.uniform(2.4, 3.6)  # well-separated direction
        dp = rng.uniform(2.4, 3.6)
        dq = rng.uniform(2.4, 3.6)
        walls[p] = (_unit(thp)[0], _unit(thp)[1], dp)
        walls[q] = (_unit(thq)[0], _unit(thq)[1], dq)
        Ip = image_of(*walls[p])
        Iq = image_of(*walls[q])
        if forced_mic0 is None:
            # mic0: close to Ip's side -> wall p ranks nearer than wall q
            t = rng.uniform(0.35, 0.55)
            perp = (-(Ip[1] - S[1]), Ip[0] - S[0])
            perp_len = math.hypot(*perp) or 1.0
            jitter = _scale(perp, rng.uniform(-0.15, 0.15) / perp_len)
            forced_mic0 = _add(_add(S, _scale(_sub(Ip, S), t)), jitter)
            # mic1: close to Iq's side -> wall q ranks nearer than wall p
            t2 = rng.uniform(0.35, 0.55)
            perp2 = (-(Iq[1] - S[1]), Iq[0] - S[0])
            perp2_len = math.hypot(*perp2) or 1.0
            jitter2 = _scale(perp2, rng.uniform(-0.15, 0.15) / perp2_len)
            forced_mic1 = _add(_add(S, _scale(_sub(Iq, S), t2)), jitter2)

    image_pts = [image_of(nx, ny, d) for (nx, ny, d) in walls]

    # ---- given microphones ----
    given_mics = []
    for i in range(K):
        if i == 0 and forced_mic0 is not None:
            given_mics.append(forced_mic0)
        elif i == 1 and forced_mic1 is not None:
            given_mics.append(forced_mic1)
        else:
            ang = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.6, 1.8)
            given_mics.append(_add(S, _scale(_unit(ang), r)))

    # ---- held-out microphones (never printed by gen.py) ----
    held_mics = []
    for _ in range(2):
        ang = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.7, 1.9)
        held_mics.append(_add(S, _scale(_unit(ang), r)))

    # sanity-verify the swap property actually holds (defensive; should
    # always hold by construction above)
    if swap_pairs and forced_mic0 is not None:
        p, q = swap_pairs[0]
        Ip, Iq = image_pts[p], image_pts[q]
        m0, m1 = given_mics[0], given_mics[1]
        assert _dist(Ip, m0) < _dist(Iq, m0), "swap trap failed at mic0"
        assert _dist(Ip, m1) > _dist(Iq, m1), "swap trap failed at mic1"

    # ---- observed per-mic reading lists (shuffled, decoys spliced) ----
    n_decoy = 0
    if "decoy1" in trap:
        n_decoy = 1
    if "decoy2" in trap:
        n_decoy = 2

    obs = []
    for i in range(K):
        m = given_mics[i]
        readings = [_dist(pt, m) for pt in image_pts]
        if n_decoy and i in (0, 1):
            base = list(readings)
            for j in range(n_decoy):
                src = base[(i + j) % len(base)]
                delta = 0.14 + 0.05 * j
                sign = 1 if (j % 2 == 0) else -1
                readings.append(max(0.05, src + sign * delta))
        rng.shuffle(readings)
        obs.append(readings)

    return dict(S=S, W=W, K=K, walls=walls, image_pts=image_pts,
                given_mics=given_mics, held_mics=held_mics, obs=obs)
