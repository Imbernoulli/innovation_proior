I wanted an animation of a flock of birds — dozens of them wheeling and turning together, never colliding, splitting around a tree and re-forming on the far side — without drawing a single bird's path by hand. Hand-scripting fails the moment the population grows: it is tedious past the first few birds, error-prone given how intricate real flight paths are, incapable of guaranteeing that no two birds touch at every frame, and nearly impossible to edit globally, since a director's "have them turn left two seconds in" would mean redrawing everything. So scripting is out; the motion has to *generate itself*. The structural paradox of a flock is the trap: it is made of discrete birds yet flows like a fluid, looks random yet is tightly synchronized, and feels as though something is steering the whole thing from above — but there is no conductor. Every shred of evidence says a flock is nothing but the aggregate of individual birds, each acting on its own local view. The existing primitives don't cover this. Reeves' particle systems give me a big population of individual dot-particles with their own position, velocity, color, and life cycle, but a particle is orientationless and — more fatally — particles as used *do not interact*, each evolving on its own internal state, whereas a bird's whole behavior depends on its neighbors. The force-field flocks (Amkraut, Girard, Karl) drive birds along a repulsion field's phase portrait, but a head-on approach to an obstacle field gets only decelerated, never turned aside — the worst possible response — and the field is too strong near, too weak far, and blind to the difference between a wall you fly toward and one you fly alongside. And the obvious "stay together" model, a central force pulling every bird to the flock's global centroid, makes a scattered flock collapse onto one point and forbids it from ever splitting around an obstacle. Supplying a simulated bird with perfect global information is exactly what makes the behavior visibly fail.

So I propose Boids. Rather than author the flock, I author one bird and let the flock emerge: each bird is an autonomous actor — an oriented point-mass flyer carrying momentum, a maximum speed, and a maximum acceleration from an underlying geometric-flight substrate — and it steers using only its *local* perception of nearby flockmates. The complex global motion is then nothing but the aggregate of three simple local steering rules, applied in order of decreasing precedence: separation (collision avoidance), steer away from neighbors that are too close, using position only; alignment (velocity matching), steer to match the average velocity of nearby neighbors, using velocity only; and cohesion (flock centering), steer toward the centroid of nearby neighbors, a longer-range position rule. What makes the whole thing work is that locality is the *mechanism*, not an optimization. The central-force pathology inverts into a virtue once "the center of the flock" is read as the centroid of the *nearby* birds: deep inside the flock a bird's neighbors surround it roughly symmetrically, so the local centroid sits almost on top of it and the centering urge is nearly zero; a bird on the edge has all its neighbors to one side, so the local centroid pulls it inward and centering is strong exactly at the boundary, gently herding stragglers back. And splitting comes for free, because a bird only cares about staying near *its* neighbors — if the rest of the flock peels away around a tree, it simply follows the locals. This also matches what zoology describes, a bird attending only to itself, its two or three nearest neighbors, and the rest of the flock as an undifferentiated mass — which is why real flocks have no size ceiling, since no bird attends to every other bird, and why my per-bird work must be bounded by neighborhood density, not flock size.

The reason there are three rules and not two is subtle and load-bearing. The zoological seed is a tension between two opposing urges, stay close and don't collide, but position-based avoidance alone is insufficient: a purely static rule, reading only where neighbors are *now*, can *establish* a minimum separation but cannot *maintain* it through flight — two boids cruising side by side at slightly different velocities drift apart or together while the rule reacts only once they are already too close, perpetually a step behind. The clean fix is velocity matching: if a boid matches its neighbors' velocity, then by construction the distances to those neighbors barely change over time. So velocity matching is predictive collision avoidance, complementary to static separation rather than redundant with it — static separation sets the minimum required spacing, velocity matching maintains it — and that is the third rule.

Now the perception and the form of each rule. A boid perceives another only if it is within a sensitivity radius *and* within a field-of-view cone about the boid's own heading — a neighbor directly behind is unseen, because a real bird has roughly a $300^\circ$ field of view with a blind cone behind the head, and a boid tugged by birds behind it would keep glancing backward at the mass it is leading. Within that zone, influence must fall off faster than linearly: a spring-like linear weighting produces a bouncy, oscillating flock, while an inverse-square falloff is well-damped and reads as natural — and it is not arbitrary, since Partridge's measurements of schooling fish found each neighbor's contribution decaying as roughly $1/r^2$ to $1/r^3$, consistent with a silhouette's area shrinking as $1/r^2$ and a pressure wave's intensity as $1/r^3$. To express a single steering suggestion cleanly I use a velocity-error form. A behavior wants the boid moving a certain way — a desired velocity — but the boid already has a current velocity, so the honest request is the difference,
$$\text{steering} = \mathbf{v}_{\text{desired}} - \mathbf{v}_{\text{current}},$$
which automatically points toward the wanted motion and shrinks to zero as the boid achieves it, so the correction fades out on its own with no separate overshoot handling. Cohesion is a seek toward the local centroid $\mathbf{c} = \frac{1}{|N|}\sum_{j\in N}\mathbf{p}_j$: the desired velocity is $\mathbf{v}_{\text{desired}} = \widehat{(\mathbf{c} - \mathbf{p})}\,v_{\max}$, "go full tilt toward the centroid," and the steering is $\mathbf{v}_{\text{desired}} - \mathbf{v}$. Alignment is even more direct, because the desired thing is itself a velocity: $\mathbf{v}_{\text{desired}} = \frac{1}{|N|}\sum_{j\in N}\mathbf{v}_j$, and the steering is that average minus my own velocity. Separation I take as a sum of per-neighbor repulsions,
$$\mathbf{f}_{\text{sep}} = \sum_{j\in N} \frac{-(\mathbf{p}_j - \mathbf{p})}{r_j^{2}},$$
the offset from neighbor to me scaled by $1/r^2$ — equivalently the unit away-vector scaled by $1/r$, so the nearest intruder dominates. I sum the individual repulsions rather than steering away from the neighbors' averaged position on purpose: a symmetric crowd would average to a centroid right on top of me and produce *no* push, the worst case for crowding, whereas summing keeps every crowding neighbor's shove. And I take this summed repulsion *directly* as the steering force without renormalizing it to $v_{\max}$, because renormalizing would flatten the near intruder's shove down to the same length as a far one's and erase exactly the distance falloff I just built in.

The combination of the three is where the naive move fails. Averaging the steering vectors — even weighted — has a catastrophe at precisely the moments that matter: two opposing urgent urges (avoidance saying "hard left, wall ahead," cohesion saying "right, toward your friends") average to a tiny turn straight into the wall, and even non-opposing advice mushes ("fly north" and "fly east" average to "fly northeast," into a building). So I do not average; I prioritize. Rank the behaviors by urgency — separation first and non-negotiable, alignment next, cohesion last and the one safely droppable — and parcel out a fixed acceleration budget equal to the maximum force in priority order: accumulate the highest-priority request, then the next, until adding one would overflow the budget, at which point I trim that request to fill the remaining budget exactly and drop the rest. In a crisis all acceleration goes to avoidance and centering is simply ignored until the crisis passes. The simpler, common approximation that the implementation below uses encodes the same precedence as fixed weights with separation weighted hardest, sums the weighted steerings, and truncates the total to the max force — the weighting encodes the precedence, the truncation enforces the budget. The flight model then integrates with momentum (so an abrupt goal change yields smooth paths, the dynamics interpolating between control points) and a *magnitude*-truncated speed cap, so that the speed and force limits shorten the command without ever rotating it. One ordering subtlety matters: I compute all boids' steerings from the current frame's positions *before* moving any of them, so each boid reacts to where its neighbors are, not to where some have already jumped this frame. Run forward a frame at a time over a population of these actors, the result is polarized, non-colliding, splitting-and-merging aggregate motion that nobody scripted — and the same scheme extends to directing the flock with a migratory urge toward an animated goal point, and to environmental obstacles via steer-to-avoid (aim just past the silhouette edge of an obstacle directly ahead) rather than the failure-prone repulsion field. The naive cost is $O(N^2)$ per frame; spatial binning or incremental nearness-testing recovers the constant-time-per-bird character of real flocks.

```python
import numpy as np

class Boid:
    # oriented point-mass flyer: position + velocity (heading = direction of velocity)
    def __init__(self, position, velocity):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)

def _norm(v):
    return float(np.linalg.norm(v))

def truncate(vector, max_magnitude):
    # cap magnitude without rotating the vector (kinematic limit: shorten, never turn)
    m = _norm(vector)
    if m > max_magnitude and m > 0.0:
        return vector / m * max_magnitude
    return vector

def desired_minus_velocity(desired, boid, max_speed):
    # steering = desired_velocity - current_velocity; fades to zero as achieved
    if _norm(desired) > 0.0:
        desired = desired / _norm(desired) * max_speed
    return desired - boid.velocity

def neighbors(boid, flock, radius, view_cos):
    # local perception: within radius AND within the field-of-view cone of the heading
    out = []
    heading = boid.velocity
    h = _norm(heading)
    for other in flock:
        if other is boid:
            continue
        offset = other.position - boid.position
        d = _norm(offset)
        if d == 0.0 or d > radius:
            continue
        if h > 0.0 and float(np.dot(heading, offset)) / (h * d) < view_cos:
            continue                       # behind the field of view -> unseen
        out.append((other, offset, d))
    return out

def separation(boid, nbrs, max_speed):
    # sum of repulsions: each neighbor's offset vector scaled by 1/r^2 (i.e. the
    # unit away-vector * 1/r) so the nearest dominates. the summed repulsion IS
    # the steering force -- don't renormalize, or the distance falloff is erased.
    acc = np.zeros_like(boid.position)
    for other, offset, d in nbrs:
        acc += -offset / (d * d)           # away-vector, offset scaled by 1/r^2
    return acc

def alignment(boid, nbrs, max_speed):
    # velocity matching: desired velocity = neighbors' average velocity; the
    # steering is that average minus the current velocity (no renormalizing)
    if not nbrs:
        return np.zeros_like(boid.velocity)
    avg_vel = np.mean([o.velocity for o, _, _ in nbrs], axis=0)
    return avg_vel - boid.velocity

def cohesion(boid, nbrs, max_speed):
    # flock centering: seek the centroid of local neighbors
    if not nbrs:
        return np.zeros_like(boid.position)
    centroid = np.mean([o.position for o, _, _ in nbrs], axis=0)
    return desired_minus_velocity(centroid - boid.position, boid, max_speed)

def steer(boid, flock, p):
    nbrs = neighbors(boid, flock, p["radius"], p["view_cos"])
    # precedence separation > alignment > cohesion, encoded as weights
    s = separation(boid, nbrs, p["max_speed"]) * p["w_sep"]
    a = alignment(boid,  nbrs, p["max_speed"]) * p["w_align"]
    c = cohesion(boid,   nbrs, p["max_speed"]) * p["w_coh"]
    return truncate(s + a + c, p["max_force"])      # spend at most the max budget

def step(flock, p, dt):
    # decide ALL steerings from this frame, THEN move everyone (react to where
    # neighbors ARE, not where some already jumped to)
    forces = [steer(b, flock, p) for b in flock]
    for b, f in zip(flock, forces):
        b.velocity = truncate(b.velocity + f * dt, p["max_speed"])   # unit mass
        b.position = b.position + b.velocity * dt

if __name__ == "__main__":
    import math
    rng = np.random.default_rng(0)
    params = dict(radius=25.0, view_cos=math.cos(math.radians(150.0)),  # ~300 deg field of view
                  max_speed=3.0, max_force=1.0,
                  w_sep=2.0, w_align=1.5, w_coh=2.0)
    flock = [Boid(rng.uniform(-15, 15, size=2), rng.uniform(-1, 1, size=2))
             for _ in range(50)]
    for _ in range(800):
        step(flock, params, dt=0.2)
    # the rules now pull the flock toward polarization (alignment), held
    # separation (separation), and clustering (cohesion) — none of it scripted.
```
