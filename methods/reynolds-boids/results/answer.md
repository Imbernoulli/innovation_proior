# Boids — a distributed behavioral model of flocking, herding, and schooling

## Problem

Animate the aggregate motion of a flock of birds (or a school of fish, or a herd of animals) without scripting each individual's path. Hand-scripting at flock scale is tedious, error-prone, cannot guarantee per-frame collision-freedom between every pair, and is nearly impossible to edit globally. The motion must look right — polarized, fluid, non-colliding, able to split around an obstacle and re-form — and be cheap enough to animate many birds for many frames.

## Key idea

Don't author the flock; author one bird and let the flock *emerge*. Model each bird (a "boid") as an autonomous actor — an oriented point-mass flyer with momentum, a max speed, and a max acceleration — that steers using only its **local** perception of neighbors. The complex global motion is the aggregate of three simple local steering rules, in order of decreasing precedence:

1. **Separation (collision avoidance):** steer to avoid crowding nearby flockmates (position only).
2. **Alignment (velocity matching):** steer to match the average velocity of nearby flockmates (velocity only — predictive collision avoidance: matching velocity keeps separations invariant).
3. **Cohesion (flock centering):** steer toward the centroid of nearby flockmates (position, longer-range).

Locality is the mechanism, not an optimization: a *global* central-force model collapses a scattered flock onto one point and cannot bifurcate around obstacles, whereas a localized centroid is near-zero deep inside the flock (neighbors surround you) and strong only at the boundary (neighbors are one-sided), which lets the flock split and re-merge. The velocity-matching and centering rules are expressed as `steering = desired_velocity − current_velocity`, a velocity error that points toward the wanted motion and fades to zero as it is achieved; separation is a summed repulsion taken directly as a force.

## Algorithm

**Neighborhood (local perception).** A boid perceives another only if it is within a sensitivity **radius** *and* within a **view angle** (field-of-view cone) about the boid's heading — a neighbor directly behind is unseen.

**The three steering vectors:**
- Separation: sum over close neighbors of the offset vector `(pos_self − pos_j)` scaled by `1/r²` — equivalently the unit away-vector scaled by `1/r` (a gravity-like, well-damped falloff, milder than the raw repulsion — close neighbors dominate, distant ones whisper; summing per-neighbor, not steering from their averaged position, so a symmetric crowd doesn't cancel to no push). The summed repulsion is the steering force itself.
- Alignment: desired velocity = the average velocity of neighbors; steering = `avg_velocity − current_velocity`.
- Cohesion (a seek): target = the average position (centroid) of neighbors; desired velocity = `normalize(target − pos) · max_speed`; steering = `desired − current_velocity`.

**Combination.** Combine by *priority*, not by averaging. Averaging cancels opposing urgent advice (a hard-left avoidance and a hard-right cohesion average to a tiny turn straight into a wall; "fly north" + "fly east" averages to "fly northeast" into a building). The robust scheme is **prioritized acceleration allocation**: take requests in priority order (separation first, cohesion last), accumulate into a budget equal to `max_force`, trim the request that would overflow, and drop the rest — in a crisis all acceleration goes to avoidance and cohesion is dropped. A simpler, common approximation gives the rules fixed weights with separation weighted hardest, sums the weighted steerings, and truncates the total to `max_force`.

**Flight model / integration** (one frame, computing all steerings before moving any boid):
```
steering_force = truncate(combined_steering, max_force)
acceleration   = steering_force / mass
velocity       = truncate(velocity + acceleration·dt, max_speed)   # truncate caps magnitude, never rotates
position       = position + velocity·dt
```
`truncate` caps a vector's magnitude without changing its direction, so the speed/force limits shorten but never turn the command. Momentum makes abrupt goal changes smooth (the dynamics interpolate between control points). A migratory urge (a global goal direction/point) lets an animator lead the flock; `steer-to-avoid` (aim just past the silhouette edge of an obstacle directly ahead) handles environmental obstacles better than a repulsion force field, which fails to turn a head-on approach. Naive cost is O(N²) per frame; spatial binning or incremental nearness-testing recovers the constant-time-per-bird character of real flocks.

## Code

A self-contained 2-D/3-D simulation built on the vector steering model above (works unchanged in any dimension since all operations are vector ops).

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

The whole model is the per-boid steering function plus the bounded-acceleration / bounded-speed flight step: separation keeps boids from crowding, alignment matches velocities (predictive avoidance), cohesion holds the group together, all over a purely local neighborhood — and coherent, splitting-and-merging flock motion emerges from running it across the population.
