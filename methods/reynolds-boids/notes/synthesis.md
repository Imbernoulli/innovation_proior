# Synthesis — Reynolds Boids

## Sources retrieved (this run)
- PRIMARY: Reynolds 1987 "Flocks, Herds, and Schools" full PDF (refs/reynolds1987-paper.pdf, 16 pp, clean pdftotext). Read end to end.
- PRIMARY-2: Reynolds 1999 "Steering Behaviors for Autonomous Characters" (refs/steering1999.html). Gives the clean vector formulation: steering = desired_velocity − velocity; vehicle model with max_force/max_speed truncation; separation/cohesion/alignment definitions; neighborhood = distance + view-angle.
- THIRD-PARTY: Conrad Parker boids pseudocode (refs/parker-boids.html) — rule1 cohesion (pc−pos)/100, rule2 separation, rule3 alignment (pv−vel)/8, speed limit, bound position.
- CODE: Ben Eater canonical boids (code/boids_ref.py, JS) — per-neighbor loops for the three rules, visualRange neighborhood, speed limit, keepWithinBounds.

## Pain point (in-frame, 1986/87)
Scripting each bird's path by hand for a flock is tedious, error-prone, impossible to keep collision-free per frame, and hard to edit globally. Prior foreflock (Amkraut/Girard "Eurythmy" force-field system) used global force fields → an object hitting a field head-on isn't turned aside, only slowed; fields too strong up close, too weak far. Central-force flock models can't bifurcate around obstacles and cause whole scattered flock to converge to one centroid. Need: emergent flock motion from a per-bird, local, constant-time-per-bird rule.

## Lineage / ancestors
- Reeves particle systems (1983): large collections of dot particles, each with own state (color/opacity/position/velocity), created/age/die. But particles DON'T interact. Boids = "subobject system" generalization: particle → oriented geometric object; and crucially boids interact (depend on external state).
- Geometric flight / Logo turtle geometry (Papert; Abelson/diSessa 3D turtle): incremental forward translation + steering rotations (pitch/yaw); conservation of momentum; viscous speed damping → max speed; max acceleration as fraction of max speed truncates over-anxious requests. This is the locomotion layer.
- Actor model (Hewitt): encapsulated process+state+procedure, message-passing → the right computational abstraction for independent interacting birds.
- Biology: Shaw (opposing urges: stay close vs avoid collision); Partridge (fish influenced by near neighbors, contribution ∝ 1/r² or 1/r³); Potts "chorus line" maneuver-wave; the "itself / 2-3 nearest neighbors / rest of flock" perception (constant-time argument).

## The three rules (1987 exact wording, decreasing precedence)
1. Collision Avoidance: avoid collisions with nearby flockmates (static, position-based).
2. Velocity Matching: match velocity with nearby flockmates (dynamic; predictive collision avoidance — if you match velocity you won't collide soon; keeps separation invariant).
3. Flock Centering: steer toward centroid of nearby flockmates (localized; strong only at flock boundary where neighbors are one-sided; allows bifurcation).
NOTE the precedence order: separation > alignment > cohesion.

## Combination — prioritized acceleration allocation (1987)
Each behavior emits an acceleration REQUEST: a 3D vector truncated to unit magnitude, scaled by a "strength" in [0,1]. Naive combination = weighted average → fails: opposing requests cancel ("fly N" + "fly E" → tiny turn into wall; indecision at a brick wall). Better: prioritized allocation — consider requests in priority order, accumulate into accelerator and accumulate magnitudes; when accumulated magnitude exceeds max_accel, trim the last request; remaining lower-priority behaviors go unsatisfied. A fixed acceleration budget parceled out by priority.

## Perception / neighborhood (1987 + 1999)
- Localized view is ESSENTIAL — central-force (global) model fails (whole flock converges; no bifurcation). "An interesting result… is that flocking depends on a limited, localized view."
- Neighborhood = spherical zone of sensitivity: a radius + an inverse-exponential-of-distance falloff (two params: radius, exponent). 1999: neighborhood = a distance + an ANGLE (field of view). Real birds ~300° FOV. Should ideally be forward-exaggerated, proportional to speed.
- Falloff: early version weighted attraction/repulsion LINEARLY by distance → "bouncy/spring-like" flock, cartoony, not realistic. Changed to INVERSE SQUARE → "more gravity-like… better damped, more natural." Matches Partridge's 1/r²–1/r³ fish data. This is the why behind separation's 1/r² weighting.

## 1999 clean vehicle model (the formulation to ground code in)
desired_velocity = normalize(target − position) * max_speed  (seek)
steering = desired_velocity − velocity
steering_force = truncate(steering, max_force)
acceleration = steering_force / mass
velocity = truncate(velocity + acceleration, max_speed)
position = position + velocity
- Separation: for each nearby char, repulsive force = normalize(pos − other.pos) scaled by 1/r (offset vector scaled by 1/r²); sum. "1/r is just a setting that worked well."
- Cohesion: average position of neighbors = target; steer = (avg_pos − pos) [or seek it].
- Alignment: average velocity of neighbors = desired_velocity; steering = avg_vel − velocity.
- Flocking = normalize each of the 3 steering components, scale by 3 weights, sum. → 9 params: weight+distance+angle per rule.

## Geometric flight constraints (the dynamics that make it real, 1987)
- Conservation of momentum (object in flight tends to stay in flight) → smooths abrupt goal changes (own dynamics = interpolation between control points).
- Viscous speed damping → max_speed cap.
- max_acceleration as fraction of max_speed → truncates acceleration requests, smooth speed/heading changes. Finite energy.
- min_speed (default 0).

## Migratory urge / scripting
A global target (direction or point) → bounded acceleration turning boid toward it; per-boid "migratory goal register"; lets the animator lead the flock by animating goal point ahead of flock. Momentum smooths.

## Obstacle avoidance
- force-field (repulsion from obstacle): head-on failure (no side thrust), too strong close / weak far, no peripheral-vision discrimination.
- steer-to-avoid (better, vision-like): only consider obstacles directly ahead (Z-axis intersection); find silhouette edge nearest impact; aim one body length beyond it.

## Complexity
Naive O(N²) — each boid reasons about each other. Real flocks ~constant-time-per-bird (2-3 neighbors). Approaches: spatial bins (dynamic partitioning), incremental nearness testing. Separate processor per boid → O(N).

## Implementation environment (1987)
Symbolics Common Lisp, Flavors (OO), on Symbolics 3600 Lisp Machine; S-Geometry/S-Dynamics. 80 boids naive O(N²) ≈ 95 s/frame.

## Design-decision → why table
- 3 separate rules not one potential: separation (position-only) + velocity-matching (velocity-only) are COMPLEMENTARY — static avoidance sets the min separation, velocity-matching maintains it (predictive). One position potential can't do the predictive part.
- precedence sep>align>coh: collision avoidance is non-negotiable (a brick wall dead ahead); centering is least urgent (can be dropped temporarily).
- prioritized allocation not weighted average: averaging cancels opposing urgent requests → indecision/crash; "fly NE" between buildings is wrong.
- local neighborhood not global: global central force → flock can't bifurcate, whole scattered flock converges. Localization IS the discovery.
- view ANGLE (FOV) not just radius: birds have ~300° FOV; a neighbor directly behind shouldn't pull you; forward-weighting improves leading-edge boids.
- 1/r² separation falloff not linear: linear → bouncy spring flock; inverse-square → gravity-like, better damped, natural; matches Partridge fish data.
- steering = desired − velocity: a velocity error, so the force always pushes toward the desired motion and vanishes when achieved — gives smooth asymptotic approach, no overshoot-by-construction (vs commanding raw position deltas).
- truncate(steering, max_force) then truncate(velocity, max_speed): finite muscle/energy → bounded acceleration; viscous drag → bounded speed; truncation (not rescale of direction) preserves the steering DIRECTION while honoring the limit.
- separation summed per-neighbor (not vs centroid): want to be pushed off the SINGLE nearest intruder hardest, not an averaged blob — averaging would let a symmetric ring cancel.

## Faithful Python plan (grounds in 1999 vehicle model + Ben Eater structure)
- Boid: position, velocity (2D numpy).
- neighbors(i): distance < radius AND within view half-angle of velocity heading (dot test). exclude self.
- separation: sum over neighbors of (pos_i − pos_j) * (1/r²) i.e. normalize(offset)/r ; then steer = normalize(sep)*max_speed − vel (desired-velocity form), truncated to max_force.
- alignment: avg neighbor velocity = desired; steer = normalize(avg)*max_speed − vel.
- cohesion: avg neighbor position = target; desired = normalize(target − pos)*max_speed; steer = desired − vel.
- combine: w_sep*sep + w_align*align + w_coh*coh (weights encode precedence sep>align>coh); truncate total to max_force.
- integrate: vel = truncate(vel + accel, max_speed); pos += vel*dt.
- bound/wrap arena.
