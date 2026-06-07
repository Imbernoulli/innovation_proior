# Synthesis — Vásárhelyi flocking

## Pain point
Decentralized flocking models (Reynolds boids, Vicsek SPP, Couzin zones, potential-field/Olfati-Saber)
are stable in idealized simulation but OSCILLATE and COLLIDE on real drones. Why: real agents have
(a) finite communication/sensor DELAY t^del (received neighbor state is stale), (b) INERTIA / limited
acceleration a^max and a low-level velocity controller with time const tau^CTRL (cannot snap to a
commanded velocity), (c) a SPEED CAP v^max, (d) sensor NOISE + outer (wind) noise, (e) LOCAL
communication (range r^c). Delay + inertia + a velocity-difference-based alignment = a classic
delayed negative-feedback loop -> self-excited oscillations whose amplitude grows with speed; at high
closing speed the braking distance exceeds the gap and you get collisions. This is the "reality gap".

## Goal precisely
A distributed control law (each agent computes a desired velocity v_i^d from delayed local neighbor
state) that is COLLISION-FREE, COHERENT (high velocity correlation), CONFINED to a bounded arena
(handles walls + obstacles), and SCALABLE in both flock size and flocking speed v^flock — under the
realistic agent model above.

## Ancestors (load-bearing) and their gap
- Reynolds 1987 "Flocks, Herds, and Schools": 3 steering rules — separation (short-range repulsion),
  alignment (match neighbor velocity), cohesion (toward local center of mass). Geometric, no dynamics,
  no delay/inertia; alignment is unbounded -> unstable on real agents.
- Vicsek 1995 "Novel type of phase transition in a system of self-driven particles": minimal SPP.
  theta_i(t+1) = <theta_j>_{r<R} + eta ; r_i(t+1)=r_i+v*dt*(cos,sin). Constant speed v, polar
  alignment to neighbor-average heading + noise eta; order parameter v_a = |sum v_i|/(N v). Shows
  order-disorder transition. But: constant speed, no collision avoidance, no inertia/delay, periodic
  boundaries — not executable on drones.
- Couzin et al. 2002 zonal model: zone of repulsion / orientation / attraction (concentric). Adds
  structure but still kinematic, idealized, no delay/accel limits.
- Potential-field / Olfati-Saber 2006 "Flocking for multi-agent dynamic systems": collective
  potential penalizing deviation from an alpha-lattice; gradient + velocity-consensus + navigational
  feedback; obstacles via virtual beta-agents, goal via gamma-agent. Provable in ideal double-integrator
  dynamics, but assumes no delay, instantaneous actuation; gradient potentials can be stiff (hard-core)
  and the consensus term is unbounded -> again destabilizes under delay. The virtual-agent idea is the
  conceptual ancestor of the "shill" wall/obstacle agents.
- Direct ancestor: Virágh/Vásárhelyi 2014 (IROS) "Flocking algorithm for autonomous flying robots":
  introduces the realistic agent model (delay, inertia, inner+outer noise, locality) AND a viscous
  friction-like alignment term v_ij^frict = C^frict*(v_ij - v^frict)*... with a FIXED velocity slack
  v^frict, plus linear short-range repulsion, plus shill wall agents. Damps oscillations. Gap left:
  the fixed-slack friction does not scale with speed — at high v^flock the allowed velocity difference
  is the same regardless of how close agents are, so you either over-damp (sluggish) at distance or
  under-damp (collide) when closing fast. Not scalable to v>4 m/s; trajectories oscillatory. The 2018
  fix: make the slack DISTANCE-DEPENDENT via a braking curve.

## The key insight (re-derive in reasoning)
Because acceleration is limited (a^max), an agent needs DISTANCE to bleed off a velocity difference.
So at gap r, the maximum closing speed that can still be braked to zero before contact is set by
kinematics: v <= sqrt(2 a r) (from v^2 = 2 a d). Therefore the alignment should NOT force equal
velocities everywhere; it should only act when the actual relative speed v_ij EXCEEDS what the gap can
safely accommodate. Define a smooth "ideal braking curve" D(r,a,p):
  D = 0 (r<=0); = r*p (small r, constant-gain linear phase); = sqrt(2 a r - a^2/p^2) (large r,
  constant-deceleration phase). The crossover r = a/p^2 makes value+slope continuous (at r=a/p^2:
  r*p = a/p, and sqrt(2a*a/p^2 - a^2/p^2)=sqrt(a^2/p^2)=a/p ✓; slope of rp is p; slope of sqrt branch
  d/dr sqrt(2ar - a^2/p^2) = a/sqrt(2ar-a^2/p^2) = a/(a/p)=p ✓ -> C^1 continuous). The linear phase
  avoids infinite gain / division blowup at small r where the sqrt slope -> infinity; p sets the
  low-speed relaxation rate (exponential-in-time approach at low speed). a sets the high-speed
  constant-deceleration regime.
Then v_ij^frictmax = max(v^frict, D(r_ij - r0^frict, a^frict, p^frict)) — a floor v^frict of allowed
slack (lets the flock turn collectively without fighting tiny velocity diffs) plus the braking curve.
Alignment fires only if v_ij > v_ij^frictmax, magnitude C^frict*(v_ij - v_ij^frictmax), direction
-(v_i-v_j)/|v_i-v_j| (reduce the difference). This is a distance-gated, speed-scalable viscous damper.
WHY it stabilizes: it bounds closing speed to the braking-feasible envelope at every gap, so the
delayed feedback loop's gain stays below the oscillation threshold AND collisions are kinematically
prevented; the damper removes energy from the relative-velocity oscillation. Locality is automatic:
interaction range is the distance where D = 2 v^max (beyond that no agent can exceed the threshold).

## Term-by-term build
1. Repulsion (Eq 2-3): half-spring linear, v_ij^rep = p^rep (r0^rep - r_ij)(r_i-r_j)/r_ij for r<r0^rep.
   WHY linear not Lennard-Jones/hard-core: noise in measured position would make a stiff potential
   inject huge spurious accelerations; a soft extended repulsion is robust to noise. (Paper found the
   optimizer PREFERS extended-smoother repulsion over hard-core — diagnostic in context.)
2. Alignment/friction (Eq 4-7): the distance-dependent braking-curve damper above.
3. Walls/shill (Eq 8-9): virtual shill agents along arena edges (and at closest obstacle point) moving
   inward (outward for obstacles) at v^shill; real agent relaxes its velocity to the shill's velocity
   with the SAME friction form but NO slack (v^frict=0) and gain fixed to 1 (strongest), so the wall
   is a hard confinement. Replaces long-range cohesion/attraction — bounded arena keeps flock together.
4. Self-propulsion: (v_i/|v_i|) v^flock — pushes speed toward target v^flock along current heading.
5. Superpose (Eq 10) and cap magnitude at v^max keeping direction (Eq 11).

## Tuning (CoFlyers task)
11 params { r0^rep, p^rep, r0^frict, C^frict, v^frict, p^frict, a^frict, r0^shill, v^shill, p^shill,
a^shill } — nonlinear, noisy, multimodal map to behavior. Hand-tuning hopeless. Define order params
(velocity correlation phi^corr, collision risk phi^coll, wall collisions phi^wall, speed phi^vel,
disconnected N^disc, min cluster N^min) -> single scalar fitness F = product of partial fitnesses via
transfer functions F1 (monotone), F2 (Gaussian peak at 0), F3 (sharp peak at 0). Maximize with CMA-ES
(derivative-free, handles noise/multimodality), pop 100, 150 gens. Collision fitness must be smooth
(not a hard 0) or optimizer can't find gradients out of bad regions.

## Code grounding
CoFlyers Vasarhelyi_module_generate_desire_i.m + Dfunction.m. Signs: vFrictId = -c_frict*sum(...),
vShillId = -sum(...). Final = vFlockId + vRepId + vFrictId + vShillId, then clamp to v_max.
