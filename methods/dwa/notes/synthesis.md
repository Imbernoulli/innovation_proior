# DWA synthesis notes

## Pain point / research question
Indoor service robot must move toward a goal in a populated/dynamic environment, replanning fast (~4 Hz, every 0.25 s) from local sonar/laser. Global planners (road-map, cell decomposition, potential-field navigation functions) need a complete, accurate, static world model and are too slow to redo when the world changes on the fly. So we need a LOCAL reactive method that uses only nearby obstacles. But existing local methods either (a) ignore the robot's dynamics — they pick a desired *direction* then turn it into a steering command, only valid if infinite forces/accelerations are available — or (b) get trapped in local minima. At high speed (RHINO ran up to 95 cm/s) bounded acceleration means the robot physically cannot make a sharp turn it "wants" to; a method that ignores this commands a turn the robot can't execute and drives into the wall.

## Ancestors (load-bearing)
- **Khatib 1986, artificial potential fields.** Obstacles exert repulsive force, goal exerts attractive force; robot follows the negated gradient of U_att+U_rep. Extremely fast, uses only nearby obstacles. Failure modes: (1) **local minima** — gradient can vanish away from the goal (U-shaped obstacles, symmetric configs), robot stalls; Borenstein&Koren noted failure to pass between closely spaced obstacles and oscillation in narrow corridors. (2) It produces a force/direction and assumes the robot can instantly accelerate that way — **ignores dynamics**. It works in position/force space, not velocity space.
- **Two-stage local methods (VFH, virtual force field, potential field) — Borenstein/Koren, Ulrich/Borenstein.** Stage 1 picks desired motion direction from an occupancy histogram; stage 2 turns it into a steering command. "Strictly justifiable only if infinite forces can be asserted on the robot." For bounded-acceleration robots this is the core flaw.
- **Simmons 1996, curvature-velocity method (CVM).** THE immediate ancestor. Formulates obstacle avoidance as **constrained optimization in velocity space (v, ω)**: constraints from physical limits (velocity, acceleration) and from obstacle configuration; choose (v,ω) maximizing an objective trading off speed, safety, goal-directedness. DWA inherits: velocity-space framing, circular-arc trajectory model, the speed/safety/goal objective. What DWA adds/refines: an explicit, clean **dynamic window** (reachable-velocity rectangle from acceleration limits over one tick) and the explicit **admissibility/braking** condition derived directly from the synchro-drive dynamics. (Both are "steer-angle-field" approaches per Brock&Khatib.)
- **Pure pursuit** (path tracking): given a path, pick a lookahead point and steer along the arc to it; tracks a *given* path, no obstacle reasoning, no dynamics-feasibility guarantee. Shows arcs are the natural primitive but doesn't solve avoidance.

## Core derivation (primary text, eqs 1–15)
Synchro-drive: translational velocity v always points along heading θ (non-holonomic). Global position by integrating velocity:
- (1) x(tn)=x(t0)+∫ v(t)cosθ(t) dt ; (2) y analogous.
- v(t), θ(t) themselves come from initial v(t0),ω(t0),θ(t0) plus accelerations v̇,ω̇ → eq (3) double-integral form: trajectory depends only on initial dynamic config + accelerations (accelerations ≈ motor currents, bounded → "controllable").
- Digital control: accelerations piecewise constant over n ticks → eq (4) exact discrete form (note the ½ω̇Δt² term inside cos).
- **Approximation**: over a small interval take v, ω *constant* (vᵢ∈[v(tᵢ),v(tᵢ₊₁)], ωᵢ∈[ω(tᵢ),ω(tᵢ₊₁)]) → eq (5), integrate → eqs (6)-(9):
  - ωᵢ≠0: Fxⁱ(t)=(vᵢ/ωᵢ)(sinθ(tᵢ) − sin(θ(tᵢ)+ωᵢ(t−tᵢ))); Fyⁱ(t)=−(vᵢ/ωᵢ)(cosθ(tᵢ) − cos(θ(tᵢ)+ωᵢ(t−tᵢ)))
  - ωᵢ=0: straight line, vᵢcosθ·t, vᵢsinθ·t.
- These trace a **circle**: center Mxⁱ=−(vᵢ/ωᵢ)sinθ(tᵢ), Myⁱ=(vᵢ/ωᵢ)cosθ(tᵢ), radius |vᵢ/ωᵢ| — eqs (10)-(12) via (Fx−Mx)²+(Fy−My)²=(vᵢ/ωᵢ)². So trajectory ≈ sequence of **circular arcs**, one per (v,ω). Each (v,ω) ↔ a curvature. Easy obstacle intersection (circle vs obstacle).
- **Error bound (3.3)**: piecewise-constant-velocity error per interval ≤ |v(tᵢ₊₁)−v(tᵢ)|·Δtᵢ, linear in Δt → vanishes as ticks shrink; used for position-uncertainty modeling. Re-measured by encoders 4×/s.

## The dynamic window approach (Section 4)
Reduce the (exponential, n-interval) search to **only the first interval**, assume the other n−1 are constant (zero accel after) → 2-D search over a single (v,ω). Justified: (a) 2-D is tractable, (b) re-searched every tick, (c) velocities stay constant by default if no new command.

Three set restrictions on (v,ω):
1. **V_s** — possible velocities from hardware: v∈[0,v_max], ω∈[−ω_max,ω_max] (a rectangle). [PythonRobotics also allows small reverse v.]
2. **V_a — admissible (braking) velocities**, eq (14):
   V_a = { (v,ω) | v ≤ √(2·dist(v,ω)·v̇_b)  ∧  ω ≤ √(2·dist(v,ω)·ω̇_b) }
   where dist(v,ω) = distance to nearest obstacle on the arc of (v,ω), v̇_b,ω̇_b = braking (deceleration) accelerations. Derivation: braking distance from speed v under constant decel a is v²/(2a); require it ≤ dist → v ≤ √(2·a·dist). So a velocity is admissible iff the robot can stop on the arc before the obstacle.
3. **V_d — dynamic window**, eq (15): velocities reachable within one tick t given current (v_a,ω_a) and accel limits v̇,ω̇:
   V_d = { (v,ω) | v∈[v_a−v̇·t, v_a+v̇·t] ∧ ω∈[ω_a−ω̇·t, ω_a+ω̇·t] } — a rectangle centered on current velocity.
- **Resulting search space V_r = V_s ∩ V_a ∩ V_d** (confirmed Brock&Khatib + lecture notes). This is what makes the turn-into-the-wall impossible: a sharp turn whose (v,ω) lies outside V_d simply isn't a candidate, so the robot stays on its straight fast arc rather than commanding an infeasible turn (the Fig.2 corridor/door example).

## Objective (eq 13)
G(v,ω) = σ( α·heading(v,ω) + β·dist(v,ω) + γ·velocity(v,ω) ), maximized over V_r.
- **heading**: progress toward goal. In original = 180° − θ, θ = angle between robot heading and goal direction, evaluated at the **predicted position** after applying (v,ω) for one interval *and then braking to a stop* (so the term accounts for the turn the robot will actually have made). Maximal when pointed straight at goal. Confirmed by Brock&Khatib align(p,v)=1−|θ|/π, normalized [0,1].
- **dist**: clearance = distance to nearest obstacle on the arc; if no obstacle on arc, a large constant. Small dist ⇒ low score ⇒ avoid; encourages going around.
- **velocity**: forward speed v (projection on translational velocity), rewards fast motion (otherwise robot would creep / could pick v=0).
- All three **normalized to [0,1]** then weighted (α,β,γ). 
- **σ smoothing**: a low-pass/smoothing of the weighted sum over the (v,ω) grid → "results in more side-clearance from obstacles" (smooths the objective surface so the max sits in a broad safe valley, not on a knife-edge next to an obstacle). 
- Why all three needed: heading alone → cuts corners, hugs/hits obstacles & ignores dynamics; dist alone → safe but won't progress; velocity alone → fast but blind. heading+velocity drive to goal fast; dist keeps clearance; together = trade-off "move fast to goal vs. ship around obstacles." 
- Note the conflict heading vs dist: max-heading often points at the obstacle between robot and goal; dist pulls away → the balance gives smooth avoidance. Why the robot doesn't fall into potential-field-style local minima as easily: it's a known limitation that local methods (incl. DWA) CAN still get stuck in U-shapes — DWA is reactive, not complete (primary text says so). Don't overclaim.

## PythonRobotics code mapping (canonical impl)
- `calc_dynamic_window`: V_s (min/max speed & yaw) ∩ V_d (current v±max_accel·dt, ω±max_dyaw·dt). The intersection by max/min of bounds = V_s∩V_d.
- `predict_trajectory` + `motion`: forward-simulate the (v,ω) arc for predict_time (=3 s) at dt (=0.1) — the circular-arc rollout (eqs 6–9 in discretized Euler form: θ+=ω·dt; x+=v cosθ·dt; y+=v sinθ·dt).
- `calc_control_and_trajectory`: grid over (v,ω) in the window at v_resolution/yaw_rate_resolution; cost = to_goal + speed + obstacle, pick min.
- `calc_to_goal_cost`: |angle to goal − heading| at trajectory end → the heading term (as a cost, so minimized).
- speed_cost = (max_speed − v_end) → the velocity term (as cost).
- `calc_obstacle_cost`: min distance to any obstacle along rollout; returns 1/min_r (small clearance ⇒ big cost) and **Inf if any point collides** — this Inf is the admissibility/V_a braking-safety filter folded into the cost (collision ⇒ rejected). 
- Differences vs original to flag in-frame: code MINIMIZES a cost sum (sign-flipped maximization), uses explicit forward rollout instead of the closed-form arc + analytic dist, folds admissibility into an Inf cost rather than the explicit √(2·a·dist) set, and adds an anti-stuck nudge (force ω to turn when v≈0). Same algorithm.

## Empirical discipline
Context.md (known, pre-method, sourced): potential-field local minima (Khatib/Borenstein-Koren); bounded actuator torque ⇒ bounded acceleration ⇒ can't turn arbitrarily sharp at speed; braking distance = v²/2a (physics). reasoning.md: invoke these as known and DERIVE the braking condition & arc geometry on the page. NEVER fabricate a trajectory measurement or claim DWA's own results (the 95 cm/s RHINO runs are the proposed method's results — exclude from reasoning; the *bounded-accel* fact and the corridor thought-experiment are fine).
