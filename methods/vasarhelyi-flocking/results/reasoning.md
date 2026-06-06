OK, let me think this through from scratch. I want a swarm of *real* quadcopters to flock — move together, coherently, without hitting each other, staying inside a bounded arena, and I want it to keep working as I crank up the flocking speed. Not on a screen; on hardware, where each drone only knows what its own noisy sensors and short-range radio tell it about nearby neighbors, and where that information arrives late.

Start from what's known to work in idealized form. Reynolds' three rules are the obvious skeleton: short-range repulsion so agents don't crowd into each other, alignment so each agent matches the velocity of its neighbors, and cohesion so the group stays together. Vicsek stripped this to its essence and showed that *alignment alone* — set my heading to the average heading of everyone within radius R, plus a little noise — already produces a genuine order–disorder phase transition, an honest collective. So alignment is the engine of coherence; I can't throw it away. Couzin's zones and Olfati-Saber's lattice-potential-plus-velocity-consensus dress this up with more structure and, in Olfati-Saber's case, with Lyapunov guarantees. All of them share one fatal assumption: the agent is an idealized particle that instantly knows its neighbors and instantly achieves whatever velocity it commands.

My agent does neither, and I have to take that seriously before writing a single control term. A quadcopter has mass; it changes velocity through a low-level controller that relaxes the real velocity toward the commanded one with some time constant τ_CTRL ≈ 1 s, and it can't accelerate harder than some a_max ≈ 6 m s⁻². It learns a neighbor's position and velocity from a radio packet that was sent a while ago — a communication delay t_del that can be a second or two — and the measurement itself is noisy, sampled at a finite rate, and only available out to a finite range r_c. So the honest model of one agent is: a desired velocity v_i^d computed from *delayed, noisy, local* neighbor data, fed into

a_i = η_i + [(v_i^d − v_i − v_i^s)/|v_i^d − v_i − v_i^s|] · min{ |v_i^d − v_i − v_i^s|/τ_CTRL , a_max },

where v_i^s is inner sensor noise and η_i is outer (wind) noise. The real velocity chases the desired velocity exponentially, but no faster than a_max allows. My whole design freedom is in the function that produces v_i^d.

Now drop the textbook alignment rule into this and watch it break. Alignment says: command a velocity that reduces the difference between mine and my neighbor's. But the neighbor's velocity I'm reacting to is t_del old, and my own velocity can't snap to the command — it lags by τ_CTRL. So I'm steering to cancel a velocity difference that no longer exists, with an actuator that responds late. That's a negative feedback loop with delay in it, and a delayed negative-feedback loop with enough gain *oscillates*. Each agent overshoots the (stale) target, the neighbor reacts to *that* overshoot one delay later, and the relative velocity rings. Worse, the ringing amplitude scales with the velocity scale of the problem: the faster the flock, the larger the velocity excursions, and at some point the excursion is a closing motion that the agents physically cannot brake out of before they touch — because braking takes acceleration, and acceleration is capped. The naive flock doesn't gently degrade; it shakes itself apart into oscillations and collisions, and the threshold drops as I speed up. This is the thing I actually have to fix. Everything below is forced by it.

Let me build v_i^d term by term, each term answering one failure.

"Don't collide" is non-negotiable, so I start there. I want a central force that pushes agent i away from a too-close neighbor j. The naive instinct from physics is a stiff potential — Lennard-Jones, hard core — that blows up as r_ij → 0 so nobody can ever touch. But stare at my agent model: the position I plug in is *measured*, and the measurement noise on relative position is on the same scale as the repulsion range itself. A stiff potential has enormous gradient at small r, so a noisy position estimate near contact produces a huge, jittery, spurious acceleration — exactly the kind of energy injection that destabilizes the actuator. A hard core fights the noise and loses. What I want instead is *soft and bounded*: a force that grows as agents close in but never explodes. A linear half-spring does it — repulsion proportional to how far inside the repulsion radius the neighbor is, and exactly zero beyond it:

v_ij^rep = p^rep (r_0^rep − r_ij) · (r_i − r_j)/r_ij   if r_ij < r_0^rep,  else 0.

The unit vector (r_i − r_j)/r_ij points from j to i, i.e. away from the neighbor — correct sign for repulsion. p^rep is a gain with units of 1/s, r_0^rep the cutoff. Linear, not quadratic-or-stiffer, precisely so that measurement noise doesn't get amplified into a singular kick; the cutoff makes it strictly local and cheap. Sum over neighbors: v_i^rep = Σ_{j≠i} v_ij^rep. That handles crowding at the level of "always push apart" — but pushing apart is reactive; it can't by itself prevent a collision if two agents are already closing faster than they can decelerate. That guarantee has to come from the alignment term, which I now have to redo around the acceleration limit. I keep the *purpose* of alignment from Vicsek — synchronize velocities so the group moves as one — but I now need it to also be a *damper* that kills the delay-induced oscillation, and to be *scalable*: at high flocking speed the velocity differences are large, and I must not fight every large velocity difference everywhere, or I over-damp the flock into sluggishness and it can't even turn collectively. Three requirements at once: relax velocity differences, stay local, stay bounded even as r_ij → 0.

My predecessor's answer was a viscous-friction term, v_ij^frict = C_frict (v_j − v_i)/(max{r_min, |d_ij|})² — always pull velocities together, with the strength decaying like 1/r² so it's local and the max{r_min,·} keeps it from blowing up at contact. That damps the oscillation, and it flies at low speed. But the law is *blind to kinematics*: how much velocity difference it tolerates at a given distance is set by a fixed 1/r² profile and a single gain, with no reference to whether that velocity difference is *survivable* given a_max. So one gain can't be right at two speeds. Turn up the gain to kill oscillation at high speed and the flock is molasses at low speed; turn it down to let the flock turn and it under-damps at high speed and the agents collide. It doesn't scale. I keep hitting this wall: a fixed friction law has no notion of "how close is too close *for this closing speed*."

So ask the kinematic question directly. Two agents with relative speed v are closing across a gap r. To not collide, I must be able to brake the closing speed to zero within r, using at most acceleration a. From v² = 2 a d, the largest closing speed I can still arrest within distance r is v = √(2 a r). That's the curve I should be enforcing. The alignment term should *not* try to equalize velocities everywhere — it should only object when the actual relative speed exceeds what the current gap can brake away. Below that envelope, leave the agents alone; let them have velocity differences, let the flock turn and breathe. Above it, dump the excess relative velocity. The alignment becomes a one-sided damper gated by a *distance-dependent speed threshold*, and that threshold is the braking curve √(2 a r).

Let me turn √(2 a r) into the actual function I'll use, because √(2 a r) has a problem at small r: its slope a/√(2 a r) → ∞ as r → 0. A threshold with infinite slope near contact is a numerical and stability hazard — tiny distance noise produces huge threshold swings. I want the high-speed part to be the constant-deceleration √-law (that's the physics), but near r = 0 I want a gentle, finite-gain approach instead. So make the threshold linear in r for small r — constant gain p — and switch to the √-branch for large r where constant deceleration a governs:

D(r, a, p) = 0                         if r ≤ 0,
           = r p                       if 0 < r p < a/p   (i.e. r < a/p²),
           = √(2 a r − a²/p²)          otherwise.

Check that this is smooth at the crossover r* = a/p². Value: left branch gives r* p = (a/p²)·p = a/p; right branch gives √(2a·a/p² − a²/p²) = √(a²/p²) = a/p. They meet. Slope: left branch slope is p; right branch slope is a/√(2ar − a²/p²), and at r* that denominator is a/p, so the slope is a/(a/p) = p. They meet too — C¹ continuous. So the linear phase gives a finite gain p near contact (a low-speed exponential-in-time relaxation rate, since "command ∝ distance" with first-order dynamics is exponential decay) and hands off seamlessly to the constant-deceleration √-phase that encodes the a_max budget at speed. p sets where the changeover happens and how stiff the low-speed approach is; a is the deceleration I'm willing to spend.

Now wire D into the alignment threshold. The maximum velocity difference I'll *tolerate* between i and j at gap r_ij is the braking-feasible speed at that gap, but I also want a floor — a constant velocity slack v^frict that's allowed regardless of distance, so the flock can turn and jostle a little without the damper constantly firing on trivial differences:

v_ij^frictmax = max( v^frict , D(r_ij − r_0^frict, a^frict, p^frict) ).

The argument is r_ij − r_0^frict, not r_ij, because the "expected stopping point" isn't contact — I want to be braked to matching velocity a comfortable offset r_0^frict before contact. Then the alignment term fires *only* when the actual relative speed v_ij = |v_i − v_j| exceeds this threshold, and it removes exactly the excess:

v_ij^frict = C^frict (v_ij − v_ij^frictmax) · (v_i − v_j)/v_ij   if v_ij > v_ij^frictmax,  else 0.

Sign and direction: (v_i − v_j)/v_ij is the unit vector along my velocity-relative-to-neighbor; multiplied by the positive excess (v_ij − v_ij^frictmax) and gain C^frict. To *reduce* the relative speed I must push my velocity *against* (v_i − v_j), so this contribution enters the desired velocity with a minus sign — it's a braking command proportional to how much I'm overspeeding the kinematic envelope. C^frict is the rate at which I bleed off that excess. Sum over neighbors: v_i^frict = Σ_{j≠i} v_ij^frict.

Why does this stabilize where the fixed friction failed? First, kinematic safety is now built in: the term guarantees that whenever two agents are closing faster than the gap can brake, a braking command appears, scaled to the overshoot — so closing speed is held inside the √(2 a r) envelope at *every* distance, automatically, at any flocking speed. Collisions are prevented by construction, not by hoping the gain was tuned right. Second, the damper only acts on the *excess* relative velocity, so within the safe envelope it's silent and the flock isn't over-damped — it can turn and accelerate collectively. The delayed feedback loop's effective gain is now bounded by the envelope rather than by a fixed constant, so the oscillation that the naive rule rang up gets its energy removed precisely in the regime where it would have grown dangerous. And locality falls out for free: the largest relative speed worth thresholding is bounded by 2 v^max (no two agents can differ by more than twice the speed cap), so the interaction range is just the distance where D(·) = 2 v^max — beyond that, no agent can ever exceed the threshold, so there's nothing to compute. The term is local without me imposing a separate radius.

That covers collisions and coordination; the flock still needs to stay together and inside the arena. Classic flocking keeps the group together with a long-range *attraction* (Reynolds' cohesion, Olfati-Saber's lattice). But long-range attraction is exactly another unbounded, delay-sensitive term, and on real hardware with finite range it's awkward. I don't actually need the group to self-attract if I instead put it in a *box* and make the box repel. So drop cohesion entirely and define a bounded arena with soft repulsive walls. Here Olfati-Saber's virtual-agent idea is the right tool: represent the wall by *shill* agents — fictitious agents sitting on the arena boundary, each moving inward at a fixed speed v^shill, perpendicular to the wall edge. A real agent near the wall should relax its velocity to the shill's velocity, which means it gets gently turned back inward before it leaves the arena.

And I already have the perfect machinery for "relax your velocity to a target velocity within a braking-feasible envelope" — it's exactly the alignment term, applied between the real agent and the shill. So reuse it:

v_is^shillmax = D(r_is − r_0^shill, a^shill, p^shill),
v_is^wall = (v_is − v_is^shillmax) · (v_i − v_s)/v_is   if v_is > v_is^shillmax,  else 0,

with r_is = |r_i − r_s|, v_is = |v_i − v_s|, v_s the inward shill velocity of magnitude v^shill. Two deliberate differences from the inter-agent case: no velocity slack (no v^frict floor) — I don't want to "tolerate" drifting toward the wall, I want it braked the moment it exceeds the kinematic threshold — and the proportional gain is fixed to 1, the strongest setting, because the wall is a hard confinement, not a gentle preference. (This is literally the inter-agent term with C → 1 and the floor removed.) Obstacles inside the arena get the same treatment with the shill moving *outward* from the obstacle surface, a single shill placed at the closest point of the obstacle to the agent. So confinement and obstacle avoidance are one mechanism, and it reuses the braking curve — no new stability question to answer.

Nothing so far sets the cruising speed; repulsion, alignment, and walls only shape *differences* and boundaries. Add a simple self-propelling term parallel to the agent's own current velocity, with constant magnitude v^flock, the target flocking speed:

v_i^selfprop = (v_i/|v_i|) v^flock.

It pushes the agent to keep moving along its present heading at v^flock; the alignment term then makes everyone's heading agree, so the group cruises together at v^flock. (If |v_i| = 0, pick a random direction to break the symmetry.)

The desired velocity is the vector sum of all of these:

ṽ_i^d = (v_i/|v_i|) v^flock + v_i^rep + v_i^frict + Σ_s v_is^wall + Σ_s v_is^obstacle.

But this sum can exceed the vehicle's hard speed cap, so I cap the *magnitude* while preserving the *direction* — I never want the cap to rotate my command, only to shorten it:

v_i^d = (ṽ_i^d / |ṽ_i^d|) · min( |ṽ_i^d| , v^max ).

That's the whole control law: self-propulsion sets the cruise, linear half-spring repulsion prevents crowding, the braking-curve-gated friction simultaneously damps oscillation and guarantees collision-feasible closing speeds, shill agents confine the flock and dodge obstacles, and a magnitude cap respects the vehicle limit. Every term traces to one failure of the naive rule on real hardware.

That's the law, but it isn't usable until I fix its parameters, and there are a lot of them. Count the free knobs: r_0^rep, p^rep for repulsion; r_0^frict, C^frict, v^frict, p^frict, a^frict for alignment; r_0^shill, v^shill, p^shill, a^shill for walls. Eleven, and they're not independent in any human-readable way — the braking crossover a/p² couples a and p, the locality radius depends on D which depends on (a,p), the right friction gain depends on v^flock. The map from these eleven numbers to "does the flock fly well at 8 m/s" is nonlinear, multimodal, and *noisy*, because the simulator itself is stochastic (sensor noise, wind, packet outages). Sweeping or hand-tuning an 11-D noisy multimodal landscape is a non-starter. I need to (a) reduce "flies well" to a single scalar I can maximize and (b) maximize it with an optimizer that tolerates noise and multimodality without gradients.

To get a single scalar I can maximize, I start from the measurable requirements: velocity correlation φ_corr (maximize, it's a cosine-similarity average inside clusters, lives in [−1,1]); collision risk φ_coll (minimize, the fraction of time pairs sit inside the r_coll = 3 m danger zone); wall collisions φ_wall (minimize, how far agents stray outside the arena); speed φ_vel (drive to v^flock); plus communication-graph health — number of disconnected agents N_disc and minimum cluster size N_min, where I keep N_min > N/5 as a "the flock didn't shatter" threshold. To form clusters I need a characteristic interaction distance; the natural one is r_cluster = max( r_0^rep, r_0^frict + D̃(v^flock, a^frict, p^frict) ), where D̃(v,a,p) is the braking distance — the r at which D(r,a,p) = v — i.e. two agents are "in the same cluster" if they're within repulsion range or within the distance over which they'd brake to matching velocity.

I want to *multiply* these into a global fitness so that *any* one of them being bad tanks the whole score — flocking is conjunctive; a beautiful correlation with frequent collisions is worthless. But raw order parameters have different ranges and different ideal points, so map each through a transfer function into [0,1] shaped to its meaning. Three shapes:

- For a quantity that should be *large* (speed reaching its target), a monotonically growing sigmoid F1(φ, φ0, d) = 1 − S(φ, φ0, d), where S decays smoothly (a raised-cosine ramp) from 1 at φ0−d to 0 at φ0. Smooth so the optimizer sees a gradient, not a cliff.
- For a quantity that should sit *at a target value with two-sided tolerance* (wall excursion near zero), a Gaussian F2(φ, s) = exp(−φ²/s²) — single peak at 0, smooth on both sides.
- For a quantity where 0 is a *hard* requirement (collision risk, disconnection), a sharp peak F3(φ, a) = a²/(φ + a)² — pinned at 1 for φ = 0 and falling off harshly. This is the crucial one for collisions: I deliberately do *not* use a hard 0/1 indicator. If collisions scored a flat zero, the optimizer would see a vast flat dead region and have no gradient telling it *which way* is less collision-prone; the smooth F3 gives a slope out of the bad region while still punishing collisions severely.

Assemble:

F = F^speed · F^coll · F^disc · F^cluster · F^wall · F^corr,
F^speed = F1(φ^vel, v^flock, v^tol),  F^coll = F3(φ^coll, a^tol),  F^disc = F3(N^disc, N/5),
F^cluster = F3(N^min, N/5, N/5),  F^wall = F2(φ^wall, r^tol),  F^corr = Θ(φ^corr)·φ^corr.

φ^corr enters multiplicatively but raw (it's already in [−1,1]), cut off at 0 by the Heaviside so anti-correlated motion can't accidentally help. The tolerances v^tol, a^tol, r^tol set the relative importance of speed/collision/wall and are chosen for a balanced weighting (e.g. v^tol = (1.5/4) v^flock, a^tol = 0.00003, r^tol = 2 m). The product is the single scalar to maximize.

With one scalar to maximize, I need an optimizer suited to it. The landscape is noisy, multimodal, 11-dimensional, no gradients. That's the home turf of CMA-ES — covariance-matrix-adaptation evolution strategy: sample a population from a Gaussian, evaluate fitness, and adapt the mean and the full covariance toward the better samples, learning the local shape of the landscape as it goes. Derivative-free, handles correlated parameters via the covariance, and robust to the stochastic fitness because it works with rank statistics over a population. Initialize each parameter at the middle of its allowed range with initial standard deviation about one-sixth of the range, population ~100, run ~150 generations (~15000 fitness evaluations), each evaluation a full multi-minute stochastic flight of the whole flock. Out comes an optimized set of the eleven parameters for a given v^flock — and re-run for v^flock = 4, 6, 8 m/s (with v^max = 6, 8, 10) to get a parameter set per speed regime, which is exactly the scalability the fixed-friction predecessor couldn't deliver.

Now to real code. The simulator hands each agent its own state, the delayed/noisy local neighbor states, distances and unit vectors to them, and a terrain grid; my job is the per-agent desired-velocity hook. I'll write it the way the working implementation does — build the local shill agents from the terrain, then accumulate self-propulsion, repulsion, the braking-gated friction for neighbors, and the braking-gated wall/obstacle term for shills, sum, and cap.

```matlab
function D = Dfunction(r, a, p)
    % the ideal braking curve: linear (constant gain p) near r=0,
    % constant-deceleration sqrt-branch for large r. C^1 at r = a/p^2.
    D = r*0;
    temp        = r < a/p/p;            % linear-phase mask (r < a/p^2)
    condition1  = r > 0 & temp;         % 0 < r < a/p^2  -> D = r*p
    condition2  = ~temp;                % r >= a/p^2     -> D = sqrt(2ar - a^2/p^2)
    D(condition1) = r(condition1) * p;
    D(condition2) = sqrt(2*a*r(condition2) - a*a/(p*p));
end
```

```matlab
function [posDesired_id, velDesired_id, accDesired_id, control_mode_id] = ...
        generate_desire_i(id, state_i, states_neighbor, dis_to_neighbor, posid_to_neighbor, ...
                          terrain, terrain_params)

    % --- 11 control-law knobs (tuned by CMA-ES) plus vehicle/world constants ---
    [r_com, v_flock, r_rep_0, p_rep, r_frict_0, c_frict, v_frict, p_frict, a_frict, ...
     r_shill_0, v_shill, p_shill, a_shill, v_max, dim, height, dr_shill, ...
     pos_shill, vel_shill] = load_params();

    posDesired_id   = [state_i(1:2); height; 0];
    velDesired_id   = zeros(4,1);
    accDesired_id   = zeros(4,1);
    control_mode_id = 7;                 % horizontal-velocity command mode

    pos2DId        = state_i(1:2);
    vel2DId        = state_i(4:5);
    vel2D_neighbor = states_neighbor(4:5,:);

    % --- build wall/obstacle shill agents local to this agent from the terrain ---
    %     each shill sits at an obstacle cell taller than the agent and "pushes"
    %     the agent away (unit vector from obstacle to agent); arena-edge shills
    %     are supplied in pos_shill/vel_shill already.
    if ~isempty(terrain)
        r_w   = 5;
        r_sub = floor((pos2DId(2)-terrain_params(2,1))/terrain_params(2,2));
        c_sub = floor((pos2DId(1)-terrain_params(1,1))/terrain_params(1,2));
        h_sub = floor(r_w/terrain_params(2,2));  w_sub = floor(r_w/terrain_params(1,2));
        [h,w] = size(terrain);
        r_min = max(1,r_sub-h_sub); r_max = min(h,r_sub+h_sub);
        c_min = max(1,c_sub-w_sub); c_max = min(w,c_sub+w_sub);
        [r_obs,c_obs] = find(terrain(r_min:r_max,c_min:c_max) > state_i(3));
        if ~isempty(r_obs)
            r_obs = r_obs + r_min - 1;  c_obs = c_obs + c_min - 1;
            temp_p_shill = [(c_obs'*terrain_params(1,2))+terrain_params(1,1);
                            (r_obs'*terrain_params(2,2))+terrain_params(2,1)];
            temp      = pos2DId - temp_p_shill;
            vel_shill = [vel_shill, temp./vecnorm(temp)];   % shill velocity: away from obstacle
            pos_shill = [pos_shill, temp_p_shill];
        end
    end

    % --- self-propulsion toward target speed v_flock, along current heading ---
    velIdNorm = norm(vel2DId);
    if velIdNorm == 0
        vr = rand(dim,1); vr = vr/norm(vr);  vFlockId = v_flock * vr;   % break symmetry
    else
        vFlockId = v_flock * vel2DId/velIdNorm;
    end

    vRepId   = zeros(2,1);
    vFrictId = zeros(2,1);
    if ~isempty(dis_to_neighbor)
        % --- short-range linear half-spring repulsion (noise-robust, bounded) ---
        inRep = find(dis_to_neighbor < r_rep_0);
        if ~isempty(inRep)
            d = repmat(dis_to_neighbor(inRep), dim, 1);
            vRepId = p_rep * sum((r_rep_0 - d) .* posid_to_neighbor(:,inRep)./d, 2);
        end

        % --- braking-gated velocity alignment ("friction") ---
        % threshold = max(floor slack, braking curve at the gap minus offset)
        vijFrictMax = max(v_frict, Dfunction(dis_to_neighbor - r_frict_0, a_frict, p_frict));
        velij = repmat(vel2DId,1,length(dis_to_neighbor)) - vel2D_neighbor;  % v_i - v_j
        vij   = sqrt(sum(velij.^2,1));                                       % |v_i - v_j|
        inFr  = find(vij > vijFrictMax);            % fire only above the kinematic envelope
        if ~isempty(inFr)
            vN = repmat(vij(inFr),dim,1);  vM = repmat(vijFrictMax(inFr),dim,1);
            vFrictId = -c_frict * sum((vN - vM).*velij(:,inFr)./vN, 2);     % brake the excess
        end
    end

    % --- walls/obstacles via shill agents: same braking-gated form, no slack, gain 1 ---
    vShillId = zeros(dim,1);
    posis = repmat(pos2DId,1,size(pos_shill,2)) - pos_shill;
    disis = sqrt(sum(posis.^2,1));
    inS   = find(disis < r_com);  disisIn = disis(inS);
    visFrictMax = Dfunction(disisIn - r_shill_0, a_shill, p_shill);
    velis = repmat(vel2DId,1,length(disisIn)) - v_shill * vel_shill(:,inS);  % v_i - v_s
    vis   = sqrt(sum(velis.^2,1));
    inFrS = find(vis > visFrictMax);
    if ~isempty(inFrS)
        vN = repmat(vis(inFrS),dim,1);  vM = repmat(visFrictMax(inFrS),dim,1);
        vShillId = - sum((vN - vM).*velis(:,inFrS)./vN, 2);                 % gain fixed to 1
    end

    % --- superpose all terms, then cap the magnitude (keep direction) at v_max ---
    v2D = vFlockId + vRepId + vFrictId + vShillId;
    s = norm(v2D);
    if s > v_max,  v2D = v2D./s * v_max;  end
    velDesired_id(1:2) = v2D;
end
```

Tracing the causal chain once more: the naive alignment rule, dropped onto a delayed, inertia-limited, speed-capped vehicle, becomes a delayed negative-feedback loop that self-oscillates and, at speed, collides — because braking needs distance and acceleration is bounded. Linear half-spring repulsion (soft, so sensor noise isn't amplified) stops crowding; the kinematic limit v ≤ √(2 a r) becomes a smooth braking curve D(r,a,p) (linear near contact for finite gain, constant-deceleration √-branch at range, stitched C¹), and gating the alignment to fire only when relative speed exceeds D simultaneously damps the oscillation's excess energy and guarantees every closing motion is brakeable — with a velocity-slack floor so the flock can still turn; the same braking-gated form against inward-moving shill agents confines the flock to the arena and dodges obstacles, replacing fragile long-range cohesion; self-propulsion sets the cruise speed and a magnitude cap respects the vehicle limit; and because the eleven knobs map nonlinearly and noisily to behavior, they're set by maximizing a single conjunctive fitness — a product of smoothly-shaped order parameters (collision risk through a soft peak, never a hard 0/1) — with CMA-ES, re-run per target speed to get the speed-scalable flocking the fixed-friction predecessor could not reach.
