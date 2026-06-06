# Context: distributed flocking control for real autonomous drone swarms

## Research question

How do you make a *large group of real autonomous flying robots* — quadcopters, each running only on-board sensing and computation, exchanging only local radio messages — move together as a coherent flock that is **collision-free**, stays **confined to a bounded arena** (and avoids obstacles), and remains **stable across a wide range of flocking speeds and flock sizes**?

The motivation is concrete and physical. Decades of statistical-physics and computer-graphics work show that simple decentralized rules reproduce the rich collective motion of bird flocks and fish schools in *simulation*. The hard, unsolved problem is the transfer to hardware. A real quadcopter is not an idealized self-propelled point: it receives neighbor positions and velocities with a non-negligible **communication delay**; it cannot change its velocity instantly because of **inertia and a finite maximum acceleration**, mediated by a low-level velocity controller with its own relaxation time; it has a hard **speed cap**; its on-board sensors return **noisy** relative positions and velocities at a finite **refresh rate**; and its radio has a finite **communication range** so it sees only nearby neighbors, with **outages** that grow with distance. Under these deficiencies the textbook rules do not just degrade gracefully — they go unstable: the alignment rule, which matches an agent's velocity to its delayed neighbors, closes a delayed negative-feedback loop that breaks into **self-excited velocity oscillations** whose amplitude grows with flocking speed, and at high closing speeds an agent cannot decelerate within the gap to its neighbor, so the oscillation turns into a **collision**. A solution must therefore be a distributed control law that produces coherent confined flocking *while explicitly respecting* delay, inertia, acceleration limits, noise, and locality — and it must work at speeds well beyond the few-m/s regime where naive models survive.

## Background

**Reynolds' three rules (boids, 1987).** Reynolds modeled flocking as three local steering behaviors per agent: *separation* (steer to avoid crowding nearby flockmates — short-range repulsion), *alignment* (steer toward the average heading/velocity of local neighbors), and *cohesion* (steer toward the average position of local neighbors). It is a purely geometric/kinematic recipe — there is no agent dynamics, no delay, no acceleration limit, and the alignment steering is unbounded. It produces convincing flocks on a screen precisely because the simulated agents have no inertia and perfect, instantaneous knowledge of neighbors.

**Vicsek's self-propelled particles (1995).** The minimal physics model: N point particles move at *constant* speed v; at each discrete step each particle sets its heading to the average heading of all neighbors within radius R, plus an angular noise η: θ_i(t+1) = ⟨θ_j⟩_{|r_i−r_j|<R} + η, with r_i(t+1) = r_i(t) + v Δt (cosθ_i, sinθ_i). The order parameter φ = |Σ_j v_j| / (N v) measures global alignment, and the model exhibits a noise-/density-driven order–disorder phase transition. This pinned down *alignment* as the essential ingredient of collective order — but the model has constant speed, no collision avoidance, no inertia or delay, and periodic boundaries; it is a statistical-mechanics object, not a controller.

**Couzin's zonal model (2002).** Generalizes the rules into three concentric zones around each agent — an inner *zone of repulsion*, a middle *zone of orientation* (alignment), and an outer *zone of attraction* — and shows how group structure (swarm, torus, parallel/dynamic groups) emerges as the zone sizes change. Richer collective behavior, but still kinematic and idealized: no communication delay, no acceleration constraints, no sensor noise.

**Potential-field / consensus flocking (Olfati-Saber, 2006).** Brings flocking into control theory with provable guarantees. Agents are double integrators; a collective potential penalizes deviation from an "α-lattice" (a target inter-agent spacing), and the control law is the gradient of that potential plus a *velocity-consensus* term that drives neighbor velocities to agreement, plus a navigational feedback term toward a goal. Obstacles are handled by virtual "β-agents" on obstacle surfaces and a group objective by a virtual "γ-agent." This gives Lyapunov stability and flock cohesion — but only under ideal double-integrator dynamics with *instantaneous, noise-free* actuation and sensing. The gradient of a lattice potential can be stiff (large forces for small spacing errors), and the consensus term is unbounded; both destabilize once delay and acceleration limits are present. The virtual-agent idea, however, is the conceptual seed for representing walls and obstacles as agents rather than as forces.

**The realistic-agent model and the friction-like alignment term (Virágh, Vásárhelyi et al., 2014).** The direct predecessor makes the deficiencies of a real flying robot first-class citizens of the model. It defines an agent whose desired velocity v_i^d is a function of *delayed, noisy, local* neighbor positions and velocities, and whose actual acceleration obeys
a_i(t) = η_i(t) + [ (v_i^d − v_i − v_i^s) / |v_i^d − v_i − v_i^s| ] · min{ |v_i^d − v_i − v_i^s| / τ_CTRL , a_max },
i.e. the real velocity relaxes toward the desired velocity with characteristic time τ_CTRL but is capped at acceleration a_max, with inner sensor noise v_i^s, outer (wind) noise η_i, finite sensor refresh t_s, time delay t_del, and communication range r_c. With quadcopter values τ_CTRL ≈ 1 s, a_max = 6 m s⁻², σ_s = 0.005 m² s⁻², t_s = 0.2 s, t_del = 0–2 s, r_c = 30–140 m, and desired speeds saturated at v_max = 4 m s⁻². Crucially, it observes that a naive alignment rule is *destabilized* by delay and noise, and damps the oscillations with a **viscous-friction-like alignment term**: a pairwise term that pulls neighbor velocities together, of the form
v_ij^frict = C_frict · (v_j − v_i) / (max{r_min, |d_ij|})²,
chosen to satisfy three requirements — relax velocity differences, remain local, and stay upper-bounded even as the inter-agent distance goes to zero. Its short-range repulsion is a *linear half-spring* (with the explicit argument that a stiff Lennard-Jones-type potential would, under noisy position measurements, inject huge spurious accelerations, whereas a soft linear repulsion is robust to noise), and confinement is via virtual *shill* wall agents whose velocity the real agents align to. This model flies on real quadcopters at low speed.

**The diagnostic failure mode that motivates the next step.** The 2014 viscous term damps oscillations by a *fixed* law: the amount of velocity-difference it tolerates depends only on inter-agent distance through a 1/r² decay, not on whether that velocity difference is *kinematically survivable*. With limited acceleration a_max, an agent needs a *distance* proportional to (closing speed)² to brake to zero. So a single global friction strength either over-damps at low speed (sluggish, agents fight tiny velocity differences and the flock cannot turn collectively) or under-damps at high speed (two agents approach faster than the gap allows them to brake → collision). Empirically the 2014 model does not scale past a few m/s: trajectories stay oscillatory and intergroup distances become dangerous as flocking speed rises. The pre-method fact to carry forward is this observed instability — the alignment threshold must become *distance-dependent* and tied to the braking kinematics, not a fixed 1/r² law.

## Baselines

- **Reynolds boids (1987).** Separation + alignment + cohesion as geometric steering. Gap: no dynamics, delay, or acceleration limit; unbounded alignment → unstable on hardware.
- **Vicsek SPP (1995).** Constant-speed heading alignment + noise; demonstrates the order–disorder transition. Gap: constant speed, no collision avoidance, no inertia/delay, periodic boundaries — not executable on drones.
- **Couzin zonal model (2002).** Repulsion/orientation/attraction zones; emergent group shapes. Gap: kinematic and idealized; no delay, acceleration limits, or noise.
- **Olfati-Saber potential-field flocking (2006).** Lattice-potential gradient + velocity consensus + navigation, obstacles/goal as virtual β/γ agents; provably stable for ideal double integrators. Gap: assumes instantaneous noise-free actuation; stiff gradients and unbounded consensus destabilize under delay and acceleration limits.
- **Virágh/Vásárhelyi realistic model (2014).** Realistic agent (delay, inertia, a_max, noise, locality) + linear repulsion + 1/r² viscous-friction alignment + shill walls; flies on quadcopters. Gap: the *fixed* friction law does not scale with speed — no distance-dependent, braking-aware velocity threshold — so it over-/under-damps and collides above a few m/s.

## Evaluation settings

The natural yardstick is a stochastic simulation of the realistic-agent model in a square arena, plus real outdoor multi-drone flights. Simulation: a square arena of characteristic size L_arena (e.g. 250 m), up to 100s–1000s of agents, communication range r_c on the order of tens of meters, a constant communication delay (e.g. 1 s), and target flocking speeds spanning several m/s up to tens of m/s, each with a corresponding speed cap v_max. The dangerous-collision distance is r_coll = 3 m. Hardware: a swarm of quadcopters (e.g. 30 units), each with a Pixhawk autopilot and an on-board companion computer issuing desired-velocity commands at ~20 Hz, two parallel radios (a long-range low-bandwidth link and a short-range high-bandwidth link) broadcasting position/velocity status packets. Quality is judged by the *order parameters* below; the question is whether a control law exists that keeps them all good simultaneously at high speed.

Order parameters (the measurable requirements of "good flocking"), to be computed from a trajectory and used as optimization targets:
- **velocity correlation** φ_corr — cluster-internal time-average of v_i·v_j/(|v_i||v_j|), to be maximized;
- **collision risk** φ_coll — time-average over agent pairs of the Heaviside Θ(r_coll − r_ij), the fraction of time pairs sit inside the 3 m danger zone, to be minimized;
- **wall collisions** φ_wall — time-average of how far agents stray outside the arena (signed distance), to be minimized;
- **speed** φ_vel — time-averaged |v_i|, to be driven toward v_flock;
- **disconnected agents** N_disc and **minimum cluster size** N_min (from the communication graph, with the practical threshold N_min > N/5).
A characteristic clustering distance r_cluster = max( r_0^rep, r_0^frict + D̃(v_flock, a_frict, p_frict) ), where D̃(v,a,p) is the braking distance r at which D(r,a,p) = v, defines the communication graph used to form clusters.

## Code framework

The simulator already exists: a discrete-time stochastic integrator for the realistic-agent dynamics (Euler/Euler–Maruyama), a neighbor/communication-graph builder with range r_c and delay buffers, a terrain/wall description, and a generic per-agent "compute desired velocity" hook. The control law to be designed plugs into that hook. Below is the pre-method scaffold — the harness that is known to exist, with one empty slot where the distributed control law will go.

```matlab
% --- known: per-agent desired-velocity hook called every control step ---
% Inputs: this agent's state (pos, vel), the (delayed, noisy, local) states of
% its neighbors, distances/unit-vectors to them, and a terrain/wall description.
% Output: a desired velocity command handed to the low-level controller, which
% relaxes the real velocity toward it with time constant tau_CTRL, capped at a_max.

function [posDesired_i, velDesired_i, accDesired_i, control_mode_i] = ...
        generate_desire_i(id, state_i, states_neighbor, dis_to_neighbor, posid_to_neighbor, ...
                          terrain, terrain_params)

    params = load_params();          % the tunable knobs of the control law (TODO: define them)
    v_max  = params.v_max;           % known hard speed cap of the vehicle
    dim    = params.dim;

    pos2D_i = state_i(1:2);
    vel2D_i = state_i(4:5);

    % TODO: build the set of "wall/obstacle" virtual agents local to this agent
    %       from the terrain description (the confinement mechanism).
    pos_shill = [];  vel_shill = [];   % to be filled by the method

    % TODO: a term that drives the agent's speed toward the target flocking speed.
    v_selfprop = zeros(dim,1);

    % TODO: a short-range term that prevents collisions between agents.
    v_rep = zeros(dim,1);

    % TODO: a velocity-coordination term that damps delay-induced oscillation
    %       AND keeps closing speed within what the gap can brake — the crux.
    v_align = zeros(dim,1);

    % TODO: a wall/obstacle term that keeps the flock inside the arena.
    v_wall = zeros(dim,1);

    % --- known: superpose the contributions and cap the magnitude at v_max ---
    v_desired_2D = v_selfprop + v_rep + v_align + v_wall;        % (filled by method)
    s = norm(v_desired_2D);
    if s > v_max
        v_desired_2D = v_desired_2D / s * v_max;                 % keep direction, cap speed
    end
    velDesired_i = [v_desired_2D; 0; 0];
end

% --- a generic smooth speed-vs-distance shaping helper the method will need ---
function D = brake_curve(r, a, p)        % TODO: choose its exact piecewise form
    D = zeros(size(r));                  % (the kinematically-motivated curve)
end

% --- the tuner: a derivative-free optimizer over the control-law knobs against
%     a single scalar fitness built from the order parameters above ---
function theta_star = tune(theta0)
    % TODO: define fitness F(theta) from simulated order parameters
    %       (velocity correlation, collision risk, wall collisions, speed, clusters),
    %       then maximize it with a black-box evolution strategy.
    theta_star = theta0;
end
```
