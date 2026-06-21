## Research question

How do you make a *large group of real autonomous flying robots* — quadcopters, each running only on-board sensing and computation, exchanging only local radio messages — move together as a coherent flock that is collision-free, stays confined to a bounded arena (and avoids obstacles), and remains stable across a wide range of flocking speeds and flock sizes?

The motivation is concrete and physical. Decades of statistical-physics and computer-graphics work show that simple decentralized rules reproduce the rich collective motion of bird flocks and fish schools in *simulation*. A real quadcopter is not an idealized self-propelled point: it receives neighbor positions and velocities with a non-negligible communication delay; it cannot change its velocity instantly because of inertia and a finite maximum acceleration, mediated by a low-level velocity controller with its own relaxation time; it has a hard speed cap; its on-board sensors return noisy relative positions and velocities at a finite refresh rate; and its radio has a finite communication range so it sees only nearby neighbors, with outages that grow with distance. The question is how to design a distributed control law that produces coherent, confined flocking while accounting for these physical realities.

## Background

**Reynolds' three rules (boids, 1987).** Reynolds modeled flocking as three local steering behaviors per agent: *separation* (steer to avoid crowding nearby flockmates — short-range repulsion), *alignment* (steer toward the average heading/velocity of local neighbors), and *cohesion* (steer toward the average position of local neighbors). It is a purely geometric/kinematic recipe — there is no agent dynamics, no delay, no acceleration limit, and the alignment steering is unbounded. It produces convincing flocks on a screen precisely because the simulated agents have no inertia and perfect, instantaneous knowledge of neighbors.

**Vicsek's self-propelled particles (1995).** The minimal physics model: N point particles move at *constant* speed v; at each discrete step each particle sets its heading to the average heading of all neighbors within radius R, plus an angular noise η: θ_i(t+1) = ⟨θ_j⟩_{|r_i−r_j|<R} + η, with r_i(t+1) = r_i(t) + v Δt (cosθ_i, sinθ_i). The order parameter φ = |Σ_j v_j| / (N v) measures global alignment, and the model exhibits a noise-/density-driven order–disorder phase transition. This pinned down *alignment* as the essential ingredient of collective order — but the model has constant speed, no collision avoidance, no inertia or delay, and periodic boundaries; it is a statistical-mechanics object, not a controller.

**Couzin's zonal model (2002).** Generalizes the rules into three concentric zones around each agent — an inner *zone of repulsion*, a middle *zone of orientation* (alignment), and an outer *zone of attraction* — and shows how group structure (swarm, torus, parallel/dynamic groups) emerges as the zone sizes change.

**Potential-field / consensus flocking (Olfati-Saber, 2006).** Brings flocking into control theory with provable guarantees. Agents are double integrators; a collective potential penalizes deviation from an "α-lattice" (a target inter-agent spacing), and the control law is the gradient of that potential plus a *velocity-consensus* term that drives neighbor velocities to agreement, plus a navigational feedback term toward a goal. Obstacles are handled by virtual "β-agents" on obstacle surfaces and a group objective by a virtual "γ-agent." This gives Lyapunov stability and flock cohesion for ideal double-integrator dynamics.

**The realistic-agent model and the friction-like alignment term (Virágh, Vásárhelyi et al., 2014).** This model makes the physical constraints of a real flying robot first-class citizens. It defines an agent whose desired velocity v_i^d is a function of *delayed, noisy, local* neighbor positions and velocities, and whose actual acceleration obeys
a_i(t) = η_i(t) + [ (v_i^d − v_i − v_i^s) / |v_i^d − v_i − v_i^s| ] · min{ |v_i^d − v_i − v_i^s| / τ_CTRL , a_max },
i.e. the real velocity relaxes toward the desired velocity with characteristic time τ_CTRL but is capped at acceleration a_max, with inner sensor noise v_i^s, outer (wind) noise η_i, finite sensor refresh t_s, time delay t_del, and communication range r_c. Quadcopter values: τ_CTRL ≈ 1 s, a_max = 6 m s⁻², σ_s = 0.005 m² s⁻², t_s = 0.2 s, t_del = 0–2 s, r_c = 30–140 m, and desired speeds saturated at v_max = 4 m s⁻¹. It damps alignment oscillations with a **viscous-friction-like alignment term**: a pairwise term that pulls neighbor velocities together, of the form
v_ij^frict = C_frict · (v_j − v_i) / (max{r_min, |d_ij|})²,
which relaxes velocity differences, remains local, and stays upper-bounded as inter-agent distance goes to zero. Short-range repulsion uses a *linear half-spring* (soft linear repulsion is robust to noisy position measurements, unlike a stiff Lennard-Jones-type potential), and confinement uses virtual *shill* wall agents whose velocity the real agents align to. This model has been demonstrated on real quadcopters.

## Baselines

- **Reynolds boids (1987).** Separation + alignment + cohesion as geometric steering, implemented as pure kinematic rules with no agent dynamics or acceleration limit.
- **Vicsek SPP (1995).** Constant-speed heading alignment + noise; demonstrates the order–disorder transition with periodic boundaries.
- **Couzin zonal model (2002).** Repulsion/orientation/attraction zones; emergent group shapes depending on zone sizes.
- **Olfati-Saber potential-field flocking (2006).** Lattice-potential gradient + velocity consensus + navigation, obstacles/goal as virtual β/γ agents; provably stable for ideal double integrators.
- **Virágh/Vásárhelyi realistic model (2014).** Realistic agent (delay, inertia, a_max, noise, locality) + linear repulsion + 1/r² viscous-friction alignment + shill walls; demonstrated on real quadcopters.

## Evaluation settings

The natural yardstick is a stochastic simulation of the realistic-agent model in a square arena, plus real outdoor multi-drone flights. Simulation: a square arena of characteristic size L_arena (e.g. 250 m), up to 100s–1000s of agents, communication range r_c on the order of tens of meters, a constant communication delay (e.g. 1 s), and target flocking speeds spanning several m/s up to tens of m/s, each with a corresponding speed cap v_max. The dangerous-collision distance is r_coll = 3 m. Hardware: a swarm of quadcopters (e.g. 30 units), each with a Pixhawk autopilot and an on-board companion computer issuing desired-velocity commands at ~20 Hz, two parallel radios (a long-range low-bandwidth link and a short-range high-bandwidth link) broadcasting position/velocity status packets. Quality is judged by the *order parameters* below; the question is whether a control law exists that keeps them all good simultaneously at high speed.

Order parameters (the measurable requirements of "good flocking"), to be computed from a trajectory and used as optimization targets:
- **velocity correlation** φ_corr — cluster-internal time-average of v_i·v_j/(|v_i||v_j|), to be maximized;
- **collision risk** φ_coll — time-average over agent pairs of the indicator that is one when r_ij < r_coll, the fraction of time pairs inside the 3 m danger zone, to be minimized;
- **wall collisions** φ_wall — time-average of how far agents stray outside the arena (signed distance), to be minimized;
- **speed** φ_vel — time-averaged |v_i|, to be driven toward v_flock;
- **disconnected agents** N_disc and **minimum cluster size** N_min (from the communication graph, with the practical threshold N_min > N/5).
Forming clusters requires a characteristic interaction distance over which two agents count as coupled; the control law must supply one.

## Code framework

The simulator already has a discrete-time stochastic integrator for the realistic-agent dynamics (Euler/Euler–Maruyama), a neighbor/communication-graph builder with range r_c and delay buffers, a terrain/wall description, and a generic per-agent "compute desired velocity" hook. The control law plugs into that hook.

```matlab
% Per-agent desired-velocity hook called every control step.
% Inputs: own state; delayed, noisy local neighbor states; distances and
% unit vectors to neighbors; terrain/wall description.
% Output: desired horizontal velocity command for the low-level controller.

function [posDesired_i, velDesired_i, accDesired_i, control_mode_i] = ...
        generate_desire_i(id, state_i, states_neighbor, dis_to_neighbor, posid_to_neighbor, ...
                          terrain, terrain_params)

    params = load_params();
    v_max  = params.v_max;
    height = params.height;

    posDesired_i   = [state_i(1:2); height; 0];
    velDesired_i   = zeros(4,1);
    accDesired_i   = zeros(4,1);
    control_mode_i = 7;

    % TODO: compute a distributed desired horizontal velocity from local
    %       neighbor states and terrain geometry.
    v_command_2D = zeros(2,1);

    s = norm(v_command_2D);
    if s > v_max
        v_command_2D = v_command_2D / s * v_max;
    end
    velDesired_i(1:2) = v_command_2D;
end
```
