# Vasarhelyi flocking model

## Problem

Make a swarm of real autonomous quadcopters flock — coherent, collision-free, confined to a bounded arena, scalable to high flocking speeds — using only on-board sensing and short-range radio. Idealized flocking rules (Reynolds, Vicsek, Couzin, potential-field/consensus) fail on hardware: communication delay, inertia and limited acceleration, a speed cap, sensor noise, and finite communication range turn the velocity-alignment rule into a delayed negative-feedback loop that self-oscillates and, at speed, collides — because braking requires distance and acceleration is bounded.

## Key idea

Build the per-agent desired velocity as a sum of physically motivated terms, with the alignment term redesigned around the acceleration limit. Since the maximum closing speed that can still be braked to zero across a gap r with deceleration a is v = √(2 a r), the velocity-alignment ("friction") term fires only when relative speed exceeds this kinematic envelope, then commands a correction opposite the excess. This makes one damper both kill delay-induced oscillation and keep closing speeds tied to braking distance. Confinement and obstacle avoidance reuse the same braking-gated form against virtual "shill" wall agents. The eleven resulting parameters are tuned by CMA-ES against a single conjunctive fitness built from flocking order parameters.

## The model

Per agent i, the momentary desired velocity is the vector sum of four terms, magnitude-capped at v_max.

**Repulsion (linear half-spring, noise-robust).** For each neighbor within r_0^rep:
v_ij^rep = p^rep (r_0^rep − r_ij) (r_i − r_j)/r_ij  if r_ij < r_0^rep, else 0;  v_i^rep = Σ_{j≠i} v_ij^rep.
Linear (not Lennard-Jones) so noisy position estimates near contact don't inject singular accelerations.

**Ideal braking curve.** Smooth velocity-vs-distance bound, C¹-continuous at the crossover r = a/p²:
D(r,a,p) = 0 (r ≤ 0); = r·p (0 < r·p < a/p, i.e. r < a/p²); = √(2 a r − a²/p²) (otherwise).
Linear constant-gain p near contact (finite gain), constant-deceleration √-branch at range (encodes a_max).

**Velocity alignment / friction.** Distance-dependent tolerated velocity difference:
v_ij^frictmax = max( v^frict , D(r_ij − r_0^frict, a^frict, p^frict) ),
Δv_ij^frict = −C^frict (v_ij − v_ij^frictmax) (v_i − v_j)/v_ij  if v_ij > v_ij^frictmax, else 0;  v_i^frict = Σ Δv_ij^frict.
It fires only above the braking-feasible envelope, so it damps excessive relative motion while leaving the flock free to turn within the envelope. The floor v^frict allows a constant velocity slack. Locality is automatic: the interaction range is where D(·) = 2 v^max.

**Walls and obstacles (shill agents).** Virtual shill agents on arena edges move inward at speed v^shill (obstacle shills move outward, one at the closest point). The real agent relaxes to the shill velocity with the same braking-gated form, but **no velocity slack and gain fixed to 1**:
v_is^shillmax = D(r_is − r_0^shill, a^shill, p^shill),
Δv_is^wall = −(v_is − v_is^shillmax)(v_i − v_s)/v_is  if v_is > v_is^shillmax, else 0.
Replaces fragile long-range cohesion with a soft repulsive box.

**Self-propulsion and cap.**
ṽ_i^d = (v_i/|v_i|) v^flock + v_i^rep + v_i^frict + Σ_s Δv_is^wall + Σ_s Δv_is^obstacle,
v_i^d = (ṽ_i^d/|ṽ_i^d|) · min(|ṽ_i^d|, v^max).

**Realistic agent dynamics** (what v_i^d is fed into):
a_i = η_i + [(v_i^d − v_i − v_i^s)/|v_i^d − v_i − v_i^s|] · min{ |v_i^d − v_i − v_i^s|/τ_CTRL , a_max },
with communication delay t_del, inner sensor noise v_i^s (SD σ_s), outer noise η_i (SD σ), sensor refresh t_s, communication range r_c. Typical quadcopter values: τ_CTRL ≈ 1 s, a_max = 6 m s⁻².

## Tuning

Eleven parameters { r_0^rep, p^rep, r_0^frict, C^frict, v^frict, p^frict, a^frict, r_0^shill, v^shill, p^shill, a^shill } map nonlinearly and noisily to behavior, so they are optimized rather than hand-tuned.

**Order parameters** (from a stochastic simulation): velocity correlation φ^corr ∈ [−1,1] (maximize); collision risk φ^coll = time-average of the pair indicator that is one when r_ij < r_coll, with r_coll = 3 m (minimize); wall collisions φ^wall (minimize); speed φ^vel → v^flock; disconnected agents N^disc and minimum cluster size N^min (N^min > N/5). Cluster distance r^cluster = max(r_0^rep, r_0^frict + D̃(v^flock, a^frict, p^frict)), D̃ = braking distance.

**Single-objective fitness** (multiplicative — any bad component tanks the score), with each parameter mapped to [0,1] by a transfer function:
F1(φ,φ0,d) = 1 − S(φ,φ0,d) (monotone, raised-cosine ramp S); F2(φ,s) = exp(−φ²/s²) (Gaussian peak at 0); F3(φ,a) = a²/(φ+a)² (sharp soft peak at 0 — used for collisions so the optimizer keeps a gradient out of bad regions instead of a flat 0/1 cliff).
F = F^speed · F^coll · F^disc · F^cluster · F^wall · F^corr,
F^speed = F1(φ^vel, v^flock, v^tol), F^coll = F3(φ^coll, a^tol), F^disc = F3(N^disc, N/5),
F^cluster = F1(N^min, N/5, N/5), F^wall = F2(φ^wall, r^tol), F^corr = Θ(φ^corr)·φ^corr.
Tolerances e.g. v^tol = (1.5/4) v^flock, a^tol = 0.00003, r^tol = 2 m.

**Optimizer:** CMA-ES (derivative-free, noise- and multimodality-tolerant), population ~100, ~150 generations (~15000 evals, each a multi-minute stochastic flight), parameters initialized mid-range with initial SD ≈ 1/6 of the range. Re-run per target speed (e.g. v^flock = 4, 6, 8 m/s with v^max = 6, 8, 10).

## Code

```matlab
function D = Dfunction(r, a, p)
    % ideal braking curve: linear (gain p) near 0, constant-deceleration sqrt-branch at range; C^1 at r = a/p^2
    D = r*0;
    temp       = r < a/p/p;
    condition1 = r > 0 & temp;          % 0 < r < a/p^2  -> r*p
    condition2 = ~temp;                 % r >= a/p^2     -> sqrt(2ar - a^2/p^2)
    D(condition1) = r(condition1) * p;
    D(condition2) = sqrt(2*a*r(condition2) - a*a/(p*p));
end
```

```matlab
function [posDesired_id, velDesired_id, accDesired_id, control_mode_id] = ...
        Vasarhelyi_module_generate_desire_i(id, state_i, states_neighbor, ...
                                            dis_to_neighbor, posid_to_neighbor, terrain, terrain_params)
    file_name_param = 'Vasarhelyi_module_parameters';
    [~, str_core] = get_multi_core_value();
    fun_params = str2func([file_name_param, str_core]);

    % 11 tuned control-law knobs + vehicle/world constants
    [r_com, v_flock, r_rep_0, p_rep, r_frict_0, c_frict, v_frict, p_frict, a_frict, ...
     r_shill_0, v_shill, p_shill, a_shill, v_max, dim, height, dr_shill, ...
     pos_shill, vel_shill] = fun_params();

    VELOCITY_HORIZONTAL_CONTROL_TYPE = 7;
    posDesired_id   = [state_i(1:2); height; 0];
    velDesired_id   = zeros(4,1);  accDesired_id = zeros(4,1);
    control_mode_id = VELOCITY_HORIZONTAL_CONTROL_TYPE;
    pos2DId        = state_i(1:2);   vel2DId = state_i(4:5);
    vel2D_neighbor = states_neighbor(4:5,:);

    % build obstacle shill agents from local terrain (each pushes the agent away from the obstacle)
    if ~isempty(terrain)
        r_w = 5;
        r_sub = floor((pos2DId(2)-terrain_params(2,1))/terrain_params(2,2));
        c_sub = floor((pos2DId(1)-terrain_params(1,1))/terrain_params(1,2));
        h_sub = floor(r_w/terrain_params(2,2));  w_sub = floor(r_w/terrain_params(1,2));
        [h,w] = size(terrain);
        r_min = max(1,r_sub-h_sub); r_max = min(h,r_sub+h_sub);
        c_min = max(1,c_sub-w_sub); c_max = min(w,c_sub+w_sub);
        terrain_sub = terrain(r_min:r_max,c_min:c_max);
        [r_obs,c_obs] = find(terrain_sub > state_i(3));
        if ~isempty(r_obs)
            r_obs = r_obs + r_min - 1;  c_obs = c_obs + c_min - 1;
            temp_p_shill = [(c_obs'*terrain_params(1,2))+terrain_params(1,1);
                            (r_obs'*terrain_params(2,2))+terrain_params(2,1)];
            temp = pos2DId - temp_p_shill;
            vel_shill = [vel_shill, temp./vecnorm(temp)];
            pos_shill = [pos_shill, temp_p_shill];
        end
    end

    % self-propulsion toward v_flock along current heading
    velIdNorm = norm(vel2DId);
    if velIdNorm == 0
        vr = rand(dim,1); vr = vr/norm(vr); vFlockId = v_flock * vr;
    else
        vFlockId = v_flock * vel2DId/velIdNorm;
    end

    vRepId = zeros(2,1); vFrictId = zeros(2,1);
    if ~isempty(dis_to_neighbor)
        % linear half-spring repulsion
        inRep = find(dis_to_neighbor < r_rep_0);
        if ~isempty(inRep)
            d = repmat(dis_to_neighbor(inRep), dim, 1);
            vRepId = p_rep * sum((r_rep_0 - d) .* posid_to_neighbor(:,inRep)./d, 2);
        end
        % braking-gated velocity alignment
        vijFrictMax = max(v_frict, Dfunction(dis_to_neighbor - r_frict_0, a_frict, p_frict));
        velij = repmat(vel2DId,1,length(dis_to_neighbor)) - vel2D_neighbor;
        vij   = sqrt(sum(velij.^2,1));
        inFr  = find(vij > vijFrictMax);
        if ~isempty(inFr)
            vN = repmat(vij(inFr),dim,1); vM = repmat(vijFrictMax(inFr),dim,1);
            vFrictId = -c_frict * sum((vN - vM).*velij(:,inFr)./vN, 2);
        end
    end

    % walls/obstacles: braking-gated form vs shill agents, no slack, gain 1
    vShillId = zeros(dim,1);
    posis = repmat(pos2DId,1,size(pos_shill,2)) - pos_shill;
    disis = sqrt(sum(posis.^2,1));
    inS   = find(disis < r_com); disisIn = disis(inS);
    visFrictMax = Dfunction(disisIn - r_shill_0, a_shill, p_shill);
    velis = repmat(vel2DId,1,length(disisIn)) - v_shill * vel_shill(:,inS);
    vis   = sqrt(sum(velis.^2,1));
    inFrS = find(vis > visFrictMax);
    if ~isempty(inFrS)
        vN = repmat(vis(inFrS),dim,1); vM = repmat(visFrictMax(inFrS),dim,1);
        vShillId = - sum((vN - vM).*velis(:,inFrS)./vN, 2);
    end

    % superpose and cap magnitude (keep direction) at v_max
    v2D = vFlockId + vRepId + vFrictId + vShillId;
    s = norm(v2D);
    if s > v_max,  v2D = v2D./s * v_max;  end
    velDesired_id(1:2) = v2D;
end
```

The implementation is a distributed desired-velocity controller: soft repulsion handles crowding, braking-gated alignment damps unsafe relative motion, shill-agent corrections handle walls and obstacles, self-propulsion sets the cruise speed, and CMA-ES tunes the coupled parameters through the conjunctive fitness.
