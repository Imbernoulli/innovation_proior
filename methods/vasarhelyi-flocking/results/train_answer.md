The canonical name for this method is the **Vásárhelyi flocking model**. It is a distributed control law designed to make a large swarm of real autonomous quadcopters move as a coherent, collision-free flock inside a bounded arena, using only on-board sensing, local radio messages, and realistic actuator dynamics. I think of it as the point where statistical-physics flocking finally meets the constraints of physical hardware.

The motivating failure is straightforward but severe. In simulation, Reynolds' boids, Vicsek particles, and Couzin's zonal model all produce beautiful collective motion because their agents are idealized: they sense neighbors instantly, change velocity without inertia, and never hit acceleration limits. Drop the same rules onto a real quadcopter and they collapse. A quadcopter receives neighbor states over a radio packet that is already delayed by a second or two, its velocity is tracked by a low-level controller with a relaxation time on the order of one second, and it cannot accelerate harder than about six meters per second squared. The naive alignment rule therefore becomes a delayed negative-feedback loop: the agent steers to cancel a velocity difference that no longer exists, overshoots, and the overshoot rings through the flock. The amplitude of those self-excited oscillations grows with flocking speed, and once the closing speed between two agents exceeds what their capped acceleration can brake within the available gap, the oscillation becomes a collision. A working method has to respect delay, inertia, acceleration limits, sensor noise, communication range, and a hard speed cap from the start.

My design of the desired velocity for each agent has four physical terms and a final magnitude cap. First, short-range repulsion. I use a linear half-spring: if a neighbor is inside a cutoff radius, the agent gets a repulsive velocity command proportional to how far inside the cutoff the neighbor is, pointing directly away. The force is linear rather than, say, a Lennard-Jones potential, because under noisy position measurements a stiff potential would amplify sensor noise into huge, spurious accelerations. Linearity keeps the repulsion bounded and noise-robust.

Second, and most distinctive, is the braking-gated velocity-alignment term. The kinematic fact I build around is that the largest closing speed that can still be arrested across a gap r with maximum deceleration a is sqrt(2 a r). I encode this as a smooth, distance-dependent speed envelope D(r, a, p). Near contact the function is linear with gain p so that tiny distance errors do not create infinite slopes; beyond a crossover distance a over p squared it switches to the constant-deceleration square-root branch. The two branches meet with continuous value and continuous slope, so the threshold is stable to integrate. I then allow each pair of agents a tolerated relative speed equal to the maximum of a small constant velocity slack and the value of this braking curve evaluated at their separation minus a small safety offset. Only when the actual relative speed exceeds this envelope does the alignment term fire, and when it fires it commands a correction opposite the excess. Below the envelope the agents are left alone, which lets the flock turn and breathe; above the envelope the dangerous relative motion is damped. Locality is automatic because the maximum possible relative speed is bounded by twice the speed cap, so beyond the distance where the braking curve reaches that value there is nothing to correct.

Third, confinement and obstacle avoidance are handled by the same braking-gated form, but applied to virtual shill agents placed on the arena boundary and on obstacle surfaces. A wall shill sits just outside the arena and moves inward; an obstacle shill sits at the closest obstacle point and pushes outward. The real agent treats the shill like a neighbor, except that there is no velocity slack and the gain is fixed to one, because walls are hard constraints. This replaces the fragile long-range cohesion used in idealized flocking with a soft repulsive box that keeps the group inside the arena without creating another delay-sensitive attraction term.

Fourth, a self-propulsion term drives the agent along its current heading at the target flocking speed. This sets the cruise speed while the alignment term synchronizes headings. The four terms are summed vectorially and the resulting desired velocity is capped in magnitude at the vehicle's maximum speed, preserving direction. The real acceleration then follows a first-order relaxation toward that capped command with bounded acceleration, exactly as the hardware does.

The method has eleven coupled parameters, including repulsion cutoff and gain, friction offset and gain, braking-curve parameters, wall-shill parameters, and the shill speed. Their map to collective behavior is nonlinear, multimodal, and noisy, so I do not hand-tune them. Instead I optimize them with CMA-ES. The objective is a single conjunctive scalar fitness built from measurable order parameters: velocity correlation, collision risk, wall excursion, mean speed relative to the target, number of disconnected agents, and minimum cluster size. Each requirement is mapped smoothly into the unit interval, with a soft peak for quantities that should be near zero such as collision risk, so the optimizer retains a gradient even when the flock is performing badly. CMA-ES runs for roughly a hundred and fifty generations on a population of about a hundred individuals, each evaluation being a stochastic multi-agent flight, and the process is repeated for different target speeds because the friction law must scale with speed rather than being fixed once and for all.

The two pieces below are the actual onboard control-law code: the braking curve, and the per-agent desired-velocity generator called every control cycle by each vehicle (or by each simulated agent). The generator turns any nearby terrain higher than the vehicle's own altitude into virtual obstacle shills that push it away from the surface, forms the self-propulsion term along the current heading at $v_\text{flock}$, sums the linear half-spring repulsion and the braking-gated friction over whatever neighbors are within communication range, folds in the same braking-gated correction against every wall and obstacle shill — with the slack term dropped and the gain fixed to one — and finally caps the summed command's magnitude at $v_\text{max}$ before handing it back as the horizontal velocity set-point.

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

In short, the Vásárhelyi flocking model solves real-robot flocking by redesigning alignment around the hard kinematic fact that braking distance scales with the square of speed. Soft linear repulsion prevents crowding, the braking curve gates velocity synchronization so it damps only dangerous relative motion, virtual shill agents enforce confinement and obstacle avoidance, self-propulsion sets the cruise, and CMA-ES tunes the coupled parameters against a conjunctive fitness of order parameters. The result is a control law that remains coherent, bounded, and collision-free across the high speeds where earlier fixed-friction rules fail.
