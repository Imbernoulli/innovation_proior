# The Dynamic Window Approach (DWA)

## Problem

A mobile robot with bounded velocity and bounded acceleration must drive toward a goal in a cluttered, dynamic environment, choosing a new motion command every control tick from only local proximity-sensor data — and must never command a motion it cannot physically execute or one from which it cannot stop in time. Methods that pick a desired *direction* and then steer toward it implicitly assume infinite forces; at speed, the robot's bounded acceleration makes the desired turn unreachable and it can collide.

## Key idea

Search directly in **velocity space** (v, ω) — translational and rotational velocity — instead of in direction space, because that is where the robot's acceleration limits become a literal, bounded region. A constant (v, ω) traces a circular arc of radius v/ω, so each candidate velocity is one cheap-to-test trajectory. Restrict the candidates to velocities that are simultaneously (a) physically allowed by the hardware, (b) *admissible* — the robot can brake to a halt before the nearest obstacle on the arc, and (c) inside the **dynamic window** — reachable within one control tick given the acceleration limits. Over that small 2-D set, pick the velocity maximizing an objective that trades off heading toward the goal, clearance from obstacles, and forward speed. Execute one tick, re-sense, repeat.

## Method

Synchro-drive motion: v points along heading θ, so ẋ = v cos θ, ẏ = v sin θ; a constant (v, ω) gives a circle of radius v/ω about center (−(v/ω) sin θ, (v/ω) cos θ). The trajectory over a short horizon is a sequence of such arcs.

Each cycle the search space is reduced to **V_r = V_s ∩ V_a ∩ V_d**:

- **V_s** (possible): v ∈ [0, v_max] (optionally slightly negative), ω ∈ [−ω_max, ω_max].
- **V_a** (admissible / braking-safe): with dist(v, ω) the distance to the nearest obstacle along the arc and v̇_b, ω̇_b the braking decelerations, the stopping distance v²/(2 v̇_b) must fit in the clearance:

  V_a = { (v, ω) : v ≤ √(2·dist(v,ω)·v̇_b)  ∧  ω ≤ √(2·dist(v,ω)·ω̇_b) }.

- **V_d** (dynamic window): with current velocity (v_a, ω_a), tick t, accelerations v̇, ω̇:

  V_d = { (v, ω) : v ∈ [v_a − v̇·t, v_a + v̇·t]  ∧  ω ∈ [ω_a − ω̇·t, ω_a + ω̇·t] }.

Over V_r, maximize the objective (each term normalized to [0, 1], σ a smoothing):

  **G(v, ω) = σ( α·heading(v, ω) + β·dist(v, ω) + γ·velocity(v, ω) )**

- **heading(v, ω)** — alignment with the goal direction (e.g. 1 − |θ|/π), evaluated at the pose predicted after applying (v, ω) and braking; maximal when pointed at the goal.
- **dist(v, ω)** — clearance to the nearest obstacle on the arc (capped at a constant if the arc is clear); small clearance scores low, so the robot rounds obstacles.
- **velocity(v, ω)** — forward speed v; rewards moving, breaking ties away from standing still.

The smoothing σ moves the chosen optimum into a broad safe basin rather than onto a velocity one grid cell from grazing an obstacle, giving extra side-clearance. Because the dynamic window is part of the search space, every commanded velocity is reachable this tick and stoppable in time — feasibility holds by construction.

## Code

A clean implementation, gridding the window, forward-rolling each candidate arc, and scoring as a cost to minimize (the negative of G, with collision rejected by an infinite cost — the admissibility filter folded in):

```python
import math
import numpy as np


class Config:
    def __init__(self):
        self.max_speed = 1.0                              # [m/s]
        self.min_speed = -0.5                             # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0        # [rad/s]
        self.max_accel = 0.2                              # [m/s^2]   width of V_d
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s^2]
        self.v_resolution = 0.01
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0
        self.dt = 0.1                                     # control / prediction tick
        self.predict_time = 3.0                           # arc rollout horizon
        self.to_goal_cost_gain = 0.15                     # alpha (heading)
        self.speed_cost_gain = 1.0                        # gamma (velocity)
        self.obstacle_cost_gain = 1.0                     # beta  (clearance)
        self.robot_stuck_flag_cons = 0.001
        self.robot_radius = 1.0                           # [m]


def motion(x, u, dt):
    # synchro-drive arc step
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x


def calc_dynamic_window(x, config):
    # V_s (hardware) ∩ V_d (reachable this tick under accel limits)
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]
    return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]


def predict_trajectory(x_init, v, w, config):
    x = np.array(x_init)
    traj = np.array(x)
    t = 0.0
    while t <= config.predict_time:
        x = motion(x, [v, w], config.dt)
        traj = np.vstack((traj, x))
        t += config.dt
    return traj


def calc_to_goal_cost(trajectory, goal):
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    return abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))


def calc_obstacle_cost(trajectory, ob, config):
    ox, oy = ob[:, 0], ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)
    if np.any(r <= config.robot_radius):
        return float("inf")            # collision -> inadmissible
    return 1.0 / np.min(r)             # closer obstacle -> higher cost


def calc_control_and_trajectory(x, dw, config, goal, ob):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_traj = np.array([x])
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for w in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            traj = predict_trajectory(x_init, v, w, config)
            to_goal = config.to_goal_cost_gain * calc_to_goal_cost(traj, goal)
            speed   = config.speed_cost_gain * (config.max_speed - traj[-1, 3])
            obst    = config.obstacle_cost_gain * calc_obstacle_cost(traj, ob, config)
            cost = to_goal + speed + obst
            if cost <= min_cost:
                min_cost = cost
                best_u = [v, w]
                best_traj = traj
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    best_u[1] = -config.max_delta_yaw_rate   # anti-stall nudge
    return best_u, best_traj


def dwa_control(x, config, goal, ob):
    dw = calc_dynamic_window(x, config)
    return calc_control_and_trajectory(x, dw, config, goal, ob)


def main(gx=10.0, gy=10.0):
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])  # [x, y, yaw, v, omega]
    goal = np.array([gx, gy])
    config = Config()
    ob = config_obstacles()  # local obstacle points from the proximity sensors
    while True:
        u, _ = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)
        if math.hypot(x[0] - goal[0], x[1] - goal[1]) <= config.robot_radius:
            break
```

Per tick: build the dynamic window (V_s ∩ V_d), grid it, roll out each candidate arc, reject collisions and score by heading + speed + clearance, take the best, drive one tick, re-sense. The acceleration limits live inside the window, so every command is dynamically feasible and stoppable by construction — the property the direction-first methods cannot guarantee.
