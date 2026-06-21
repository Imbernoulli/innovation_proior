The problem is reactive collision avoidance for a synchro-drive mobile robot that must reach a goal using only local proximity-sensor data, without a global map. The robot has bounded translational and rotational accelerations, so it cannot instantly change either its speed or its heading. Existing local methods fall into a common trap: they first select a desired travel direction from the sensor data and only afterwards convert that direction into a steering command. That two-stage split is only valid if the robot can apply arbitrarily large forces. At the speeds we want, the desired heading may lie outside what the motors can reach in one control tick, so the robot either misses the maneuver or, worse, carries its momentum into the obstacle it was trying to avoid. Potential fields add the separate problem of local minima, where attractive and repulsive forces cancel away from the goal and the robot stalls or oscillates. What is missing is a planner that reasons about what the robot can actually do and stop from, not just where it should point.

The right space to search is velocity space, not direction space. The motors set accelerations, so the natural decision variables are the translational velocity v and the rotational velocity ω. Under a constant (v, ω) a synchro-drive robot traces a circular arc of radius |v/ω|, or a straight line when ω is zero. That means every candidate command is one cheap-to-test arc. The constraints also become simple geometric statements in the (v, ω) plane: hardware limits form a box; the braking requirement that the robot stop before hitting the nearest obstacle becomes the condition |v| ≤ √(2·dist(v,ω)·v̇_b); and the one-tick reachability under bounded acceleration becomes the dynamic window centered on the current velocity. The legal commands are exactly the intersection of these three sets, and we simply search that small two-dimensional region.

The method is the Dynamic Window Approach, or DWA. Each control cycle it forms the dynamic window V_d of velocities reachable from the current (v, ω) in one tick, intersects it with the hardware box V_s and the braking-safe set V_a, grids the resulting region, forward-rolls each candidate arc, and picks the velocity that best trades off three objectives. The heading term rewards arcs whose final heading points toward the goal. The clearance term penalizes arcs that pass close to obstacles. The velocity term rewards forward speed so the robot does not simply stand still. A small amount of smoothing over the grid keeps the chosen velocity away from the boundary of the safe set. Because acceleration limits live inside the dynamic window itself, every command DWA evaluates is one the robot can physically reach next tick; because the obstacle cost rejects collisions and the braking set guarantees a stop before impact, the chosen command is also safe.

DWA therefore replaces the direction-first pipeline with a direct search over dynamically feasible arcs. It runs fast enough to live inside a 0.1 s control loop, needs no global map, and naturally re-plans each tick as new sensor data arrives. The code below implements the core loop: `calc_dynamic_window` builds V_s ∩ V_d, `predict_trajectory` rolls out the constant-velocity arc, `calc_obstacle_cost` rejects collisions and scores clearance, and `calc_control_and_trajectory` grids the window and returns the best (v, ω).

```python
import math
from enum import Enum

import numpy as np


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    def __init__(self):
        self.max_speed = 1.0
        self.min_speed = -0.5
        self.max_yaw_rate = 40.0 * math.pi / 180.0
        self.max_accel = 0.2
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0
        self.v_resolution = 0.01
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0
        self.dt = 0.1
        self.predict_time = 3.0
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001
        self.robot_type = RobotType.circle
        self.robot_radius = 1.0
        self.robot_width = 0.5
        self.robot_length = 1.2
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]])


def motion(x, u, dt):
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x


def calc_dynamic_window(x, config):
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
    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("inf")
    return 1.0 / np.min(r)


def calc_control_and_trajectory(x, dw, config, goal, ob):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_traj = np.array([x])
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for w in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            traj = predict_trajectory(x_init, v, w, config)
            to_goal = config.to_goal_cost_gain * calc_to_goal_cost(traj, goal)
            speed = config.speed_cost_gain * (config.max_speed - traj[-1, 3])
            obst = config.obstacle_cost_gain * calc_obstacle_cost(traj, ob, config)
            cost = to_goal + speed + obst
            if cost <= min_cost:
                min_cost = cost
                best_u = [v, w]
                best_traj = traj
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_traj


def dwa_control(x, config, goal, ob):
    dw = calc_dynamic_window(x, config)
    return calc_control_and_trajectory(x, dw, config, goal, ob)


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    goal = np.array([gx, gy])
    config = Config()
    config.robot_type = robot_type
    ob = config.ob
    while True:
        u, _ = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)
        if math.hypot(x[0] - goal[0], x[1] - goal[1]) <= config.robot_radius:
            break
```
