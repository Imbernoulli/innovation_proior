## Research question

An indoor service robot has to drive toward a goal through a space full of people and furniture that move and appear without warning. It carries proximity sensors (sonar, later laser) that see only a small patch of the world around it, and it has to decide on a new steering command several times a second — in the working system, every 0.25 s — using just that local snapshot. The robot is not a point: it has a maximum speed, and, more importantly, its motors can only deliver bounded torque, so its translational and rotational accelerations are bounded. At the speeds we want (well under a meter per second up to nearly a meter per second), this matters — the robot physically cannot make an arbitrarily sharp turn on demand, because braking and turning take time and distance it may not have.

The precise problem: given the robot's current translational velocity v and rotational velocity ω, its current pose, and the obstacles currently visible nearby, choose the next motion command so that (1) the robot makes progress toward the goal, (2) it never commands a motion it cannot physically execute given its acceleration limits, and (3) it never drives into an obstacle — and do all of this within a single control cycle, repeatedly, without a global map. A solution has to respect the robot's *dynamics*, not just its kinematics: a command is only meaningful if the robot can actually reach that velocity in the next tick and can still stop before hitting something.

## Background

The field splits collision avoidance into two camps. **Global** methods (road-map, cell-decomposition, potential-field navigation functions) assume a complete, accurate, static model of the environment and compute a full path off-line. They are strong at global path planning but inappropriate for fast reactive avoidance: motion planning is inherently expensive, so when the world changes on the fly the global plan must be recomputed repeatedly, which is too slow; and they fail when the world model is inaccurate or simply unavailable, as it usually is in a populated indoor space. **Local** (reactive) methods use only a small slice of the world to generate a command. They cannot produce optimal global solutions and can get trapped in local minima (a U-shaped obstacle, a symmetric configuration), but their low computational cost is exactly what lets them run inside a tight sensor-driven control loop.

Several load-bearing facts about the robot and about prior local methods set up the problem:

- **Synchro-drive kinematics.** On a synchro-drive base the translational velocity v always points along the robot's heading θ — a non-holonomic constraint. The pose evolves by integrating velocity: ẋ = v·cos θ, ẏ = v·sin θ, with θ driven by the rotational velocity ω. A robot moving at constant (v, ω) traces a circle of radius v/ω; v with ω = 0 is a straight line. So short trajectory segments are naturally **circular arcs**, and a circle-versus-obstacle intersection is cheap to test.

- **Bounded actuators.** For most mobile robots the accelerations are monotonic functions of the motor currents, and digital hardware can change those currents only at discrete control ticks. So the robot's translational and rotational accelerations are bounded and piecewise-constant between ticks. This is the physical fact that "ignore the dynamics" methods violate.

- **Braking is a distance.** A robot moving at speed v that decelerates at a constant rate a needs distance v²/(2a) to stop. Whether a velocity is *safe* therefore depends on how far the nearest obstacle is along the path the robot would take.

- **Potential-field local minima.** It is well documented that artificial-potential-field avoidance, which follows the negated gradient of an attractive (goal) plus repulsive (obstacle) potential, stalls at points where the gradient vanishes away from the goal — U-shaped obstacle configurations and symmetric layouts. Borenstein and Koren further observed that such methods fail to find a path between closely spaced obstacles and produce oscillatory behavior in narrow corridors. These are pre-existing, diagnosed failure modes of the design space, not measurements of any new method.

- **The "infinite force" assumption.** Most local methods generate a command in two stages: first pick a desired motion *direction* from the sensor data, then convert that direction into a steering command. This is strictly justifiable only if arbitrarily large forces can be applied to the robot — i.e. if it can instantly accelerate in the chosen direction. For a robot with bounded acceleration the desired direction may be physically unreachable in the available time, and following it can drive the robot into the obstacle it was trying to avoid.

## Baselines

**Artificial potential fields (Khatib, 1986).** Obstacles assert a repulsive force, the goal an attractive force, and the robot follows the negated gradient of the summed potential U_att + U_rep. Extremely fast, uses only nearby obstacles, and naturally produces smooth motion. Two gaps: it operates in force/position space and assumes the robot can instantly accelerate along the resulting force, ignoring the robot's dynamics; and it is prone to local minima where the net force vanishes before the goal (U-shapes, symmetric obstacles), where the robot stalls or oscillates (Borenstein & Koren).

**Two-stage histogram methods — virtual force field and vector field histogram (Borenstein & Koren; Ulrich & Borenstein).** Build an occupancy grid / polar histogram of the free space from proximity sensors, pick a desired travel direction from the histogram, then turn that direction into a steering command. Fast and adaptive to unforeseen changes. Gap: the direction-then-steer split is again only valid under infinite forces; it does not check whether the commanded direction is reachable given the robot's acceleration limits, so at speed it can command turns the robot cannot execute.

**Curvature-velocity method (Simmons, 1996).** Formulates local obstacle avoidance as a *constrained optimization in velocity space*: the decision variables are the translational and rotational velocities (v, ω); constraints come from physical limits (maximum velocities and accelerations) and from the obstacle configuration; the robot picks the (v, ω) that satisfies all constraints and maximizes an objective trading off speed, safety, and goal-directedness. This is the decisive move — reasoning about commands directly in velocity space, where the trajectories are circular arcs (constant curvature), so the dynamics can be expressed as constraints on (v, ω). Gap it leaves open: the way the acceleration limits enter the constraint set, and the way obstacle safety is enforced, are not derived directly from the synchro-drive motion equations — there is room for a cleaner, dynamics-derived restriction of the reachable velocities and a sharper admissibility (braking) condition.

**Pure pursuit (path tracking).** Given a path to follow, pick a lookahead point on it and command the circular arc that reaches that point, recomputing as the robot advances. It confirms that arcs are the right control primitive but tracks a *given* path: it does no obstacle reasoning and offers no guarantee that the commanded arc is dynamically feasible.

## Evaluation settings

The natural testbed is a real synchro-drive indoor robot (a B21-class platform) driving to goal locations in cluttered, populated office and exhibition environments, sensing obstacles with rings of ultrasonic proximity sensors (and, alternatively, cameras or infrared) from which a local world model — an obstacle line field / set of nearby obstacle points — is built. The control loop runs at a fixed rate (about four cycles per second). The quantities a method would be judged on are whether it avoids collisions reliably, the speeds it can sustain while doing so, and the smoothness of the resulting motion through corridors, doorways, and around moving people. The robot's pose is re-measured by wheel encoders several times a second, so internal trajectory prediction need only be accurate over one control interval.

## Code framework

What already exists: a robot state `[x, y, yaw, v, omega]`; a motion model that steps the pose forward under a commanded `(v, omega)` for a small time `dt` along the resulting circular arc; the robot's hardware limits (max/min translational speed, max rotational speed, max translational and rotational accelerations) and loop period; and a local obstacle list from the proximity sensors. The control loop calls a planner each tick to turn the current state, the goal, and the obstacles into the next `(v, omega)`. The planner itself — how it restricts which velocities are even worth considering, and how it scores them — is the empty slot.

```python
import math
import numpy as np


class Config:
    def __init__(self):
        self.max_speed = 1.0          # [m/s]
        self.min_speed = -0.5         # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0   # [rad/s]
        self.max_accel = 0.2          # [m/s^2]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s^2]
        self.dt = 0.1                 # [s] control / prediction tick
        self.predict_time = 3.0       # [s] how far ahead to roll an arc
        # weights for the score terms (to be designed)
        self.v_resolution = 0.01
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0
        self.robot_radius = 1.0       # [m] for collision check


def motion(x, u, dt):
    """Step the pose forward along the (v, omega) arc for dt."""
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x


def predict_trajectory(x_init, v, w, config):
    """Forward-simulate the constant-(v, omega) arc over predict_time."""
    x = np.array(x_init)
    traj = np.array(x)
    t = 0.0
    while t <= config.predict_time:
        x = motion(x, [v, w], config.dt)
        traj = np.vstack((traj, x))
        t += config.dt
    return traj


def reachable_velocity_set(x, config):
    # TODO: which (v, omega) are even worth considering this tick?
    #       (hardware limits, what's reachable under the accel limits,
    #        what is safe given the obstacles)
    pass


def score(trajectory, goal, obstacles, config):
    # TODO: how good is the arc produced by one candidate (v, omega)?
    pass


def plan(x, config, goal, obstacles):
    # TODO: pick the best (v, omega) from the reachable set by the score
    pass


def main(goal):
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    config = Config()
    obstacles = ...  # local obstacle points from the proximity sensors
    while True:
        u = plan(x, config, goal, obstacles)
        x = motion(x, u, config.dt)
        if math.hypot(x[0] - goal[0], x[1] - goal[1]) <= config.robot_radius:
            break
```
