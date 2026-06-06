OK, let me think this through from scratch. I have a robot in an office that's full of people and furniture that move around. It has a goal somewhere and a ring of sonar sensors that only see a few meters around it. Several times a second I have to hand the motors a new command and not crash. There's no reliable map — people walk in front of it — so anything that wants a complete static model of the world is out. I'm forced into a local, reactive loop: look at what's near me right now, pick a command, execute, look again.

So what command am I even choosing? The instinct from everything I've read is to choose a *direction*. Potential fields do this beautifully: pretend obstacles push the robot away and the goal pulls it, sum the forces, and the net force tells me which way to go. It's fast, it only uses nearby obstacles, the motion comes out smooth. The histogram methods do the same thing in a different dress — build a polar histogram of free space from the sonar, read off the most open direction that still points goal-ward, steer that way. In both cases the output is "go *this* way."

But hold on. Let me actually picture my robot in a corridor. It's moving fast, straight down the hall, and the goal is through a doorway off to the right. The potential field, or the histogram, says: desired direction is sharply to the right, turn now. Stage one of these methods picks that direction; stage two converts it into a steering command and sends it. And the robot... can't do it. Its motors have bounded torque. To swing the heading that far that fast it would need an angular acceleration it doesn't have. So either it executes a much gentler turn than commanded — and sails past the door — or, worse, the controller faithfully tries to follow the desired heading and the body, carrying its momentum, plows straight into the wall on the far side of the door. The method told it to do something physically impossible and then drove it into the obstacle it was trying to route around.

That's the crack. These methods are "strictly justifiable only if infinite forces can be asserted on the robot." Pick-a-direction-then-steer silently assumes the robot can instantly accelerate any way it likes. My robot can't. At low speed you get away with it; at the speeds I want, the bounded acceleration is the dominant fact, and a method that ignores it is unsafe precisely when it matters most. And separately, potential fields have the local-minimum disease — net force vanishes in a U-shaped obstacle or a symmetric layout and the robot just sits there, or oscillates in a narrow corridor — but that's a known problem; the dynamics blindness is the one nobody upstream is handling.

So the real constraint I keep tripping over is: *what can the robot actually do in the next instant?* And "direction" is the wrong variable to be reasoning in, because a direction hides the question of how fast I'm already going and how fast I can change it. Let me change variables. The thing the motors actually set, the thing that has acceleration limits attached to it, is **velocity** — the translational velocity v and the rotational velocity ω. If I reason in (v, ω) space instead of in direction space, then "bounded acceleration" becomes a clean, literal statement: from my current (v, ω), the velocities I can reach one tick from now lie within v̇·Δt and ω̇·Δt of where I am. That's not a vague "the turn is too sharp," it's a box I can draw. The dynamics stop being an afterthought and become a region in the space I'm searching.

This isn't a brand new idea — Simmons' curvature-velocity method already plants the flag here: treat avoidance as a constrained optimization over (v, ω), with constraints from the velocity and acceleration limits and from the obstacles, and an objective that trades off speed against safety against goal-directedness. That's the right frame. What I want to do is *derive* the whole thing — the reachable set and the safety condition — straight from the robot's motion equations, so the dynamics aren't bolted on as side constraints but fall out of the physics. Let me do that and see what shape the constraints actually take.

Start with the kinematics. My base is synchro-drive: the translational velocity always points along the heading θ — the robot can't slide sideways, that's the non-holonomic constraint. So in the global frame the position just integrates the velocity projected on the heading:

  x(tₙ) = x(t₀) + ∫ v(t) cos θ(t) dt,  y(tₙ) = y(t₀) + ∫ v(t) sin θ(t) dt.

But I can't set v(t) and θ(t) directly. What I set — really, what the motor currents set — are the accelerations. v(t) is the initial v(t₀) plus the integral of the translational acceleration v̇ up to t; θ(t) is θ(t₀) plus the integral of ω, and ω itself is ω(t₀) plus the integral of the rotational acceleration ω̇. Substituting all of that in, the x-coordinate becomes a double-integral expression that depends only on the initial dynamic configuration (x₀, v(t₀), θ(t₀), ω(t₀)) and the acceleration profiles v̇, ω̇. That's the honest dynamic model: trajectory as a functional of the accelerations. And the accelerations are exactly the controllable, bounded quantities — they're monotonic in the motor currents, and limiting current limits acceleration. Good; the bound I care about lives on the right variable.

Now this is unwieldy — a trajectory that's a double integral of arbitrary acceleration profiles is not something I want to be intersecting with obstacles four times a second. I need to discretize. Digital hardware only lets me change the currents at ticks, so over each interval [tᵢ, tᵢ₊₁] the accelerations v̇ᵢ, ω̇ᵢ are constant. Plugging constant accelerations into the integral gives the exact discrete form: the speed term is v(tᵢ) + v̇ᵢ·τ, and the heading inside the cosine is θ(tᵢ) + ω(tᵢ)·τ + ½·ω̇ᵢ·τ², with τ = t − tᵢ. Still messy — that ½ω̇τ² inside the cosine means the heading is sweeping quadratically and the integral has no cheap geometric form. Geometric operations like "does this trajectory cross that obstacle?" on such a curve are expensive.

So let me make one more approximation, and watch what it buys. Over a *small* interval the velocity barely changes, so instead of letting v ramp linearly across the interval I'll just hold it at some constant value vᵢ somewhere in [v(tᵢ), v(tᵢ₊₁)], and likewise hold ω at a constant ωᵢ in [ω(tᵢ), ω(tᵢ₊₁)]. (As the intervals shrink to zero this collapses back onto the exact equation, so I'm not changing the limit, only the per-step model.) Now inside the integral the heading is just θ(tᵢ) + ωᵢ·(t̂ − tᵢ) — linear in time — and v is the constant vᵢ. That integral I *can* solve:

  for ωᵢ ≠ 0:  Fₓⁱ(t) = (vᵢ/ωᵢ)·(sin(θ(tᵢ) + ωᵢ·τ) − sin θ(tᵢ)),
              F_yⁱ(t) = (vᵢ/ωᵢ)·(cos θ(tᵢ) − cos(θ(tᵢ) + ωᵢ·τ)),
  for ωᵢ = 0:  Fₓⁱ = vᵢ·cos θ(tᵢ)·τ,  F_yⁱ = vᵢ·sin θ(tᵢ)·τ.

Stare at the ω ≠ 0 case. There are sines and cosines of (θ + ω·τ) scaled by v/ω, added to a constant offset. That's the parametrization of a circle. Let me check it: set the center, relative to the starting point of this interval, to Mₓⁱ = −(vᵢ/ωᵢ)·sin θ(tᵢ), M_yⁱ = (vᵢ/ωᵢ)·cos θ(tᵢ). Then Fₓ − Mₓ = (vᵢ/ωᵢ)·sin(θ(tᵢ)+ωᵢτ), and F_y − M_y = −(vᵢ/ωᵢ)·cos(θ(tᵢ)+ωᵢτ). Squaring and adding leaves exactly (vᵢ/ωᵢ)². So the point stays at constant distance |vᵢ/ωᵢ| from (Mₓⁱ, M_yⁱ): the trajectory under a constant (v, ω) is a **circle of radius |v/ω|**. And the ω = 0 case is a straight line. So a robot's path over any short horizon is, to good approximation, a sequence of circular arcs — one arc per (v, ω) pair. Each velocity pair *is* a curvature.

This is exactly the leverage I wanted. Two things just happened. First, my decision variable is genuinely two-dimensional and concrete: pick a (v, ω) and I get a specific arc. Second, "where does this trajectory go and does it hit anything?" reduces to "where does this circle go and does it intersect a nearby obstacle?" — and circle-versus-point intersection is cheap. The approximation error from holding velocity constant per interval is, per step, bounded by |v(tᵢ₊₁) − v(tᵢ)|·Δtᵢ — linear in the tick length, so it vanishes as the loop runs faster, and anyway the encoders re-measure the true pose a few times a second, so the prediction only has to be good over one interval. I can live with that.

Now, the full problem is to pick a *sequence* of (vᵢ, ωᵢ), one per interval out to the horizon, such that the stitched-together arcs reach the goal and miss every obstacle. That search is exponential in the number of intervals — hopeless inside a 0.25 s loop. So I cut it down hard: I'll only optimize over the **first** interval's (v, ω) and pretend the velocity stays constant for all the remaining intervals (equivalently, zero acceleration after the first tick). Why is that not a cheat? Because (a) it makes the search two-dimensional and therefore actually solvable in real time; (b) I re-run the whole thing next tick, so whatever I "froze" gets re-decided 0.25 s later with fresh sensor data; and (c) physically, if I send no new command the velocity *does* stay constant, so the frozen-tail assumption is just "what happens if I commit to this velocity and coast." So my search space is a single (v, ω). Good — now which (v, ω) are allowed?

First cut, the trivial one: hardware limits. v can't exceed the max speed (and can go slightly negative if I allow reversing), ω is bounded by the max yaw rate. Call that rectangle of physically possible velocities Vₛ. Every candidate has to live in there.

Second cut, the important one — safety. A velocity is only acceptable if, committing to it, I can still stop before I hit whatever's on my arc. Let me actually compute when that's true. Along the arc of (v, ω), let dist(v, ω) be the distance to the nearest obstacle (a circle-versus-obstacle intersection — exactly what the arc geometry made cheap). If I'm moving at v and I slam on the brakes with translational deceleration v̇_b, how far do I travel before stopping? Constant deceleration: time to stop is v/v̇_b, distance is v·(v/v̇_b) − ½·v̇_b·(v/v̇_b)² = v²/(2·v̇_b). So my braking distance is v²/(2·v̇_b), and to stop in time I need that to be no more than the clearance:

  v²/(2·v̇_b) ≤ dist(v, ω)  ⟺  v ≤ √(2·dist(v, ω)·v̇_b)  for v ≥ 0,

and if I let the implementation command reverse motion I use |v| in the same inequality. The rotational rate is signed too, so the braking test has to be on its magnitude: |ω| ≤ √(2·dist(v, ω)·ω̇_b). So the **admissible** velocities are

  V_a = { (v, ω) : |v| ≤ √(2·dist(v, ω)·v̇_b)  ∧  |ω| ≤ √(2·dist(v, ω)·ω̇_b) }.

These are exactly the velocities from which the robot can brake to a halt before reaching the closest obstacle on the corresponding curvature. Notice how clean this is and how it differs from the direction-first methods: I never command a heading and hope; I keep only velocities that come with a guaranteed stopping distance. The faster I want to go, the more clearance I need — the constraint couples speed to the geometry, which is precisely the coupling the infinite-force methods threw away.

Third cut — the dynamics, the whole reason I switched to velocity space. Even among the safe velocities, I can't *reach* most of them this tick. From my current (v_a, ω_a), one control interval t of acceleration can only move me by ±v̇·t in translational velocity and ±ω̇·t in rotational velocity. So the reachable set is

  V_d = { (v, ω) : v ∈ [v_a − v̇·t, v_a + v̇·t]  ∧  ω ∈ [ω_a − ω̇·t, ω_a + ω̇·t] } —

a rectangle centered on my current velocity, its size set by the accelerations. This is the **dynamic window**. It's the formal version of "you can't turn that sharp right now": a velocity demanding a yaw rate outside [ω_a − ω̇·t, ω_a + ω̇·t] is simply not in the candidate set, so it can't be chosen, so the robot won't try it. Go back to the corridor-and-doorway picture: I'm barreling straight ahead, the door is hard right, the sharp-right velocity that the potential field wanted sits *outside* my dynamic window because I can't build up that yaw rate in one tick. It's not a candidate. The best candidate left is to keep going more or less straight — so the robot stays on its safe straight arc and doesn't smear itself on the far wall. The thing that made the other methods unsafe is now structurally impossible.

The legal velocities are the ones that survive all three cuts at once: the resulting search space is the intersection

  V_r = Vₛ ∩ V_a ∩ V_d —

physically possible, safe (can brake in time), and reachable this tick. It's a small region in the (v, ω) plane, and since it's two-dimensional I can just grid it and evaluate every cell.

Now, among the survivors, which one do I actually pick? I need an objective over (v, ω). Let me build it from what I want and watch what each desire forces.

I want to get to the goal, so I want to be pointed at it. Define a heading term that's maximal when the robot's heading lines up with the direction to the goal and falls off as the angle θ between them grows — concretely something like (180° − θ), or equivalently 1 − |θ|/π once I normalize. One subtlety: if I evaluate "am I pointed at the goal?" at my *current* pose I'm ignoring that the candidate (v, ω) is about to turn me. So I evaluate the heading at the pose I'd be in after applying this (v, ω) for the interval (and notionally braking to a stop), so the term credits the candidate for the turning it actually does. Call it heading(v, ω).

But heading alone is a disaster, and I can see why before I even run it: the velocity that points most directly at the goal will happily point straight at an obstacle sitting between me and the goal, and it says nothing about going fast or slow. Pointed-at-goal is necessary but blind to obstacles and blind to speed.

So bring back the clearance. I already compute dist(v, ω), the distance to the nearest obstacle on the arc; make it a second term. Small clearance → low score → the robot is pushed off curvatures that grind close to obstacles and toward ones that swing wide. (If the arc is clear all the way out, cap dist at some large constant so an open path isn't penalized for being open.) Now heading and dist pull against each other in exactly the productive way: heading aims at the goal, often through the obstacle; dist pulls the chosen arc to one side of it; the balance is a path that rounds the obstacle. That tension *is* the avoidance behavior.

Two terms still aren't enough, and here's the failure I can see coming. Among all the safe, goal-ward, high-clearance velocities, v = 0 is often available and looks fine to heading-plus-dist: standing still (or creeping) is maximally safe and can even be perfectly goal-aligned. The robot would dither or stop. So I need a term that actively rewards *moving* — a velocity term equal to the forward speed v (normalized). It breaks the tie toward progress and keeps the robot briskly moving when the way is clear.

Put them together, each normalized to [0, 1] so the weights are comparable, and maximize over the surviving velocities:

  G(v, ω) = σ( α·heading(v, ω) + β·dist(v, ω) + γ·velocity(v, ω) ).

The weights α, β, γ set the personality — how boldly it cuts toward the goal versus how wide a berth it gives obstacles versus how much it hurries. And the σ: I smooth the weighted sum over the (v, ω) grid before taking the max. Why bother? Without smoothing the maximum can sit on a knife-edge — a velocity that scores high but is one grid cell away from a velocity that grazes an obstacle, because dist can drop sharply as the arc just barely clears or just barely clips. Smoothing pulls the chosen optimum into a broad safe basin rather than onto a spike next to a cliff, which buys extra side-clearance from obstacles for free. So σ isn't cosmetic; it's what keeps the picked velocity comfortably inside the safe region instead of on its boundary.

Now the loop is clear. Each cycle: from the current velocity, form the dynamic window V_d; intersect with the hardware box Vₛ and the braking-admissible set V_a to get V_r; over that 2-D region, score every (v, ω) by α·heading + β·dist + γ·velocity, smooth, take the max; execute that command for one tick; then the encoders give me a fresh pose, the sonar gives me fresh obstacles, and I do it all again. The dynamics aren't a correction applied after the fact — they're the window itself, so every command I ever issue is one the robot can actually reach and can actually stop from. That's the thing the potential field couldn't promise.

Let me write it the way I'd actually run it. I'll grid the hardware-and-acceleration window and forward-simulate each candidate arc rather than carry the closed-form circle around — the rollout is the same arc, just stepped, and it makes the obstacle and goal bookkeeping trivial. If I want the exact stopping guarantee, the braking-admissible filter belongs before scoring; in the compact rollout version below I keep the reachable window V_s ∩ V_d explicit, reject any sampled footprint collision with infinite cost, and make near-obstacle arcs expensive through 1/min_r. I'll phrase the score as a *cost to minimize*: heading becomes the angular error to the goal, velocity becomes (max_speed − v) so slow or reverse commands are penalized, and clearance becomes 1/dist so close obstacles cost a lot.

```python
import math
from enum import Enum

import numpy as np


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    def __init__(self):
        self.max_speed = 1.0                              # [m/s]      hardware box Vs
        self.min_speed = -0.5                             # [m/s]      (allow slight reverse)
        self.max_yaw_rate = 40.0 * math.pi / 180.0        # [rad/s]
        self.max_accel = 0.2                              # [m/s^2]    sets width of Vd
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s^2]
        self.v_resolution = 0.01                          # grid step over the window
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0
        self.dt = 0.1                                     # control / prediction tick
        self.predict_time = 3.0                           # how far ahead to roll each arc
        self.to_goal_cost_gain = 0.15                     # alpha (heading)
        self.speed_cost_gain = 1.0                        # gamma (velocity)
        self.obstacle_cost_gain = 1.0                     # beta  (clearance)
        self.robot_stuck_flag_cons = 0.001
        self.robot_type = RobotType.circle
        self.robot_radius = 1.0                           # [m] circle footprint
        self.robot_width = 0.5                            # [m] rectangle footprint
        self.robot_length = 1.2                           # [m]
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
    # synchro-drive arc step: heading turns at omega, position rides v along heading
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x


def calc_dynamic_window(x, config):
    # Vs : what the hardware permits
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]
    # Vd : what one tick of acceleration can reach from the current (v, omega)
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]
    # the candidate window is Vs ∩ Vd
    return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]


def predict_trajectory(x_init, v, w, config):
    # forward-roll the constant-(v, omega) circular arc out to the horizon
    x = np.array(x_init)
    traj = np.array(x)
    t = 0.0
    while t <= config.predict_time:
        x = motion(x, [v, w], config.dt)
        traj = np.vstack((traj, x))
        t += config.dt
    return traj


def calc_to_goal_cost(trajectory, goal):
    # heading term: angular error between final heading and direction to goal
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    return abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))


def calc_obstacle_cost(trajectory, ob, config):
    # sampled footprint collision check plus reciprocal clearance cost
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
    return 1.0 / np.min(r)           # closer obstacle -> higher cost


def calc_control_and_trajectory(x, dw, config, goal, ob):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_traj = np.array([x])
    # grid Vs ∩ Vd; sampled collisions are rejected by the obstacle cost
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
                # anti-stall: if best is "stand still and already pointed right",
                # nudge a turn so the robot doesn't freeze in front of an obstacle
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_traj


def dwa_control(x, config, goal, ob):
    dw = calc_dynamic_window(x, config)          # hardware limits ∩ one-tick reachability
    return calc_control_and_trajectory(x, dw, config, goal, ob)  # pick the best (v, omega)
```

The causal chain, end to end: reasoning in *direction* space lets a method command a turn the robot can't physically make, so it crashes — therefore reason in *velocity* (v, ω) space, where bounded acceleration is a literal box. Deriving the motion from the synchro-drive equations shows that a constant (v, ω) traces a circular arc of radius |v/ω|, so every candidate velocity is one cheap-to-test arc. The clean search space is the intersection of the hardware box, the braking-admissible set |v| ≤ √(2·dist·v̇_b), |ω| ≤ √(2·dist·ω̇_b), and the one-tick reachable rectangle. In the compact rollout code I actually execute, `calc_dynamic_window` builds the hardware-and-reachability rectangle, each sampled arc is rejected if its footprint intersects an obstacle, and `1/min_r` keeps the selected command away from nearby obstacles. The essential move is still the same: dynamics live in the velocity-space search itself, so the controller chooses among commands the robot can actually reach instead of issuing an impossible desired direction and hoping the body can follow.
