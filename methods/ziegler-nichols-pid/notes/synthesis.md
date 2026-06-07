# Synthesis — PID feedback + Ziegler–Nichols tuning

## The pain (research question)
A process controller (valve driven by a measured variable) must hold a variable at setpoint despite load changes. The dominant control effect, proportional response, has two visible defects observed on chart records:
- **Offset**: proportional-only control can hold only one valve position when the pen is exactly at setpoint. A sustained load change forces the pen to move off setpoint far enough to command the needed valve change. Offset ∝ 1/sensitivity and ∝ load change. (Z-N pp. 760–761.)
- **Overshoot / oscillation / amplitude ratio**: at high sensitivity the loop rings. There is a definite **ultimate sensitivity** Su above which any oscillation grows, below which it decays; AT Su the oscillation is sustained with amplitude ratio = 1 (each wave equal to the last). (pp. 759–760.) Stability measured by amplitude ratio = ratio of a wave's amplitude to the preceding wave.

Quarter-amplitude decay = amplitude ratio 0.25 (each wave 1/4 the last) is the chosen good compromise; it occurs at ≈ 0.5 Su.

## Three control effects (Z-N names → modern names)
- **Proportional response** — valve movement ∝ pen movement. Measure = **sensitivity** Su (= proportional gain Kp, units psi/in; reciprocal = throttling range = proportional band).
- **Automatic reset** — valve *velocity* ∝ pen displacement from setpoint. Purpose: eliminate offset. Measure = **reset rate** in per-minute = number of times per minute reset duplicates the proportional correction = 1/Ti (integral). Increasing reset → faster offset removal but *decreases stability and increases period*.
- **Pre-act** — additional valve movement ∝ rate of pen movement. Measure = **pre-act time** in minutes = Td (derivative). Increases stability, decreases period, allows larger settings of the other two. (Predictive/anticipatory.) Above its optimum, pre-act again *reduces* stability.

## Background / ancestry (Minorsky 1922)
Minorsky, "Directional Stability of Automatically Steered Bodies," J. Amer. Soc. Naval Engineers 34 (1922) 280–309. Designing autopilot for USS New Mexico. Watched a helmsman: steers on present course error, on how long it's been off (accumulated), and on how fast it's changing. Wrote the ship rotational ODE (inertia, damping, rudder angle) and solved for rudder angle → three terms ∝ deviation, integral of deviation, derivative of deviation = P, I, D. I term cancels persistent disturbance (wind) offset; D term gives predictive damping of the hunting oscillation. Constants set by trial and error on USS New Mexico (1923) — worked. This is the conceptual ancestor; the gap it leaves is *how to choose the three constants systematically without a plant model*.

Also: on-off control (infinite sensitivity) always oscillates; throttling (finite proportional) gives damped oscillation then straight-line control with residual offset. Stability is NOT automatic — high gain destabilizes.

## The intellectual move
P alone: offset + ringing. Add I (reset) to kill offset (integrate until error = 0). Add D (pre-act) to damp the ring and allow higher P. Then: a model-free *recipe* to pick the three from the observed response — two routes:

### Method 1: Ultimate-cycle (closed-loop) — from Su, Pu
Raise sensitivity (P only, no reset, no pre-act) until sustained oscillation: that is ultimate sensitivity Su = Ku, with period Pu = Tu.
- P:   Kp = 0.5 Su
- PI:  Kp = 0.45 Su, reset rate = 1.2/Pu  → Ti = Pu/1.2 = 0.833 Pu
- PID: Kp = 0.6 Su,  reset rate = 2/Pu    → Ti = Pu/2 = 0.5 Pu;  pre-act = Pu/8 = 0.125 Pu (Td)
(Note PI sensitivity dropped 0.5→0.45 because adding reset reduces stability.)

### Method 2: Process-reaction curve (open-loop) — from R, L
Open the loop (disconnect controller), make a sudden step ΔF in valve. Pen traces an S-curve. Draw tangent at inflection: slope = reaction rate R (in/min); tangent intersects initial pen level at time = **lag** L (dead time). Unit reaction rate R₁ = R/ΔF (per psi). Empirically Pu ≈ 4L (period at Su).
- P:   sensitivity = 1/(R₁L)
- PI:  sensitivity = 0.9/(R₁L), reset rate = 0.3/L  → Ti = L/0.3 = 3.33L
- PID: sensitivity = 1.2/(R₁L), reset rate = 0.5/L → Ti = L/0.5 = 2L; pre-act = 0.5L (Td)
Derivation of PI/PID reaction-curve numbers: substitute Pu = 4L into the ultimate-cycle formulas. e.g. reset 1.2/Pu = 1.2/4L = 0.3/L; reset 2/Pu = 2/4L = 0.5/L; pre-act Pu/8 = 4L/8 = 0.5L; sensitivity uses Su = 2/(R₁L) so 0.45Su = 0.9/(R₁L), 0.6Su = 1.2/(R₁L), 0.5Su = 1/(R₁L). All consistent.

In FOPDT modern terms R = slope = K_proc/T per unit step (process gain K over time constant T), so 1/(R L) = T/(K L); PID Kp = 1.2 T/(K L), Ti=2L, Td=0.5L. (textbook form).

## Standard PID parallel form & relations
u(t) = Kp·e + Ki·∫e dt + Kd·de/dt, with
- Ki = Kp/Ti  (Ti = integral/reset time; reset rate = 1/Ti)
- Kd = Kp·Td  (Td = derivative/pre-act time)

## Practical implementation details (for code)
- **Anti-windup**: when actuator saturates, integral keeps accumulating → big overshoot on recovery. Fix: clamp integral / back-calculation (only integrate when not saturated, or clamp output and back-calc integral term).
- **Derivative on measurement**: take d/dt of the process variable (−Kd·dPV/dt) not of the error, to avoid derivative "kick" on a step setpoint change (a step in setpoint gives an impulse in de/dt). Also usually filter the derivative.
- Discrete update each Δt: e=sp−pv; integral += e·Δt (with clamp); deriv = (pv − pv_prev)/Δt; u = Kp·e + Ki·integral − Kd·deriv; saturate u.

## In-frame constraints
- Never cite the Z-N 1942 paper as published artifact. May cite Minorsky 1922, on-off control, Wiener/Hall etc as prior art.
- reasoning.md: continuous first person, discovery order, recall-and-reason about offset/oscillation phenomena (don't "measure"). The sustained oscillation at Su, offset ∝ 1/sensitivity, Pu≈4L, amplitude-ratio chart records are established phenomena → invoke as observed, reason about them. The numbers 0.5/0.45/0.6, 1.2/2 per Pu, Pu/8, 4L are the derived recipe — present them as the recipe being built.
