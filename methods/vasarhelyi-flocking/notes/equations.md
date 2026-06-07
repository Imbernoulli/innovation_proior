# Exact equations — Vásárhelyi et al. 2018 (Materials & Methods)

All read VISUALLY from pages 7-11 of the author PDF (hal.elte.hu). Cross-checked with Virágh 2014.

## Repulsion (Eq. 2)
Half-spring (linear), cutoff at r0^rep:
  v_ij^rep = p^rep * (r0^rep - r_ij) * (r_i - r_j)/r_ij    if r_ij < r0^rep
           = 0                                              otherwise
- p^rep : linear gain of pairwise repulsion (units 1/s)
- r_ij = |r_i - r_j|
- direction (r_i - r_j)/r_ij points i AWAY from j (repulsive).
Total (Eq. 3):  v_i^rep = sum_{j != i} v_ij^rep

## Velocity alignment / "friction" (Eqs. 4-7)
Ideal braking curve (smooth velocity decay function), Eq. 4:
  D(r, a, p) = 0                              if r <= 0
             = r*p                            if 0 < r*p < a/p   (i.e. r < a/p^2)
             = sqrt(2*a*r - a^2/p^2)          otherwise
- r : distance to expected stopping point
- a : preferred deceleration (acceleration magnitude)
- p : linear gain determining crossover between the two deceleration phases.
Two regimes: linear (constant gain p) at small r, then sqrt (constant decel a) at large r.

Dynamic max allowed velocity difference (Eq. 5):
  v_ij^frictmax = max( v^frict , D(r_ij - r0^frict, a^frict, p^frict) )
- v^frict : velocity slack (allowed velocity diff independent of distance, a floor)
- r0^frict : stopping-point offset distance for the alignment.

Pairwise alignment term (Eq. 6):
  v_ij^frict = C^frict * (v_ij - v_ij^frictmax) * (v_i - v_j)/v_ij    if v_ij > v_ij^frictmax
             = 0                                                       otherwise
- v_ij = |v_i - v_j| (magnitude of velocity difference)
- C^frict : linear coefficient of velocity-alignment error reduction (gain)
- only acts when actual velocity diff EXCEEDS the distance-dependent threshold; pulls
  velocities together (direction -(v_i - v_j) since it reduces difference; written as
  proportional to (v_i - v_j)/v_ij with the bracket (v_ij - v_ij^frictmax) and it is
  subtracted — it damps relative velocity).
Total (Eq. 7):  v_i^frict = sum_{j != i} v_ij^frict
Locality: interaction range upper-bounded by distance where D(.) = 2 v^max.

## Wall / shill (Eqs. 8-9)
Shill agents near walls move at velocity v_s (magnitude v^shill) perpendicular to wall edge, inward.
  v_is^shillmax = D(r_is - r0^shill, a^shill, p^shill)        (Eq. 8)
  v_is^wall = v_is^frict(C-1) =
            = (v_is - v_is^shillmax) * (v_i - v_s)/v_is      if v_is > v_is^shillmax  (Eq. 9)
            = 0                                               otherwise
- r_is = |r_i - r_s|, v_is = |v_i - v_s|
- NO velocity slack (no v^frict floor), and proportional gain fixed to 1 (strongest possible).
- Obstacles: shill agent moves OUTWARD from obstacle; single shill at closest point.
  v_is^obstacle similarly to Eq. 9 with same shill params.

## Self-propulsion
Term parallel to actual velocity v_i, constant magnitude v^flock:
  v_i^selfprop-direction = (v_i/|v_i|) * v^flock

## Final desired velocity (Eq. 10) and speed cap (Eq. 11)
  vtilde_i^d = (v_i/|v_i|) v^flock + v_i^rep + v_i^frict + sum_s v_is^wall + sum_s v_is^obstacle
  v_i^d = (vtilde_i^d / |vtilde_i^d|) * min( |vtilde_i^d| , v^max )   (cap magnitude, keep direction)

## Realistic agent model (Langevin-type) — the "reality gap" features
- (1) Communication delay t^del (constant): received pos/vel data is old.
- (2) Inertia: real velocity v_i converges to desired v_i^d exponentially with time const tau^CTRL;
      max acceleration a^max.
- (3) Refresh rate of sensors: nonzero time period t^s.
- (4) Locality of communication: max comm range r^c.
- (5) Sensor inaccuracy: Langevin eq -> Gaussian noise + parabolic potential at r_i, SD sigma^s.
- (6) Outer noise: delta-correlated Gaussian, SD sigma, added to acceleration (wind etc).
Param set of realistic effects: { t^del, tau^CTRL, a^max, r^c, t^s, sigma^s, sigma }.

## Order parameters
- Velocity correlation phi^corr (Eq. 13): cluster-dependent avg of v_i.v_j/(|v_i||v_j|), maximize.
- Collision risk phi^coll (Eq. 14): time-avg of Heaviside Theta(r_ij(t) - r^coll) summed over pairs;
  r^coll = 3 m dangerous zone; minimize.
- Wall collision phi^wall (Eq. 15): time-avg over agents outside arena (rtilde_is signed).
- Speed phi^vel (Eq. 16): time-avg of |v_i(t)| -> v^flock.
- N^disc disconnected agents; N^min minimum cluster size (chose N^min > N/5 threshold).
- r^cluster (Eq. 12) = max(r0^rep, r0^frict + Dtilde(v^flock, a^frict, p^frict)); Dtilde = braking distance.

## Fitness (single-objective, product of partial fitnesses)
Transfer functions (codomain [0,1]):
- F1(phi,phi0,d) = 1 - S(phi,phi0,d), S sigmoid sinusoidal decay (Eq 17-18): monotonic growth.
- F2(phi,s) = exp(-phi^2/s^2): peak at phi=0, smooth decay (Eq 19).
- F3(phi,a) = a^2/(phi+a)^2: sharp peak, harsh constraint at phi=0 (Eq 20).
Global fitness (Eq. 21-22):
  F = F^speed * F^coll * F^disc * F^cluster * F^wall * F^corr
  F^speed   = F1(phi^vel, v^flock, v^tol)
  F^coll    = F3(phi^coll, a^tol)
  F^disc    = F3(N^disc, N/5)
  F^cluster = F3(N^min, N/5, N/5)
  F^wall    = F2(phi^wall, r^tol)
  F^corr    = Theta(phi^corr) * phi^corr      (multiplicative, cutoff at 0; phi^corr in [-1,1])
Tolerances chosen: v^tol = (1.5/4) v^flock m/s, a^tol = 0.00003, r^tol = 2 m.

## Optimizer
CMA-ES (covariance matrix adaptation evolution strategy), open-source Python lib (cma).
Pop size 100, 150 generations -> 15000 fitness evals. Each eval = 10-min stochastic sim.
11-dimensional parameter space:
  { r0^rep, p^rep, r0^frict, C^frict, v^frict, p^frict, a^frict, r0^shill, v^shill, p^shill, a^shill }
Params initialized at mid-value, initial SD = 1/6 of allowed range.

## Setup values (from text)
- v^flock = 4,6,8 m/s -> v^max = 6,8,10 m/s respectively.
- L^arena = 250 m square. Comm range r^c = 80 m. 100 sim agents. Comm delay 1 s.
- r^coll = 3 m. Real flights: 30 quadcopters, Pixhawk + Odroid, 20 Hz desired velocity cmd.
