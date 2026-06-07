# Context: Minimum-time/fuel ocean routing through a time-varying weather field

## Research question

A ship must cross an ocean from a departure port A to an arrival port B. On a featureless,
calm sea the answer is trivial: follow the great circle, the shortest path on the sphere, at
the ship's service speed. The open ocean is not calm. Waves, wind, and surface currents vary
over thousands of kilometres and change over the days a crossing takes, and they directly
degrade the speed the ship can actually make good: a vessel butting into a head sea slows
sharply, while in a following sea it holds speed almost unchanged. So the shortest path in
distance is generally not the fastest in time, and not the cheapest in fuel — a longer route
that skirts a storm and rides favourable seas can arrive earlier and burn less.

The precise problem: given the departure and arrival points, the ship's speed/fuel response to
the sea state it meets, and a forecast of the weather field over the crossing window, find the
route (and, in the richer version, the engine setting along it) that minimises passage time —
or fuel, or a weighted combination — subject to staying on water and within safety limits on
the seas the ship is exposed to.

The difficulty that makes this more than a textbook shortest path: the cost of traversing any
leg depends on **when** the ship traverses it, because the weather there is different at noon
than it is two days later, and *when* the ship is at a place is itself determined by the route
chosen up to that point. The field the ship encounters is coupled to the ship's own progress.
A method that ignores this — that prices each leg by today's weather, or by an average — will
plan against a sea that will not be there when the ship arrives. A solution has to propagate
the route and the clock together.

## Background

**The great circle as the calm-water optimum.** On a sphere the minimum-distance track between
two points is the great circle; at constant speed it is also the minimum-time track in the
absence of weather. This matters as a reference: when weather is moderate the optimal route is
a perturbation of the great circle, staying near it, so a method can concentrate its effort in
a band around the great circle rather than over the whole ocean.

**Speed loss in a seaway.** The load-bearing physical fact is that achievable ship speed is a
function of the local sea state and of the ship's heading relative to the waves. Mechanistically
the ship's surge obeys a force balance, (M + m₁₁(ω))·dV/dt = T(V) − R(V, sea state, heading),
with M the mass, m₁₁ the added mass in surge, T the propeller thrust, and R the resistance —
still-water resistance plus the **added resistance in waves and wind**, which grows with wave
height and is strongly dependent on encounter angle. Solving the full seakeeping problem inside
a route optimiser is too expensive, so practical routing uses a fitted speed-loss curve. A
representative form, fitted to container-ship data, is

  V = (a·n + b) − (c·H₁/₃ + d·H₁/₃²)·f(θ),  f(θ) = 0.75·exp(−0.65·θ²) + 0.25,

with n the propeller revolutions, H₁/₃ the significant wave height, θ the relative wave
direction in radians (θ = 0 is a head sea), and a,b,c,d fitted constants. The first bracket is
the calm-water speed set by the engine; the second is the wave-induced loss. The shape of f(θ)
encodes the dominant asymmetry: the loss is largest in head seas (θ ≈ 0, f ≈ 1) and falls to a
floor of about 0.25 in following seas (θ ≈ π), so the same wave height costs a lot of speed
beating into it and little running before it. Currents enter separately, as a vector added to
the ship's water-relative velocity to get speed and course over ground, and as a drift angle.

**The min-time problem on a time-varying field.** Cast continuously, the route is a path X(t)
on the sphere whose speed at each instant is V(X, t, heading) set by the field and the speed-loss
curve, and passage time is the integral of arclength over that speed. Because V depends on t
through the moving weather, this is optimisation over a **time-varying cost field**, not a static
metric. The classical attack — calculus of variations — writes the Euler–Lagrange equations for
the least-time track; it treats routing as a continuous problem, and is known to be brittle
because it needs second-order derivatives of the speed field, and numerical error in those can
grow unacceptably along a long integration.

**The isochrone idea.** A different, discrete way to think about least time: from the departure
point, ask "where can the ship be after one time step Δt?" Fanning the heading and applying the
local achievable speed gives a set of reachable points; their outer boundary is the **isochrone**
— the contour of farthest reach in equal sailing time. Repeating from that boundary gives the
next isochrone (reach after 2Δt), and so on. This is a front-propagation view of least time:
successive isochrones are reachability fronts, and when a front reaches the destination the
number of steps gives the passage time. James (1957), in the first study of optimal ship
routing, used exactly this construction with synoptic and prognostic wave charts to pick a
least-time track by hand, including curves relating ship speed to head/beam/following waves of
varying height.

**Dominated points and the front.** When a reachable cloud is collapsed to its outer boundary,
the interior points are discarded — and rightly so: a point the front already encloses has been
reached at least as early along the boundary, so it cannot lie on a faster route. Concretely, if
two candidate points fall in the same narrow direction sector out of the start, the one that has
advanced farther dominates; the nearer one can be pruned. Keeping only the farthest-advanced
candidate per direction sector is what makes the front a thin curve rather than a thickening
blob, and is the discrete engine of the method.

**The time-dependent shortest path / dynamic programming view.** Discretise the ocean into a
grid of nodes and let c_arc(i, j, t) be the time to sail from node i to a neighbour j when
departing i at clock time t, computed from the weather at t through the speed-loss curve. Because
the arc cost carries t, this is a **time-dependent shortest-path** problem. Its forward dynamic
program is

  f*(i, t) = 0 if i = finish, else min_j [ c_arc(i, j, t) + f*(j, t + c_arc(i, j, t)) ],

with the minimising j giving the successor on the optimal path (Allsopp 1998). Equivalently a
two-dimensional dynamic program over a great-circle-referenced grid (Bijlsma 1975; De Wit 1990;
Calvert 1991) solves the same recursion stage by stage. The connection to label-setting search
is direct: propagating earliest-arrival labels outward and settling the frontier is Dijkstra's
algorithm on the time-dependent graph; biasing the frontier toward the destination is A*. The
isochrone is the geometric face of this — an isochrone at k·Δt is the set of nodes whose
earliest-arrival label is k·Δt, and pruning dominated points is the settling of the frontier.

**The computer-implementation failure mode.** Drawn by hand the isochrone is a smooth curve, but
when the geometric front-construction is programmed directly it can fold over itself, producing
spurious self-crossing "isochrone loops" that corrupt the boundary and the recovered route. This
is a known pathology of the naive computerised isochrone and is the specific defect later
refinements set out to remove — by representing the front on a structured grid rather than as a
freely-drawn geometric boundary.

## Baselines

**Hand-drawn isochrones on weather charts (James 1957).** The original method: propagate
reachability fronts on synoptic/prognostic wave charts using tabulated speed-vs-wave curves, by
hand, to select an approximate least-time track. Core idea is exactly the isochrone propagation
above. Gap: manual, low-resolution, not reproducible, and it offers no clean rule for collapsing
the reachable cloud — the dominated-point pruning is done by eye.

**Calculus of variations (Bijlsma 1975).** Treats routing as a continuous least-time problem and
solves the Euler–Lagrange / characteristic equations for the optimal track in the time-varying
field. Core math: stationarity of the time integral ∫ ds/V(X,t,heading). Gap: requires
second-order derivatives of the speed field; numerical errors in those compound along a long
voyage and can reach unacceptable levels, and handling land/safety constraints and discrete
weather grids is awkward in the continuous formulation.

**Grid dynamic programming, 2DDP (De Wit 1990; Calvert 1991).** Lay a grid of stage lines
perpendicular to the great circle; at each stage keep a set of grid points; apply Bellman's
principle to find, for each grid point, the best predecessor. Core algorithm: the forward/backward
recursion f*(i,t) above on a fixed grid, with heading as the control and engine power held
constant. Gap: with the propeller setting fixed it optimises only heading; accuracy is bounded by
grid fineness, and treating time as merely a by-product of progress (the usual 2D reduction)
assumes a one-to-one progress↔time map that the speed-loss coupling does not strictly honour.

**Isopone method (Klompstra; Spaans 1995).** Extends the isochrone to a three-dimensional
"isopone" — a surface of equal fuel consumption bounding the attainable region in (position, time)
space — so engine power can be varied. Core idea: front propagation in fuel rather than time. Gap:
reported as mathematically elegant but hard for shipboard operators to understand and operate, so
in practice it was set aside in favour of the simpler front method.

**Generic Dijkstra/A* on a weather graph (Padhy 2008).** Build a graph over ocean grid nodes with
time-priced edges and run a shortest-path search. Core: label-setting search with the time-varying
edge cost. Gap: a vanilla static-graph search prices edges without honouring that the cost depends
on arrival time, and an unstructured node graph spreads search effort far from the great circle
where the route does not live, costing memory and time.

## Evaluation settings

The natural yardstick is a set of long ocean crossings where weather genuinely bites — North
Pacific routes (e.g. Japan ↔ U.S. west coast) and North Atlantic routes (e.g. Lisbon–Miami,
Halifax–Plymouth) over storm seasons — using a standard test vessel such as the SR-108 container
ship or a bulk carrier with a known speed-loss curve. The weather input is a reanalysis or forecast
field of significant wave height, wave direction and period, wind, and surface current at a few
degrees and roughly six-hour resolution, from sources such as NCEP/WaveWATCH III or ECMWF. The
metrics are passage time and total fuel consumption for the planned route (and, where safety is
modelled, a voyage-risk coefficient from the exposure to severe seas), with runtime and memory as
practicality measures. The departure point, arrival point, departure time, ship speed-loss model,
and the weather field are the inputs; the great-circle route and a constant-heading rhumb route
are the trivial comparison tracks.

## Code framework

The primitives that already exist: geodesic computations on the sphere (given a point, a course,
and a distance, the destination point; given two points, the distance and initial course); a
weather field that returns wave height, direction, wind, and current by interpolation at a queried
(lat, lon, time); a ship model that, given heading and local weather, returns the achievable speed
and the fuel rate via the speed-loss curve; and a constraints check that flags a leg crossing land,
shallow water, or unsafe seas. A routing pass consumes the start point, the finish point, the
departure time, the ship model, and the weather field, and emits a route as a sequence of waypoints
with times. The empty bodies below are the generic slots where the front-propagation step, the
dominated-point reduction, and the route backtrack still have to be designed.

```python
# --- existing primitives -------------------------------------------------
geod.direct(lats, lons, courses, dists)   # advance along geodesic -> new lat/lon
geod.inverse(lat1, lon1, lat2, lon2)      # -> distance s12, initial azimuth azi1

class WeatherField:
    def get(self, lat, lon, time):        # -> wave height H, wave dir, wind, current
        ...

class Ship:
    def speed_and_fuel(self, course, weather):
        # achievable speed and fuel rate from the speed-loss curve
        ...

class Constraints:
    def violated(self, lat1, lon1, lat2, lon2):   # land / depth / unsafe sea
        ...

# --- the routing pass: pre-method skeleton -------------------------------
class Router:
    def __init__(self, start, finish, start_time, ship, weather, constraints):
        self.start, self.finish = start, finish
        self.time = start_time
        self.ship, self.weather, self.constraints = ship, weather, constraints
        # great-circle course start -> finish, used as reference frame
        self.gcr_course = geod.inverse(*start, *finish)['azi1']

    def expand_front(self, front):
        # TODO: from each point of the current front, fan headings, advance one
        #       step using the achievable speed in the local weather -> candidates
        pass

    def reduce_front(self, candidates):
        # TODO: collapse the candidate cloud to the surviving front
        #       (the slot where dominated points are dropped)
        pass

    def reached_destination(self, front):
        # TODO: is the finish within one step of the front?
        pass

    def backtrack(self, front):
        # TODO: follow parent pointers from the destination back to the start
        pass

    def run(self):
        front = [self.start]      # step-0 front is the departure point
        while not self.reached_destination(front):
            candidates = self.expand_front(front)
            candidates = [c for c in candidates
                          if not self.constraints.violated(*c.parent, *c.point)]
            front = self.reduce_front(candidates)
            self.time += self.delta_t   # advance the clock with the front
        return self.backtrack(front)
```
