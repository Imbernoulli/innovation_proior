# Ship weather routing by the isochrone method

## Problem

Find the minimum-time (or minimum-fuel) ocean route from a departure point A to an arrival
point B when the ship's achievable speed depends on a **time-varying** weather field — waves,
wind, current — that the ship encounters along the way. The static great-circle / shortest-path
answer is wrong because the cost of any leg depends on *when* the ship sails it (the weather
there changes over the multi-day crossing), and that timing is itself an output of the route.

## Key idea

Treat least-time routing as **front propagation**. From the start, propagate the boundary of all
points reachable in equal sailing time — an **isochrone** — one step at a time: fan a set of
headings off every point on the current front, apply the local achievable speed from the ship's
speed-loss curve in the weather at the current clock, and step forward. The outer boundary of the
resulting cloud is the next isochrone. Repeat until a front reaches B; backtrack the parent
pointers to recover the route.

The cloud is kept thin by **dropping dominated points**: within a narrow bearing sector out of the
start, only the farthest-advanced candidate can lie on an optimal route (a nearer point is already
enclosed, hence reached at least as early via the boundary). Binning candidates by bearing and
keeping the per-bin maximum-progress survivor is exactly this prune. Carrying the bins as
structured cross-track lanes aligned to the great circle — the **modified** isochrone — prevents
the front from self-crossing into spurious "isochrone loops" that the naive geometric construction
produces when computerised.

This is precisely the forward dynamic program of a **time-dependent shortest path**. With arc cost
c_arc(i, j, t) = the time to sail i → j departing at clock t (priced with the weather at t),

  f*(i, t) = 0 if i = finish, else min_j [ c_arc(i, j, t) + f*( j, t + c_arc(i, j, t) ) ],

the isochrone is the earliest-arrival frontier of this DP and sector-pruning is the frontier
settling — i.e. Dijkstra's algorithm on the time-dependent graph. The grid dynamic-programming
methods solve the same recursion on a fixed great-circle-referenced grid.

## Speed-loss model

Achievable speed from calm-water speed minus a wave-induced loss, modulated by relative wave
direction θ (θ = 0 head sea):

  V = (a·n + b) − (c·H₁/₃ + d·H₁/₃²)·f(θ),  f(θ) = 0.75·exp(−0.65·θ²) + 0.25,

with n the propeller revolutions and H₁/₃ the significant wave height. f(θ) ≈ 1 in head seas and
falls to a floor ≈ 0.25 in following seas, capturing the dominant asymmetry cheaply. Currents add
a velocity vector (speed/course over ground) and a drift angle. The mechanistic basis is the surge
balance (M + m₁₁)·dV/dt = T − R, R = still-water + added resistance in waves and wind.

## Algorithm

1. Front ← {A} at the departure clock; reference frame = great circle A → B.
2. Repeat up to a step budget:
   a. For each front point, fan headings around the current bearing to B (course_segments + 1
      courses spread over ±Δ).
   b. **Step**: look up the weather at the front point and clock; get achievable speed (and fuel
      rate) from the speed-loss curve. For an **equal-time** step (classic isochrone) advance V·Δt.
      For an **equal-fuel** step (isofuel), Δt = Δfuel / fuel_rate, then advance V·Δt.
   c. Drop legs crossing land / shallow / unsafe seas.
   d. Move along the geodesic to the candidate points; accumulate distance made good.
   e. **Prune**: bin candidates by bearing-from-start (lanes centered on the great circle); keep
      the single farthest-advanced candidate per lane. Advance the clock.
   f. Stop when a front comes within one step of B.
3. **Backtrack** the survivor that is best on the objective (least total fuel, or least total time)
   to recover the waypoint sequence.

Equal-time stepping minimises passage time; equal-fuel stepping minimises fuel; min-time is the
special case of stepping in the time resource.

## Code

```python
import numpy as np
from geovectorslib import geod          # geod.direct(lat, lon, azi, dist); geod.inverse(lat1, lon1, lat2, lon2)
from scipy.stats import binned_statistic


class IsoBased:
    """Front-propagation router. The front is parallel arrays of route-heads;
    one column = one surviving route so far. Stacked per step so backtracking
    a route is just reading a column back through the history."""

    def __init__(self, start, finish, start_time, ship, weather, constraints,
                 course_segments, course_inc_deg,        # heading fan: count, spacing
                 prune_segments, prune_sector_deg_half,   # dominated-point binning
                 ncount, delta):
        self.start, self.finish = start, finish
        self.time = start_time
        self.ship, self.weather, self.constraints = ship, weather, constraints
        self.course_segments, self.course_inc = course_segments, course_inc_deg
        self.prune_segments, self.prune_half = prune_segments, prune_sector_deg_half
        self.ncount, self.delta = ncount, delta
        # great circle A -> B: calm-water optimum, used as the reference frame
        self.gcr_course = geod.inverse([start[0]], [start[1]], [finish[0]], [finish[1]])['azi1']
        self.lats = np.array([[start[0]]]); self.lons = np.array([[start[1]]])
        self.full_dist = np.array([0.0])    # distance made good per surviving route

    def define_courses(self):
        # fan headings around the current bearing to the destination, every front point
        n = self.lats.shape[1]
        bearing = geod.inverse(self.lats[0], self.lons[0],
                               np.repeat(self.finish[0], n), np.repeat(self.finish[1], n))['azi1']
        for a in ('lats', 'lons', 'full_dist'):
            setattr(self, a, np.repeat(getattr(self, a), self.course_segments + 1, axis=-1))
        deltas = np.linspace(-self.course_segments / 2 * self.course_inc,
                             +self.course_segments / 2 * self.course_inc, self.course_segments + 1)
        return np.repeat(bearing, self.course_segments + 1) - np.tile(deltas, n)

    def step(self, courses):
        # one equal-TIME step: achievable speed in the local weather -> distance
        wx = self.weather.get(self.lats[0], self.lons[0], self.time)
        speed = self.ship.speed(courses, wx)
        return courses, speed * self.delta

    def move(self, courses, dist):
        nxt = geod.direct(self.lats[0], self.lons[0], courses, dist)
        self.lats = np.vstack((nxt['lat2'], self.lats))
        self.lons = np.vstack((nxt['lon2'], self.lons))
        self.full_dist = self.full_dist + dist

    def prune(self):
        # bin by bearing-from-start (lanes centered on the gcr); keep the
        # farthest-advanced candidate per lane = drop dominated points = settle
        # the time-dependent-Dijkstra frontier; structured lanes stop loops.
        m = self.lats.shape[1]
        bearing = geod.inverse(np.repeat(self.start[0], m), np.repeat(self.start[1], m),
                               self.lats[0], self.lons[0])['azi1']
        axis = geod.inverse([self.start[0]], [self.start[1]],
                            [self.finish[0]], [self.finish[1]])['azi1']
        edges = np.sort(np.linspace(axis - self.prune_half, axis + self.prune_half,
                                    self.prune_segments + 1))
        stat, edges, _ = binned_statistic(bearing, self.full_dist, statistic=np.nanmax, bins=edges)
        idxs = list({np.where(self.full_dist == s)[0][0] for s in stat if s > 0})
        self.lats = self.lats[:, idxs]; self.lons = self.lons[:, idxs]
        self.full_dist = self.full_dist[idxs]

    def reached(self):
        m = self.lats.shape[1]
        d = geod.inverse(self.lats[0], self.lons[0],
                         np.repeat(self.finish[0], m), np.repeat(self.finish[1], m))['s12']
        return np.any(d < self.delta * self.ship.max_speed)

    def run(self):
        for _ in range(self.ncount):
            courses = self.define_courses()
            courses, dist = self.step(courses)
            ok = ~self.constraints.violated(self.lats[0], self.lons[0], courses, dist)
            courses, dist = courses[ok], dist[ok]   # (mask the per-step arrays too in a full impl.)
            self.move(courses, dist)
            if self.reached():
                break
            self.prune()
            self.time = self.time + self.delta
        return self.backtrack()

    def backtrack(self):
        best = int(np.argmin(self.objective_total()))
        return list(zip(self.lats[:, best], self.lons[:, best]))


class IsoFuel(IsoBased):
    """Step in equal FUEL: each step burns delta_fuel, the local fuel rate sets
    how long it lasts, hence the distance. Front = isofuel line. Min-time is the
    special case of stepping in the time resource."""

    def __init__(self, *args, delta_fuel, **kw):
        super().__init__(*args, **kw)
        self.delta_fuel = delta_fuel

    def step(self, courses):
        wx = self.weather.get(self.lats[0], self.lons[0], self.time)
        speed = self.ship.speed(courses, wx)
        fuel_rate = self.ship.fuel_rate(courses, wx)
        delta_time = self.delta_fuel / fuel_rate         # equal-fuel -> variable time
        return courses, speed * delta_time

    def objective_total(self):
        return self.total_fuel_per_route()               # least total fuel among reached routes
```

The structure mirrors a real isochrone/isofuel router: a base class owns the front arrays, the
great-circle reference, the propagation loop and the dominated-point pruning; the fuel subclass
overrides only what one step holds constant (fuel instead of time) and what the final survivor is
selected on (total fuel). Backtracking reads a column back through the per-step history.
