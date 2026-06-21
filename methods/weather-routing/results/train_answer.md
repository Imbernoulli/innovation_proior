A ship must cross an ocean from a departure port $A$ to an arrival port $B$, and I want the fastest passage — or the cheapest in fuel, or some weighted blend. On a calm flat sea this is nothing: the great circle is the shortest path on the sphere, the ship holds its service speed the whole way, shortest distance equals shortest time, done. The open ocean is not flat. Waves, wind, and surface currents vary over thousands of kilometres and evolve over the days a crossing takes, and they do something specific and annoying — they change the speed the ship can actually make good. Beat into a head sea and the vessel slows sharply; run before a following sea and it holds speed almost unchanged. So the achievable speed is not a constant but a function $V(X, t, \text{heading})$ of where the ship is, when it is there, and which way it points relative to the waves. The shortest path in distance is therefore generally neither the fastest in time nor the cheapest in fuel — a longer route that skirts a storm and rides favourable seas can arrive earlier and burn less.

The honest description of that speed is a surge force balance on the hull, $(M + m_{11})\,dV/dt = T - R$, with $M$ the mass, $m_{11}$ the added mass in surge, $T$ the propeller thrust, and $R$ the resistance — still-water resistance plus the added resistance in waves and wind, which grows with wave height and depends strongly on the angle the ship meets the waves at. Solving that seakeeping problem thousands of times inside a route search is hopeless, so I lean on a fitted speed-loss curve. The calm-water speed set by the engine, minus a wave-induced loss that scales with significant wave height $H_{1/3}$ and is modulated by a function of the relative wave direction $\theta$ ($\theta = 0$ a head sea):
$$V = (a\,n + b) - (c\,H_{1/3} + d\,H_{1/3}^2)\,f(\theta), \qquad f(\theta) = 0.75\,e^{-0.65\,\theta^2} + 0.25,$$
where $n$ is the propeller revolutions and $a,b,c,d$ are fitted constants. The shape of $f(\theta)$ carries the dominant asymmetry: $f \approx 1$ in head seas ($\theta \approx 0$) and falls to a floor $\approx 0.25$ running before the sea ($\theta \approx \pi$), so the same wave height costs a lot of speed beating into it and little running with it. Currents fold in as a velocity vector added to the through-water velocity, giving speed and course over ground plus a small drift angle; that modifies $V$ and the actual track but not the structure of the problem.

The optimisation is then over a path $X(t)$ on the sphere whose passage time is the integral of arclength over $V$. The first instinct is to call this a shortest path with a funny metric: discretise the ocean into a grid, weight each edge by its length over the local achievable speed, and run Dijkstra. It breaks immediately, and the precise reason is the whole game. The weight of edge $(i,j)$ is length over $V$, but $V$ there depends on $H_{1/3}$ and $\theta$ at the moment the ship is passing through, and the field is moving and evolving over the multi-day crossing. The storm sitting on $(i,j)$ today is a thousand kilometres away in two days. So the cost of an edge depends on when the ship traverses it — and when it traverses it depends on the entire route up to that point, since the clock at node $i$ is the sum of all the edge times before it. Pricing each edge by "the weather there" is meaningless until I say at what clock value, and that clock is an output of the route, not an input. A static-weight Dijkstra is planning against a frozen snapshot of a sea that will not be there. The calculus-of-variations attack reaches the same optimum by solving the Euler–Lagrange equations for the least-time track, but it leans on second derivatives of the speed field, and over a long integration the numerical error in those second derivatives compounds to unacceptable levels; grid dynamic programming with the engine held fixed optimises only heading and is bounded by grid fineness. None of these honour the timing coupling cleanly while staying robust and cheap.

What resolves it, and the method I propose, is the **isochrone method** — least-time routing recast as **front propagation**. Make the time-dependence explicit: let $c_{\text{arc}}(i,j,t)$ be the time to sail $i \to j$ departing $i$ at clock $t$, computed from the weather at $t$ through the speed-loss curve. With $f^*(i,t)$ the minimum remaining time to reach the finish from node $i$ at clock $t$,
$$f^*(\text{finish}, t) = 0, \qquad f^*(i, t) = \min_j\big[\, c_{\text{arc}}(i,j,t) + f^*\big(j,\; t + c_{\text{arc}}(i,j,t)\big) \,\big].$$
That second argument is the load-bearing point: after taking the edge the clock has advanced by exactly the edge time, so the subproblem handed to $j$ is evaluated at the new weather. This is the forward dynamic program of a **time-dependent shortest path**, and Bellman's principle holds — the tail of an optimal route is itself optimal from the node-time pair it starts at. Running it backward is awkward, because I do not know in advance at what clock values $f^*(j,\cdot)$ will be queried, and the weather is only meaningfully indexed forward from the known departure time. So I run it forward: start at $A$ at the departure clock, propagate the earliest arrival time to every reachable point, always settling the smallest, looking up the weather at a point's settled arrival time before relaxing its outgoing edges. That is Dijkstra on the time-dependent graph.

The geometry of this forward sweep is what makes it efficient and intelligible. Consider everywhere reachable after one step $\Delta t$: fan the ship's heading off $A$, apply the achievable speed in the local weather, step forward $V\cdot\Delta t$, and you get a little arc of reachable points. After two steps each of those fans again, and the boundary of "everywhere reachable within $k\,\Delta t$" is a curve — an **isochrone**, the contour of farthest reach in equal sailing time. Propagating earliest-arrival labels *is* sweeping this curve outward, so instead of settling individual grid nodes I carry the whole frontier and advance it: fan headings off every current front point, step $\Delta t$, take the outer boundary of the resulting cloud as the next isochrone, and repeat until a front reaches $B$. The front bulges ahead where seas are kind and lags where the ship fights a head sea — the abstract DP frontier made physical.

The cloud has to be kept thin or the count multiplies by the heading fan every step and blows up worse than the grid, and the rule that thins it is **dropping dominated points**. Take two candidates in nearly the same direction out of the start, one pushed farther along that bearing than the other. The nearer one is dominated: the front already encloses it, so it was reached at least as early via the boundary, and anything achievable onward from it is achievable at least as well from the farther point on the same bearing, having spent no more time to get there. So within a narrow bearing sector only the farthest-advanced candidate can lie on an optimal route, and every interior point is throwaway. I bin the candidates by their azimuth from the start and keep, per bin, the one that has travelled farthest. This is exactly settling the time-dependent-Dijkstra frontier: a "place" resolved at the granularity of a direction sector, and "earliest arrival" being "farthest reached for the same number of steps" — equal step count means equal elapsed time, so farthest-for-equal-time is earliest-for-equal-place. The prune is the frontier settling, not a heuristic departure from it, and it sidesteps the second derivatives the calculus of variations needs.

One wrinkle appears the moment this is programmed rather than drawn. By hand the isochrone is a smooth curve and "take the outer boundary" is obvious to the eye; coded as a raw polygon through the outermost points, the front can fold over itself where the speed field is sheared, producing spurious self-intersecting "isochrone loops" that corrupt the boundary and the recovered route. The fix follows straight from the pruning rule once I make the sectors structural rather than a free-drawn polygon: lay a grid aligned to the great circle from $A$ to $B$, with lanes running across-track, and index every candidate by which cross-track lane it falls in. "Keep the farthest per sector" becomes "keep, per lane, the single sub-point that has advanced farthest," so the front is a list of lane-survivors, ordered by lane, that physically cannot self-cross — the **modified** isochrone, robust against loops. I float the grid, re-laid around the current front each step, so the lanes track the live part of the route and the resolution stays where the route is. Centering the lanes on the great circle is exactly right because the great circle is the calm-water optimum and the weather-optimal route is a perturbation of it, living in a band around it; spending the grid there bounds the search width instead of tiling the whole ocean. The heading fan is the other real design choice, and it fails on both sides: too few courses and the optimal evasive turn to dodge a storm or catch a following sea falls between the discrete bearings and is missed; too many and each front point spawns a huge fan that explodes the cloud before pruning culls it. So I take enough segments to resolve a useful turn, spread over a width $\pm\Delta$ wide enough for a meaningful detour but not the whole compass, centered on the *current* bearing to the destination (recomputed each step) so the fan stays pointed sensibly even after weather has pushed the route off the great circle.

Fuel comes back in by changing what each step holds constant. Stepping in equal time, with the engine fixed, gives the minimum-time route and merely reports its fuel; it does not minimise fuel. The cleaner move is to advance every front point by an equal amount of fuel $\Delta\text{fuel}$ instead. For a given point and heading the local fuel rate sets how long that fuel lasts, $\Delta t = \Delta\text{fuel} / \text{fuel\_rate}$, and the distance covered is $V\cdot\Delta t$; the front is now a contour of equal fuel burned — an **isofuel** line. Everything else is identical: fan, step, prune the dominated lane-survivors, advance. Minimum-time drops out as the special case where the stepped resource is time itself. (There is a more general version where the equal-cost surface lives in three dimensions, position and time together, so engine power can vary along the route — an isopone — but it is notoriously hard to reason about operationally, and the plain front in one resource at a time is what is tractable and intelligible.) Whichever resource I step in, once the front reaches $B$ I have many complete routes that got there; I pick the survivor with the least total of the objective — minimum total fuel, or minimum total time — and backtrack its parent pointers to recover the waypoints.

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
