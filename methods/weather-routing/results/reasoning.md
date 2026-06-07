OK, let me think this through from scratch. I have a ship that needs to get from port A across an ocean to port B, and I want the fastest crossing — or the cheapest in fuel, but let me start with time and come back to fuel. On a calm flat sea this is nothing: the great circle is the shortest path on the sphere, the ship holds its service speed the whole way, shortest distance equals shortest time, done. So why isn't routing trivial? Because the sea is not flat. There are waves, there's wind, there are currents, and they do something specific and annoying — they change the speed the ship can actually make. Beat into a head sea and the ship slows right down; run before a following sea and it barely loses anything. So the speed I get is not a constant, it's a function of where I am and which way I'm pointed relative to the waves.

Let me write down what I actually know about that speed, because the whole problem hinges on it. The honest version is a force balance on the hull — surge dynamics, (M + m₁₁)·dV/dt = thrust minus resistance, and the resistance is still-water resistance plus the added resistance the waves and wind pile on, which grows with wave height and depends a lot on the angle I meet the waves at. I am not going to solve a seakeeping boundary-value problem inside a route search; that's hopeless to do thousands of times. So I'll lean on a fitted curve. Something like V = (calm-water speed set by the engine) minus (a loss term that scales with significant wave height H₁/₃) — and crucially the loss is modulated by a function of the relative wave direction θ. The shape that matters: the loss is biggest when θ ≈ 0, head seas, and it decays to a small floor when θ ≈ π, following seas. Write the modulation as f(θ) = 0.75·exp(−0.65θ²) + 0.25 and it does exactly that — about 1 at head seas, about 0.25 running before the sea. So V(position, heading, time) = (a·n + b) − (c·H₁/₃ + d·H₁/₃²)·f(θ), where H₁/₃ and θ come from the weather field at my position and the current time. Currents I'll fold in as a velocity vector added to the through-water velocity, giving speed and course over the ground, plus a small drift angle; that doesn't change the structure, it just modifies V and the actual track.

So now the optimisation. The route is a path X(t) on the sphere, and at every instant my speed along it is V(X, t, heading). Passage time is the integral of arclength divided by V. The first instinct is: this is just a shortest-path problem with a funny metric, throw Dijkstra at it. Discretise the ocean into a grid of nodes, weight each edge by the time to traverse it = length over local achievable speed, find the cheapest path. Let me try that and see where it breaks.

It breaks immediately, and it's worth being precise about why, because the precise reason is the whole game. The weight of an edge — the time to sail from node i to node j — is length over V, and V depends on the weather at i and j. But the weather there is not a fixed number. It's H₁/₃ and θ *at the moment I'm passing through*, and the field is moving and evolving over the days the crossing takes. The storm that's sitting on edge (i,j) today will be a thousand kilometres away in two days. So the cost of edge (i,j) depends on *when* I traverse it. And when I traverse it depends on the whole route up to that point — my clock at node i is the sum of all the edge times before it. The edge cost and the route are coupled through time. Pricing each edge by "the weather there" is meaningless until I say *at what clock value*, and that clock is an output of the route, not an input. So a static-weight Dijkstra is planning against a frozen snapshot of a sea that won't be there when the ship arrives. Wall.

Let me make the time-dependence explicit instead of pretending it away. Let c_arc(i, j, t) be the time to sail i → j *departing i at clock time t*, computed from the weather at time t through my speed-loss curve. Now the cost carries a t. The thing I want is the earliest I can arrive at the finish. Write it as a recursion: let f*(i, t) be the minimum remaining time to reach the finish starting from node i at clock t. Then

  f*(finish, t) = 0,
  f*(i, t) = min over neighbours j of [ c_arc(i, j, t) + f*( j, t + c_arc(i, j, t) ) ].

That second argument is the point — after I take the edge, my clock has advanced by exactly the edge time, so the subproblem I hand to j is evaluated at t + c_arc(i,j,t), the new weather. This is a dynamic program, and because the cost is time-indexed it's a *time-dependent* shortest path, not a static one. That's the right object. Bellman's principle holds: the tail of an optimal route is itself optimal from the node-time pair it starts at.

But how do I actually *solve* this without enumerating routes? Going backward from the finish is awkward here, because I don't know, ahead of time, at what clock values I'll be querying f*(j, ·) — the relevant clocks are themselves determined by the forward progress, and the weather is only meaningfully indexed forward from the known departure time. So let me run it forward. Start at A at the known departure time. Sweep outward, and at every node keep the *earliest clock time* at which I can reach it. That's a label, and propagating earliest-arrival labels outward, always settling the smallest, is exactly Dijkstra — but now on the time-dependent graph, where I look up the weather at a node's settled arrival time before I relax its outgoing edges. Forward, label-setting, on a graph whose edge costs I evaluate at the arrival clock. Good — that's correct and it honours the coupling. But "grid over the whole ocean, settle every node" is a lot of nodes, most of them nowhere near where the route goes. Let me see if the structure of *this* problem lets me be smarter than a generic grid sweep.

Here's the structure. I'm propagating earliest-arrival times outward from A. Consider all the places I can be after exactly one time step Δt: from A, point the ship in a fan of headings, and for each heading apply the achievable speed in the local weather and step forward V·Δt. That gives a little arc of reachable points. After two steps, from each of those, fan again — a wider cloud. The boundary of "everywhere reachable within k·Δt" is a curve. Call it an isochrone: the contour of farthest reach in equal sailing time. Propagating earliest-arrival labels *is* sweeping this curve outward step by step. So instead of settling individual grid nodes I can carry the whole frontier as a curve and advance it: from the current isochrone, fan headings off every point, step Δt, and the outer boundary of the resulting cloud is the next isochrone. Keep going until a front reaches B; the number of steps to get there is the passage time, and I read the route off by remembering which front point each new point came from and walking the parents back. This is a beautiful reframing — the abstract DP frontier becomes a concrete moving wavefront, and it's tied to the physics: the front bulges ahead where seas are kind and lags where the ship is fighting a head sea.

Now the cloud. After a couple of steps the set of reachable points is fat — every front point spawns a fan, fans overlap, the count multiplies by the number of headings each step. If I keep all of them I'm back to exponential blowup, worse than the grid. But I don't need the interior. Think about two candidate points that lie in nearly the same direction out from the start, but one has pushed farther along that bearing than the other. The nearer one is dominated: the front already encloses it, meaning it was reached at least as early via the boundary, so it cannot possibly lie on a route that arrives at B sooner than one through the farther point — anything I could do onward from the near point, I could do at least as well starting from the far point on the same bearing, having spent no more time to get there. So within a narrow direction sector, only the farthest-advanced point can be on an optimal route; every interior point in that sector is throwaway. That's the pruning rule, and it's the thing that collapses the fat cloud back to a thin curve: slice the candidates by bearing into sectors, and in each sector keep exactly one survivor — the one that has travelled farthest. James did this by eye on the chart ("the longest distance among the routes in each sub-sector"); I'll do it by binning the candidates by their azimuth from the start and taking, per bin, the maximum distance travelled.

Let me sanity-check that this pruning is the same operation as "settle the frontier and drop dominated nodes" in the time-dependent Dijkstra, because if it isn't I've broken correctness. In label-setting, once a node has its earliest arrival it's settled and any later, slower way of reaching the same place is discarded. Here a "place" is resolved at the granularity of a direction sector, and "earliest arrival" is "farthest reached for the same number of steps" — same step count means same elapsed time, so farthest-for-equal-time is exactly earliest-for-equal-place. Keeping the max-distance point per sector and dropping the rest is keeping the earliest-arrival label per cell and discarding the dominated ones. It lines up. Good — the isochrone-with-sector-pruning is the front-propagation form of the time-dependent shortest path, not a heuristic departure from it. (The continuous calculus-of-variations attack would chase the same optimum by solving Euler–Lagrange for the least-time track, but it leans on second derivatives of the speed field, and over a multi-day integration the numerical error in those second derivatives compounds; the discrete front sidesteps the derivatives entirely, which is one more reason to like it.)

Now, a real wrinkle shows up the moment I try to *program* this rather than draw it. By hand the isochrone is a clean smooth curve and "take the outer boundary" is obvious to the eye. Coded as a raw geometric construction — connect the outermost candidate points into a polygon — the front can fold over itself. Where the speed field is strongly sheared, neighbouring front points advance by very different amounts and in different directions, and the boundary polygon crosses itself, producing spurious self-intersecting loops — "isochrone loops." Those loops corrupt the boundary and then corrupt the route I backtrack through them. I need a representation of the front that *can't* fold. The fix follows straight from the pruning rule if I make the sectors structural instead of drawing a free polygon: set up a grid aligned to the reference direction — lay it along the great circle from A to B, with lanes running perpendicular to the great circle — and index every candidate point by which cross-track lane it falls in. Then "keep the farthest per sector" becomes "keep, in each lane, the single sub-point that has advanced farthest along the great circle." One survivor per lane, indexed by a structured grid coordinate, so the front is a list of lane-survivors that is ordered by lane and physically cannot self-cross. The loops are gone, and it's the same dominated-point prune, now made robust. And I want the grid to *float* — re-laid around the current front each step rather than fixed once — so the lanes track where the front actually is and the resolution stays concentrated on the live part of the route. That the great circle is the calm-water optimum is exactly why centering the lanes on it is right: the weather-optimal route is a perturbation of the great circle, it lives in a band around it, so I spend my grid where the route is and bound the search width instead of tiling the whole ocean.

Let me pin down the heading fan, because it's a real design choice with a failure mode on each side. I fan a set of courses around the great-circle bearing toward the destination — say segments evenly spaced in [−Δ, +Δ] about that bearing. Too few headings and I can't represent the detour needed to dodge a storm or catch a following sea — the optimal evasive turn falls between my discrete courses and I miss it. Too many and each front point spawns a huge fan, the candidate cloud explodes before pruning, and even though pruning culls it I've paid to generate it. So: enough segments to resolve a useful turn, centered on the bearing to the destination so the resolution sits where the route is heading, width Δ wide enough to allow a meaningful detour but not the whole compass. Centering on the *current* bearing to the destination (recomputed each step from the front point) rather than a fixed bearing keeps the fan pointed sensibly even after the route has been pushed off the great circle by weather.

So the time-minimising algorithm is settled: front = {A} at the departure clock; repeat — fan headings off each front point, step one Δt using the achievable speed in the weather at the current clock, drop any leg that crosses land or shallow or unsafe seas, bin the survivors by cross-track lane / bearing and keep the farthest-advanced per bin, advance the clock by Δt — until a front comes within one step of B; then backtrack the parent pointers to recover the route. Equal time steps, so the front is literally an isochrone.

Now bring fuel back in, because in practice fuel is often the real objective and time is a constraint. Two ways to handle it. One: keep stepping in equal *time*, hold the engine fixed, get the min-time route, and call its fuel the cost — but that doesn't minimise fuel, it just reports the fuel of the fastest route. The cleaner move is to change what each step holds constant. Instead of advancing every front point by an equal time Δt, advance every front point by an equal *amount of fuel* Δfuel. Then for a given front point and heading, the fuel rate in the local weather sets how long that fuel lasts: Δtime = Δfuel / fuel_rate(heading, weather), and the distance covered is V·Δtime. Now the front is a contour of equal fuel burned — an "isofuel" line — rather than equal time. Everything else is identical: fan, step, prune the dominated points per lane, advance. And min-time drops out as the special case where the per-step resource *is* time. (There's an even more general version where the equal-cost surface lives in three dimensions, position and time together, so the engine power can vary along the route — an isopone, a surface of equal fuel in (position, time) space. It's the natural generalisation, but it's notoriously hard to reason about operationally, and the plain front in one resource at a time is what's tractable and intelligible, so I'll build that.) Whichever resource I step in, after the front reaches the destination I have many complete routes that got there; I pick the survivor with the least total of the objective — minimum total fuel, or minimum total time — and backtrack it.

That's the method, end to end. Let me lay it down as code. I'll structure it the way the front propagation actually runs: a base class that owns the start/finish, the great-circle reference, the step loop and the pruning; a fuel-stepping subclass that defines what one Δfuel step does. Geodesic moves come from a geodesic library (direct: advance from a point along a course by a distance; inverse: distance and bearing between two points). The weather/ship model returns achievable speed and fuel rate from heading and local weather. Arrays are carried per step so the backtrack is just reading a column back through the stacked history.

```python
import numpy as np
from geovectorslib import geod          # geod.direct(lat, lon, azi, dist), geod.inverse(lat1, lon1, lat2, lon2)
from scipy.stats import binned_statistic

class IsoBased:
    """Front-propagation router. The front is carried as parallel arrays of
    candidate route-heads; one column = one surviving route so far."""

    def __init__(self, start, finish, start_time, ship, weather, constraints,
                 course_segments, course_inc_deg,         # heading fan: how many, how wide
                 prune_segments, prune_sector_deg_half,    # dominated-point binning
                 ncount, delta):
        self.start, self.finish = start, finish
        self.time = start_time
        self.ship, self.weather, self.constraints = ship, weather, constraints
        self.course_segments, self.course_inc = course_segments, course_inc_deg
        self.prune_segments, self.prune_half = prune_segments, prune_sector_deg_half
        self.ncount, self.delta = ncount, delta
        # great circle A -> B: the calm-water optimum and our reference frame
        self.gcr_course = geod.inverse([start[0]], [start[1]], [finish[0]], [finish[1]])['azi1']
        # per-step history; step 0 is the single departure point
        self.lats = np.array([[start[0]]]); self.lons = np.array([[start[1]]])
        self.full_dist = np.array([0.0])   # distance made good per surviving route

    def define_courses(self):
        """Fan headings around the current bearing to the destination, for every
        front point. Few segments miss the detour; too many explode the cloud."""
        n = self.lats.shape[1]
        bearing = geod.inverse(self.lats[0], self.lons[0],
                               np.repeat(self.finish[0], n), np.repeat(self.finish[1], n))['azi1']
        # replicate every front point across the heading fan
        for a in ('lats', 'lons', 'full_dist'):
            setattr(self, a, np.repeat(getattr(self, a), self.course_segments + 1, axis=-1))
        deltas = np.linspace(-self.course_segments/2 * self.course_inc,
                             +self.course_segments/2 * self.course_inc, self.course_segments + 1)
        return np.repeat(bearing, self.course_segments + 1) - np.tile(deltas, n)

    def step(self, courses):
        """One propagation step: achievable speed in the local weather, advance
        the front. The base step is equal TIME (classic isochrone)."""
        wx = self.weather.get(self.lats[0], self.lons[0], self.time)
        speed = self.ship.speed(courses, wx)                 # speed-loss curve
        dist = speed * self.delta                            # equal-time: delta is a time step
        return courses, dist

    def move(self, courses, dist):
        nxt = geod.direct(self.lats[0], self.lons[0], courses, dist)
        self.lats = np.vstack((nxt['lat2'], self.lats))
        self.lons = np.vstack((nxt['lon2'], self.lons))
        self.full_dist = self.full_dist + dist               # progress made good

    def prune(self):
        """Collapse the candidate cloud to the front: bin by bearing-from-start
        and keep, per bin, the single farthest-advanced candidate. That is the
        dominated-point rule = settling the time-dependent-Dijkstra frontier,
        and binning into structured lanes is what stops the front self-crossing
        into 'isochrone loops'."""
        bearing = geod.inverse(np.repeat(self.start[0], self.lats.shape[1]),
                               np.repeat(self.start[1], self.lons.shape[1]),
                               self.lats[0], self.lons[0])['azi1']
        axis = geod.inverse([self.start[0]], [self.start[1]],
                            [self.finish[0]], [self.finish[1]])['azi1']     # center bins on the gcr
        edges = np.sort(np.linspace(axis - self.prune_half, axis + self.prune_half,
                                    self.prune_segments + 1))
        stat, edges, _ = binned_statistic(bearing, self.full_dist, statistic=np.nanmax, bins=edges)
        idxs = [np.where(self.full_dist == s)[0][0] for s in stat if s > 0]   # farthest per lane
        idxs = list(set(idxs))
        self.lats = self.lats[:, idxs]; self.lons = self.lons[:, idxs]
        self.full_dist = self.full_dist[idxs]

    def reached(self):
        d = geod.inverse(self.lats[0], self.lons[0],
                         np.repeat(self.finish[0], self.lats.shape[1]),
                         np.repeat(self.finish[1], self.lons.shape[1]))['s12']
        return np.any(d < self.delta * self.ship.max_speed)

    def run(self):
        for _ in range(self.ncount):
            courses = self.define_courses()
            courses, dist = self.step(courses)
            ok = ~self.constraints.violated(self.lats[0], self.lons[0], courses, dist)
            courses, dist = courses[ok], dist[ok]
            # (carry the same mask over the per-step arrays in a full implementation)
            self.move(courses, dist)
            if self.reached():
                break
            self.prune()
            self.time = self.time + self.delta
        return self.backtrack()

    def backtrack(self):
        # pick the surviving route best on the objective, read its column back
        best = int(np.argmin(self.objective_total()))
        return list(zip(self.lats[:, best], self.lons[:, best]))


class IsoFuel(IsoBased):
    """Step in equal FUEL instead of equal time. Each step burns delta_fuel;
    the local fuel rate sets how long that lasts, hence the distance. Min-time
    is the special case where the stepped resource is time itself."""

    def __init__(self, *args, delta_fuel, **kw):
        super().__init__(*args, **kw)
        self.delta_fuel = delta_fuel

    def step(self, courses):
        wx = self.weather.get(self.lats[0], self.lons[0], self.time)
        speed = self.ship.speed(courses, wx)
        fuel_rate = self.ship.fuel_rate(courses, wx)         # kg/s in this weather, this heading
        delta_time = self.delta_fuel / fuel_rate             # equal-fuel -> variable time
        dist = speed * delta_time
        self.time_used = delta_time
        return courses, dist

    def objective_total(self):
        # least total fuel among the routes that reached the destination
        return self.total_fuel_per_route()
```

Stepping back through the causal chain: the achievable speed depends on the sea state and heading, the sea state is a field that moves while the ship crosses, so the cost of any leg depends on when it's sailed and a static shortest path prices the wrong sea; writing the cost as time-indexed turns it into a time-dependent shortest path whose forward dynamic program propagates earliest-arrival labels outward; that frontier, made concrete, is an isochrone — fan headings off the current front, advance one resource-step using the local achievable speed, and the outer boundary is the next front; the candidate cloud is collapsed by dropping dominated points — within a bearing sector only the farthest-advanced survives, which is exactly settling the Dijkstra frontier — and binning those survivors into structured cross-track lanes around the great circle keeps the front from folding into spurious loops; step in equal time for minimum passage time, in equal fuel for minimum fuel, and when the front reaches the destination backtrack the best survivor to recover the route.
