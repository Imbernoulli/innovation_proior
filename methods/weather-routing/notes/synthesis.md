# Synthesis — Ship weather routing (isochrone method)

## Pain point / research question
Find the fastest (or min-fuel) ocean route from A to B when achievable ship speed depends on
the *time-varying* weather field (waves/wind/current) the ship actually meets en route. A static
shortest path is wrong because the field the ship encounters at a node depends on WHEN it gets
there, and that depends on the route itself (the cost is time-dependent / non-FIFO-ish).

## Three-source pillars
- Primary: James 1957 "Application of Wave Forecasts to Marine Navigation" (US Navy Hydrographic
  Office) — first optimal-ship-routing study; the *isochrone* construction (time fronts = farthest
  reachable boundary at equal sailing time). Hagiwara & Spaans 1987 "Practical weather routing of
  sail-assisted motor vessels" (J. Navigation) + Hagiwara 1989 PhD (TU Delft) — the *modified*
  isochrone that fixes computer-implementation "isochrone loops" via a recursive forward technique
  on a great-circle-referenced grid.
- Background:
  - Speed-loss model: V = (a·n + b) − (c·H₁/₃ + d·H₁/₃²)·f(θ), f(θ)=0.75·exp(−0.65θ²)+0.25,
    θ relative wave direction (0=head sea) [Kobe thesis, Sasa/Shoji model]. Calm-water speed minus
    a wave-induced loss that is largest in head seas, near-zero following seas. Newton's law form
    (M+m₁₁)dV/dt = T − R, R = still-water + added resistance in waves/wind [Kobe eq 5].
  - Min-time routing on time-varying cost field; isochrone = reachable front at time t.
  - Time-dependent shortest path / DP: Bellman f*(i,t)=min_j[c_arc(i,j,t)+f*(j,t+c_arc(i,j,t))],
    cost c_arc depends on departure time t [Allsopp 1998 / uncertainty paper eq 1]; 2DDP/3DDP
    [Bijlsma 1975, De Wit 1990, Calvert 1991, Wei&Zhou 2012]; isopone extension [Klompstra/Spaans].
  - Calculus-of-variations roots [Bijlsma] — continuous min-time, brittle 2nd-order derivatives.
  - Dijkstra/A* framing [Padhy 2008; WRT isofuel pruning ~ admissible "keep farthest per bin"].

## Isochrone construction (the core)
- Reference frame: great circle A→B. Define "sectors" = sub-divisions of heading fanned around the
  GC course; the isochrone is built sector by sector.
- Step: from each point on isochrone I_k (reached at time t_k = k·Δt), fan out a set of headings;
  for each heading compute achievable speed V from local weather via speed-loss model; advance
  distance V·Δt to a candidate point. The union of candidate points is a cloud; the new isochrone
  I_{k+1} is its outer boundary.
- Pruning (dominated points): partition candidates by *direction sector* (azimuth from start, or
  course bin). Within each sector keep only the point with maximum distance/progress along the GC
  — that is the only non-dominated point in that sector; all interior points are dominated (a later
  isochrone could reach them at least as early via the boundary). This is exactly James's "longest
  distance among multiple routes in each sub-sector". WRT does this with binned_statistic(...,
  statistic=nanmax) over course/larger-direction bins.
- Repeat until an isochrone encloses (reaches within one step of) the destination; backtrack
  parent pointers to recover the route.

## Modified isochrone (Hagiwara/Spaans) — what it fixes
- Original isochrone (geometric, hand drawn on charts) produced "isochrone loops" / self-crossing
  fronts when computerized, because the boundary was constructed geometrically and could fold.
- Fix: recursive forward technique on a *floating grid* aligned to the great circle. Lay sub-grid
  perpendicular to GC; index candidate sub-points by their GC-cross-track cell; keep the farthest
  sub-point per cell (one survivor per grid lane). This both removes loops and is the clean
  dominated-point prune. Floating = grid recomputed each step around current front [Lin 2013].

## Isofuel variant (WRT canonical code)
- Instead of equal-TIME steps (isochrone), advance equal-FUEL steps: each step consumes Δfuel;
  Δtime = Δfuel / fuel_rate(course, weather); dist = V·Δtime. Front = "isofuel" line. Min-time is
  recovered as the special case where the step cost is time. Prune: keep max full_dist_traveled per
  course/direction bin. Final pruning: argmin total fuel (or time) among reached routes.

## Time-dependent-graph equivalence (the unification)
- Discretize domain into grid nodes; arc (i→j) cost c_arc(i,j,t) = travel time from i to j departing
  at time t, using the weather at time t. Because c depends on t, this is a TIME-DEPENDENT shortest
  path; the forward Bellman recursion above is DP. The isochrone method is precisely the
  label-setting / front-propagation (Dijkstra-like) solution of this: an isochrone I_k is the set of
  nodes whose earliest-arrival label equals k·Δt, and pruning-by-sector is the "settle the frontier,
  drop dominated nodes" operation. The grid-DP (2DDP/3DDP) is the same recursion on a fixed grid;
  the isochrone is the same recursion with the frontier represented as a continuous boundary.

## Design-decision → why
- Great-circle reference frame: GC is the optimum in calm water, so the optimal weather route stays
  near it; centering sectors/grid on GC concentrates resolution where the route lives and bounds the
  search width. Alt (full lat/lon grid) wastes nodes far from the route.
- Heading fan / course segments: discretizes the continuous heading control; ± sector around GC
  course. Too few = miss good detours around weather; too many = blow up the front. WRT:
  course_segments + 1 evenly spaced in [−seg/2·inc, +seg/2·inc].
- Pruning by max-progress per sector: the non-domination argument — within a thin direction sector
  only the farthest-advanced front point can be on the optimal route; keeping interiors is wasted
  work and causes the front to balloon combinatorially. This is what makes it tractable vs full DP.
- nanmax not sum: keep the single best per bin (label-setting), not aggregate.
- Equal-fuel vs equal-time stepping: equal-fuel directly targets fuel objective and gives even fuel
  resolution; equal-time gives even time resolution (classic isochrone). Choice = which objective.
- Speed-loss f(θ): Gaussian-ish in relative wave angle — big loss head seas (θ≈0), ~0.25 floor
  following — captures the dominant physical asymmetry cheaply without solving seakeeping in the loop.
- Forward (not backward) recursion: forward lets voyage-time at each state be *discovered* as the
  front advances (needed because weather is time-indexed and only known forward from departure).

## Code structure (WRT)
- RoutingAlg base: start/finish, ncount steps, terminate→RouteParams.
- IsoBased: define_courses (fan headings around GC, branch arrays to N×(seg+1)); execute_routing loop
  {define_courses_per_step → estimate_fuel_consumption (Ship gives speed+fuel from weather) →
  move_boat (advance) → check_constraints (land/depth/wave) → update → pruning_per_step}.
- pruning: bin candidates by course (courses_based) or azimuth-from-start (larger_direction) or
  branch; keep argmax full_dist_traveled per bin (get_pruned_indices_statistics).
- IsoFuel(IsoBased): get_delta_variables_netCDF → delta_time=delta_fuel/fuel_rate, dist=bs·delta_time;
  final_pruning: argmin total fuel.
- Backtrack: lats_per_step/lons_per_step stacked per step; terminate picks final index, reads column.
