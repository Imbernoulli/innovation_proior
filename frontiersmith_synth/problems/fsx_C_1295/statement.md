# Robots That Must Not Deadlock: Warehouse Fleet Reservation Discipline

## Setting

A warehouse is a tree: a **depot** node `0` (unlimited capacity: orders start
and end here, no charging) connects to a **corridor spine** of single-lane
aisle nodes `1..L` (edges `(0,1),(1,2),...,(L-1,L)`). Off some spine nodes
hang dead-end spurs: **shelf** nodes (dock capacity 1 -- one robot can be
docked there at a time) and **charging** nodes (dock capacity `cap`, regen
`rate` battery per tick spent stationed there). Every edge in the whole tree
-- including the spurs -- can carry **at most one robot per tick**, in either
direction: this is the single-lane aisle.

`R` robots start at the depot, full battery `B_max`, tote capacity `K`. Every
move costs 1 battery; a robot at 0 battery cannot move again unless it is
sitting on a charge node dwelling. `M` orders each live at one shelf and have
a release time; a robot docked at a shelf **at the exact tick it arrives**
claims as many currently-released, unclaimed orders there (ascending order id)
as its free tote capacity allows. Arriving at the depot instantly delivers
(and permanently scores) every order it is carrying.

## Your submission: an offline fleet plan, not a live controller

You submit, once, for the whole horizon `T`:
```json
{"priority": [permutation of 0..R-1],
 "routes": [ [ {"node": <id>, "hold": <int>>=0}, ... ], ... ]}
```
`routes[i]` is robot `i`'s ordered stop list (by robot index). `priority`
says in which order the evaluator's fleet simulator **resolves** the robots.

The simulator processes robots **one at a time, fully, in priority order**.
For each stop, a robot walks the unique tree path from where it is toward the
target: at every tick it either advances one edge (if that edge is unused
this tick AND, for a shelf/charge target, the destination dock has a free
slot at the next tick) or **waits in place** (edges/docks already claimed by
an *earlier*-priority robot's finished plan are permanent; no collisions are
ever produced by construction). If a move would need a battery unit it does
not have, that stop is abandoned -- the robot is stuck wherever it got to and
tries its next stop (usually also failing, from then on). After arriving, the
robot runs its arrival action (pickup / deliver), then dwells `hold` extra
ticks in place (a charge node keeps regenerating battery, capped at `B_max`,
for every tick -- forced-wait or explicit `hold` -- the robot spends there).
**Claim priority is the same as movement priority**: a higher-priority
robot's *entire* itinerary (however late its ticks) resolves before any
lower-priority robot's first pickup attempt at a shared shelf.

## Feasibility

Reject (score 0 on this instance) if: your output is not a JSON object with
`priority` and `routes` keys; `priority` is not a list of length `R` that is
a permutation of `0..R-1`; `routes` is not a list of length `R`; any stop is
not `{"node": int, "hold": int}` with `node` a valid node id and
`0 <= hold <= 300`; any robot's stop list exceeds 400 stops;
anything non-finite/non-integral appears where an integer is required.

## Scoring

`orders_delivered` = number of distinct orders delivered to the depot within
`T` ticks. The evaluator also runs its own fixed, competent (but not
necessarily optimal) reference fleet policy on the same instance to get a
throughput `CAP`, scaled up for headroom so matching the reference cannot
saturate your score:
```
ratio = min(1, orders_delivered / CAP)
```
`CAP` is not given in the instance; you must reason about the mechanics
above, not pattern-match a revealed constant. The reported `Ratio` is the
mean of `ratio` over 10 deterministic, seeded instances -- some with few
robots on a short, lightly loaded corridor, others with many robots on a
long corridor with a tight battery budget and contested chargers/shelves;
a couple are held out at a larger scale to test that your policy is a
general recipe, not tuned to one layout.

## Example

`L=3`, depot 0, corridor 1-2-3, shelf `4` off node `2` (cap 1). One order at
shelf 4, released at tick 0. Robot 0 stops `[{"node":4,"hold":0},
{"node":0,"hold":0}]`: it walks 0->1->2->4 (3 ticks, 3 battery), claims the
order, walks back 4->2->1->0 (3 ticks), delivers it. A second robot with an
*identical* target and higher priority would claim the order first (dock
capacity 1 forces the loser to queue on node 2, which has unlimited standing
room) -- illustrating why priority order, not just distance, decides who
gets what.

## Constraints
`3 <= R <= 20`, `5 <= L <= 18`, `1 <= K <= 4`, `20 <= B_max <= 50`,
`50 <= T <= 110`, `6 <= M <= 60`. Time limit 5s, memory 512MB.
