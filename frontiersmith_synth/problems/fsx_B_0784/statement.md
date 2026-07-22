# Gantry Heads on a Mural Rail

## Problem
`H` print heads share one straight rail with integer positions `0..L`. `N` colored
strokes sit at distinct rail positions; stroke `i` is at position `p_i` with color
`c_i`, and every stroke must be painted by exactly one head (a **coverage
partition** of the strokes across the heads). Each head starts at a given position
with no color loaded and can repeat, in any order, three unit operations:

- `M dx` (`dx = +1` or `-1`): move one rail position; the head must stay in `[0,L]`.
- `S c`: load color `c` into the head (a **fixed setup**, always costs `S` ticks,
  regardless of proximity to other heads).
- `P`: paint the stroke sitting at the head's current position. This is only legal
  if a stroke exists there, it has not been painted yet, and the head's currently
  loaded color equals that stroke's color.

**Proximity throttle:** whenever a head *begins* a move, look at every head's
current rail position at that instant. If any other head is within Manhattan
distance `D` of the moving head, that single move takes `2` ticks instead of `1`
(the vibration coupling slows it to half speed); otherwise it takes `1` tick. A
head's position only changes when a move *completes* -- while a move is in
progress the head is still considered to stand at its pre-move position. Swap and
paint operations always cost `S` and `1` ticks respectively, unaffected by
proximity. All `H` heads execute their operation streams simultaneously against
one shared clock.

## Input (stdin)
```
L H D S K N
s_0 s_1 ... s_{H-1}
p_0 c_0
...
p_{N-1} c_{N-1}
```
`s_h` is head `h`'s starting position. Strokes are listed in ascending position
order; colors are integers in `1..K`.

## Output (stdout)
```
H
m_0
<m_0 ops for head 0>
m_1
<m_1 ops for head 1>
...
m_{H-1}
<m_{H-1} ops for head H-1>
```
Each op is `M dx`, `S c`, or `P`, one per line (whitespace is otherwise free).

## Feasibility
The reported head count must equal `H`. Every move must stay inside `[0,L]`.
Every paint must land on an unpainted stroke whose color matches the head's
currently loaded color. By the end of the simulation every one of the `N`
strokes must have been painted exactly once. Any violation scores `Ratio: 0.0`.

## Objective (minimize)
Let `F` be the **makespan**: the tick at which the last head finishes its
operation stream (a head with an empty stream finishes at tick 0). Minimize `F`.

## Scoring
The checker replays your operation streams tick-by-tick under the rules above to
get `F`. It also builds its own baseline `B`: a single head (head 0 only, all
other heads idle forever) visits every stroke **in the input's given order**,
moving directly to each one, swapping color only when it changes, and painting --
no throttling is possible in this baseline since only one head ever moves. Then
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
Matching the baseline scores about `0.1`; a plan ten times faster caps at `1.0`.

## Constraints
`H = 3` heads, `K = 4` colors, `10 <= L <= 60`, `N = L+1` (one stroke per rail
position), `D` and `S` are given per instance (small positive integers). Time
limit 5s; each case replays in well under a second.

## Example
Smallest case: `L=10, H=3, D=3, S=5, K=4, N=11`, starts `2, 5, 8`, colors cycle
`1,2,3,4,1,2,3,4,1,2,3` by position (every left-to-right sweep swaps almost
every stroke). Baseline `B` (one head, whole rail, given order) `= 52`. Handing
each color to one head (batch by color, ignore proximity) reaches `F = 39`
(`Ratio = 0.133`): fewer swaps, but all three heads cross the whole rail and
stay mutually throttled. Splitting the rail into three local territories (one
per head) and only THEN grouping each territory's strokes by color reaches
`F = 21` (`Ratio = 0.248`): territories stay `> D` apart so throttling nearly
vanishes, while local color runs still cut most of the swaps. Neither move
alone is optimal -- rail segment and color order must be chosen jointly.
