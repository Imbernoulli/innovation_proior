# The Sighing Caravanserai: Vent One Loop, Seal the Rest

## Problem
A desert caravanserai is a row of `W` vaulted bays. Bay `c` (`1..W`) has height
`h_c`, a heat load `q_c` (cookfires, bodies, sun-struck stone) and occupancy
`o_c` (`1` = guests quartered here, `0` = stores). `D` fixed open doorways
connect bays; doorway `i` joins bays `c_i` and `d_i` at sill height `z_i` with
conductance `g_i`. There are also `M` candidate **vents** — wall or roof cuts
the masons could open; vent `j` sits in bay `C_j` at height `Z_j` with
conductance `G_j`.

You may cut open **at most `B`** of the `M` vents. Every uncut vent stays
sealed. Choose the set that maximizes the fresh-air throughflow delivered to
occupied bays once the night air settles into steady buoyancy-driven
circulation.

## The physics (exactly how the checker scores you)
All arithmetic is exact rational. Outdoor air has pressure datum `0` and
freshness `1`.

1. **Warmth.** Let `open(c)` be the total conductance of all doorways and cut
   vents incident to bay `c`. The bay's warmth is `t_c = q_c / (k0 + open(c))`
   — venting a bay cools it.
2. **Pressure and flow.** Air pressure in bay `c` at height `z` is
   `p_c(z) = pi_c + t_c * z` (a warm column presses outward near its top, inward
   near its base); outdoor pressure is `0` at every height. A doorway
   `(c,d,z,g)` carries `f = g * (p_c(z) - p_d(z))` (positive = from `c` to `d`);
   a cut vent `(c,z,g)` carries `f = g * p_c(z)` (positive = outward). The
   references `pi_c` solve steady conservation: the net flow out of every bay
   is zero. With at least one cut vent this linear system has a unique
   solution; with none, all flows are zero.
3. **Freshness.** `In_c` = total flow entering bay `c`; `src_c` = the part
   arriving from outdoors (freshness 1). Every bay-to-bay passage credits only
   `7/8` of the transported freshness (mixing with room air):
   `In_c * phi_c = src_c + (7/8) * sum_{d->c} f * phi_d`. A bay with no inflow
   has `phi_c = 0`.
4. **Objective.** `F = sum` over occupied bays of `phi_c * In_c` — the
   freshness-credited air delivered to guests. Maximize `F`.

`F` is **not** monotone in the vent set. Extra openings short-circuit the
stack effect: they equalize pressures low in the hall, drain its warmth, and
can strictly *reduce* throughflow. Circulation wants one tall pressure
imbalance — a low inlet feeding a high outlet — and restraint elsewhere.

## Input (stdin)
```
W B k0
h_1 ... h_W
q_1 ... q_W
o_1 ... o_W
D
c_i d_i z_i g_i        (D lines; bays 1-indexed)
M
C_j Z_j G_j            (M lines; bay, height, conductance)
```

## Output (stdout)
```
k
j_1 j_2 ... j_k
```
`k` = how many vents you cut (`0 <= k <= B`), then `k` **distinct** vent
indices in `1..M` on one line, any order. Exactly `k+1` whitespace-separated
tokens; no other tokens.

## Feasibility
Score 0 unless: `k` in range; every index a distinct integer in `1..M`; the
token count matches exactly. Non-numeric or extra tokens score 0.

## Scoring
The checker itself cuts only vents `1` and `2` (always the great-hall floor
mouth and its roof lantern) and computes the baseline `B0 > 0` from the same
model. Then
```
sc = min(1000.0, 100.0 * F / B0)        Ratio = sc / 1000.0
```
The minimal hall loop scores exactly `0.1`; ten times its throughflow caps at
`1.0`.

## Constraints
- `4 <= W <= 9`, `1 <= B <= M <= 40`, `1 <= k0 <= 8`, `6 <= h_c <= 60`,
  `0 <= q_c <= 240`; doorways keep the bays connected as a row; all heights,
  sill heights (`0 <= z <= h_c`) and conductances are positive integers
  (`g <= 32`).
- Time limit 5s, memory 512m. Instance files are small; the difficulty is the
  non-monotone response, not the size.

## Example (toy, not a real test)
`W=1, B=3, k0=4`; `h=[40]`, `q=[100]`, `o=[1]`; `D=0`; `M=3` vents:
`(1,0,4)`, `(1,40,4)`, `(1,10,16)` — floor inlet, roof outlet, and a big
mid-height window.

- Cutting `{1,2}`: warmth `t = 100/12 = 25/3`, conservation gives
  `pi = -500/3`, floor inflow `= 4 * (500/3 - 0) = 2000/3`. With `phi = 1`,
  `F = 2000/3 ≈ 666.67`. Since vents 1,2 are the baseline pair,
  `Ratio = 0.100000`.
- Cutting `{1,2,3}`: the big window cools the bay (`t = 25/7`) and pressures
  equalize; now the floor *and* the window both leak air in, total inflow
  `8000/21 ≈ 380.95`, so `F = 8000/21` and `Ratio = 0.1 * 4/7 ≈ 0.057143`.

Cutting **more** scored **worse** — that is the whole problem.

## Notes
Scores are exact rationals behind the scenes; no randomness, timing, or
tolerance is involved anywhere. Same submission, same score, forever.
