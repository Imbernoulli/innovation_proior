# Ashline Refinery Bootstrap

A fuel depot sits on a volcanic flat. Its refinery turns ore into fuel, but the
only fuel the trucking fleet can burn is fuel the refinery has already made.
The depot starts with a small fuel tank `F0`, a refinery of capacity `C0`, a
fleet of `K` identical trucks, and `M` ore mines. You plan `T` days. Maximize
the fuel in the tank at the end of day `T`.

## Entities
- **Mine `i`** has ore stock `S_i` at distance `d_i` km (one way).
- **Trucks**: payload capacity `P` ore, speed `v` km/day. A trip departing at
  the start of day `t` to mine `i` with payload `p` (`0 < p <= min(P, S_i)`)
  burns `kappa * d_i * (2 + p/P)` fuel **paid from the tank at departure**
  (outbound leg empty, return leg loaded: per-km burn grows with the payload
  fraction). The round trip takes `tau_i = max(1, ceil(2*d_i/v))` whole days;
  truck and ore return at the start of day `t + tau_i`. A truck on the road
  cannot be dispatched. Ore arriving after day `T` is lost (fuel is still
  spent).
- **Refinery**: capacity `C` (start `C0`, hard ceiling `Cmax`). On each day you
  pick a run level `q` (`0 <= q <= min(ore stockpile, C)`); it consumes `q`
  ore and produces `beta * C * q / (q + C)` fuel at end of day. The conversion
  curve is **concave**: fuel per ore falls as `q` rises toward `C`.
- **Upgrades**: on day `t` you may pay `cu` fuel per unit to add capacity
  (any real amount, applied at start of day, total never above `Cmax`).
  Capacity raises both the daily ceiling and the per-ore yield.

## Day order (the checker simulates exactly this)
1. Trips returning today deliver ore; their trucks become idle.
2. Today's upgrades: fuel is paid, capacity increases.
3. Today's departures, in the order listed: checks, fuel paid, ore reserved.
4. Refining: `q` ore consumed, fuel credited.

The tank may never go negative; every check is strict (see Feasibility).

## Input (stdin)
```
T K M
F0 C0 Cmax cu beta v
P kappa
d_1 S_1
...
d_M S_M
```
All values positive reals except `T, K, M` (positive integers).

## Output (stdout)
Whitespace-separated numbers, three sections:
```
nU
t u            (nU lines: day t in [1,T], upgrade amount u >= 0)
nR
t j i p        (nR lines: day, truck 0<=j<K, mine 0<=i<M, payload p)
nQ
t q            (nQ lines: day, run level q >= 0; unlisted days run q = 0)
```
Multiple records per day are allowed and apply in listed order. Counts must
match exactly; no extra tokens. All tokens must be finite numbers; day/truck/
mine fields must be integers in range.

## Feasibility (any violation scores 0)
Tank negative at any point; capacity above `Cmax`; payload above `P` or above
the mine's remaining stock; dispatching a truck that is still on the road; run
level above `min(stockpile, C)`; malformed/truncated/non-finite output.

## Objective and scoring
Maximize `F` = final tank. The checker computes `B`, the final tank of its own
do-little reference plan (truck 0 shuttles the nearest stocked mine at half
payload, half-throttle refining, no upgrades), and reports

`Ratio = min(1, 0.1 * F / max(1e-9, B))`

so the reference plan scores 0.1 and a plan roughly 10x better than it
saturates. Exact coefficients (`cu, beta, kappa, ...`) are in the input; the
mechanics above are exact.

## Constraints
`1 <= K <= 20`, `1 <= M <= 30`, `T <= 500`. Output up to 10^6 records. Time
limit 5 s, memory 512 MB. Scoring is fully deterministic.

## Example (illustrative)
```
T K M    = 6 1 1
F0 C0 Cmax cu beta v = 40 2 6 10 8 40
P kappa  = 10 0.3
mine     = d=20, S=30
```
Plan: day 1 send truck 0 to mine 0 with `p = 10` (cost `0.3*20*3 = 18`,
`tau = 1`, ore returns day 2); never upgrade; run `q = 2` on days 2..6 (within
stockpile and capacity).

Output:
```
0
1
1 0 0 10
5
2 2
3 2
4 2
5 2
6 2
```
Fuel: day 1 pays 18 (tank 22); days 2..6 each refine `8*2*2/(2+2) = 8`, so
the final tank is `F = 22 + 5*8 = 62`. The checker's reference plan on this
instance ends with `B = 6.666667`, so this plan scores
`Ratio = min(1, 0.1*62/6.666667) = 0.93`. On the real tests the reference is
far harder to beat by brute hauling: the winning margin comes from reinvesting
early and choosing when to stop growing and start harvesting.
