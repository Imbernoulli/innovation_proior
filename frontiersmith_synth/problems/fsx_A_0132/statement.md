# Selene Station: Streaming Cargo Consolidation

## Story

Selene Station is a lunar habitat fed by an automated delivery pipeline. Supply
**canisters** drop out of the pipeline **one at a time** as a live stream — the
station has **no look-ahead** at what canister will arrive next. Each canister has
an integer stowage cost `s` (combined mass + volume units). The crew stows
canisters into pressurized **storage modules**, each with a fixed usable budget
`C`. A module may hold any set of canisters whose costs sum to at most `C`.

Powering and pressurizing a module is expensive, so the station wants to
consolidate the whole stream into **as few modules as possible**.

Placement is **online and irrevocable**. A FIXED streaming simulator processes the
canisters in arrival order. For each arriving canister it stows it into one already
open module (chosen by a *priority rule*) or, if no open module has room, powers a
fresh module. A canister is never moved once stowed, and the stream is never
reordered. This is online 1-D bin packing: canisters = items, module budget = bin
capacity, modules powered = bins used, which we **minimize**.

## Your job: supply the priority rule

You do **not** output an assignment (you never see the future stream inside the
simulator). You output the **priority rule** as a weight vector `w = [w0, w1, w2, w3]`
over a FIXED feature basis. When a canister of cost `s` arrives, the simulator, for
every open module with remaining budget `res >= s`, computes

```
score(module) = w0*phi0 + w1*phi1 + w2*phi2 + w3*phi3
    phi0 = 1.0                       # bias / constant
    phi1 = res / C                   # how empty the module is BEFORE stowing
    phi2 = (res - s) / C             # leftover budget fraction AFTER stowing
    phi3 = ((res - s) / C) ** 2      # squared leftover (nonlinear)
```

and stows the canister into the module with the **highest** score (ties broken by
**lowest module index**). If no open module fits, a fresh module is powered.

Classic dispatch rules are special cases of `w`:
- **best-fit** `w = [0, 0, -1, 0]` — minimize leftover budget.
- **worst-fit** `w = [0, 0, 1, 0]` — maximize leftover budget (the weak baseline).
- **first-fit** `w = [1, 0, 0, 0]` — all modules tie, lowest index wins.

Combining the features (leftover, current emptiness, and the nonlinear squared
term) lets you express richer consolidation policies.

## Candidate program contract (isolated stdin -> stdout)

Your program reads ONE JSON object from stdin and writes ONE JSON object to stdout.

Input (PUBLIC instance):
```json
{"name": "selene101", "capacity": 40, "n": 90, "items": [12, 3, 27, ...]}
```
- `capacity` (int C): usable budget per module, `1 <= s <= C` for every item.
- `n` (int): number of canisters.
- `items`: the full arrival sequence, provided for your analysis. The simulator —
  not your program — applies your weights online in arrival order.

Output (answer):
```json
{"weights": [0.0, 0.0, -1.0, 0.0]}
```
- exactly **4** finite real numbers.

Any malformed output (wrong type, wrong length, a non-finite weight `nan`/`inf`, a
crash, a timeout, or non-JSON) scores **0.0** on that instance.

## Objective and scoring

Deterministic, no wall-time. For each instance with items and budget `C`:

- `q_lb   = ceil(sum(items) / C)` — the L1 lower bound (generally unreachable).
- `q_base = ` modules used by the **worst-fit** rule — the weak reference.
- `q_cand = ` modules used by **your** rule under the fixed simulator.

Per-instance normalized score (worst-fit -> 0.1, L1 ideal -> 1.0):
```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

The final `Ratio` is the mean of `r` over a fixed family of 14 instances (uniform,
medium, bimodal, heavy-tailed, and Weibull cost distributions across a range of
`n` and `C`, including larger held-out instances). Because L1 is a loose lower
bound, even excellent online rules stay strictly below 1.0 on most instances —
there is real headroom, no easy optimum, and several viable strategies.

## Strategy ladder (for reference)

- **worst-fit** (`w=[0,0,1,0]`): spreads canisters into the emptiest module,
  wasting budget — reproduces the weak baseline, ~0.1.
- **first-fit** (`w=[1,0,0,0]`): stow into the lowest-index module that fits;
  reuses earlier gaps.
- **best-fit** (`w=[0,0,-1,0]`): stow into the tightest module that still fits,
  closing modules densely — the strongest simple rule here.
- **tuned / combined**: blend leftover, emptiness, and the squared term to search
  for a policy that shaves further modules on some distributions.
