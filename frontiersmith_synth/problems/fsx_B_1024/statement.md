# Campus CHP: Riding the Extraction-Condensing Envelope

A combined heat-and-power (CHP) plant serves a campus's electrical and
district-heating demand over `T` periods. Electrical power **cannot be
stored**: each period the plant must generate power exactly equal to that
period's demand `DP[t]`. Heat **can** be stored in a tank, or dumped
(wasted) for a small fee — heat is the only buffered coproduct.

## Envelope

At any fixed power output `P`, the plant's extraction-condensing turbine
allows heat output `Q` anywhere in a power-coupled interval `[L(P), U(P)]`
(a vertical slice of the plant's 2D feasible operating polygon):

```
U(P) = clip( min(Qcap, (Gamma - alpha*P) / beta), lower=0 )
L(P) = clip( delta*P - Epsilon,                    range=[0, U(P)] )
```

`Qcap, alpha, beta, Gamma, delta, Epsilon` are given in the input (fixed for
the whole horizon). `U` is non-increasing in `P` (rated-power steam-flow
limit shrinks the heat ceiling) and `L` is non-decreasing (a minimum
extraction floor appears at high power); the generator guarantees
`L(DP[t]) <= DQ[t] <= U(DP[t])` for every period, where `DQ[t]` is that
period's heat demand, so **feasibility is never the hard part**.

## Fuel cost

Power's own fuel draw is identical for every feasible plan (`DP[t]` is
exogenous demand, not a choice you make), so it is **not** part of the
graded objective. What you control is the heat output `Q`, and the
instantaneous heat-fuel cost of operating at `Q` in `[L(t), U(t)]` is

```
fuel(t, Q) = b*Q + kappa * (Q - L(t)) * (U(t) - Q)
```

The second term is a part-load inefficiency bump: **zero exactly at the two
envelope endpoints** `Q = L(t)` and `Q = U(t)`, and positive (worst) for `Q`
strictly between them. So at any single period, only the envelope's extreme
points are fuel-efficient — an interior operating point is provably worse,
however close it sits to the actual heat demand.

## Buffer and dumping

A tank starts at level `S_init` and evolves as

```
s[t] = s[t-1] + Q[t] - DQ[t] - dump[t]
```

with `dump[t] >= 0` (heat wasted, at a fee `dumpfee` per unit) and
`0 <= s[t] <= Cap` (tank capacity) required at **every** period. Note that
`Q[t]` need not equal `DQ[t]`: the tank (and dumping) is what lets you
produce at a vertex while still meeting the true demand trace on net.

## Input (stdin)

```
T Cap S_init
Pmin Pmax Qcap alpha beta Gamma delta Epsilon
b kappa dumpfee
DP[1] DQ[1]
...
DP[T] DQ[T]
```

## Output (stdout)

`T` lines, `"Q[t] dump[t]"`, giving your chosen heat output and dumped
amount for every period `t = 1..T`.

## Feasibility

For every `t`: `L(DP[t]) <= Q[t] <= U(DP[t])`, `dump[t] >= 0`, all values
finite, and the tank level stays in `[0, Cap]` at every period (a small
numerical tolerance, `2e-4`, is allowed). Any violation, wrong token count,
or non-finite value scores `0`.

## Objective & Scoring

Minimize total cost `F = sum_t [ b*Q[t] + kappa*(Q[t]-L(t))*(U(t)-Q[t]) ] + dumpfee * sum_t dump[t]`.
The checker builds its own baseline `B`: the cost of always running at the
top vertex `U(t)` and dumping the forced overflow (always feasible, since
`DQ[t] <= U(t)` for every `t`). Your score is

```
Ratio = min(1.0, 0.1 * B / F)
```

so the "always max, dump the rest" baseline scores `~0.1`, and you must
genuinely beat it to climb.

## Worked example (illustrative FORM only)

Say a single period has `L=2, U=10, b=1, kappa=0.5`. Tracking demand
pointwise at `Q=6` (midpoint) costs `6 + 0.5*4*4 = 14`. Operating at the
vertex `Q=10` costs `10 + 0 = 10` — cheaper, despite producing *more* heat
than needed, because the bump vanishes. The 4 units of surplus
then get absorbed by the tank (or dumped for `4*dumpfee` if the tank is
full) instead of being paid for as part-load inefficiency.

## Constraints

`4 <= T <= 250`. Time limit 5s, memory 512MB, deterministic scoring over 10
fixed seeded instances of increasing size, some with a tight tank forcing
real dump decisions.
