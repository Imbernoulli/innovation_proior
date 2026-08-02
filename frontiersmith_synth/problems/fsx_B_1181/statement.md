# Naming the Circuit From a Charge-Discharge Curve

## Problem
A battery cell's terminal voltage under load is modeled by a Thevenin-style
equivalent circuit (ECM): a series resistance `R0`, some number of parallel
RC branches `(R_i, C_i)` (time constant `tau_i = R_i*C_i`) capturing
polarization relaxation, and a slow hysteresis voltage `h` driven by the
*sign* of the current. You are given ONE recorded drive cycle (current and
the resulting terminal voltage, with realistic measurement noise) from a
cell. Identify an ECM -- `R0`, a set of RC branches, and a hysteresis
magnitude `M` -- that predicts terminal voltage correctly, including on a
DIFFERENT drive cycle from the same cell that you never see.

The forward model, discretized at step `dt`:
```
tau_i = R_i * C_i,        a_i = exp(-dt / tau_i)
v_i[k+1] = a_i*v_i[k] + R_i*(1-a_i)*I[k]          (v_i[0] = 0)
a_h = exp(-dt / tau_h)
h[k+1] = a_h*h[k] + M*(1-a_h)*sign(I[k])          (h[0] = 0, sign(0)=0)
soc[k+1] = soc[k] - I[k]*dt / (3600*capacity_Ah)
V[k] = OCV(soc[k]) - R0*I[k] - sum_i v_i[k] - h[k]
```
`OCV` is linearly interpolated from the given table; positive `I` means
discharge.

## Input (stdin)
```
N test_id dt Kmax R0_lo R0_hi R_lo R_hi C_lo C_hi tau_lo tau_hi M_lo M_hi tau_h capacity_Ah soc_init
n_ocv
soc_0 ocv_0
...
soc_{n_ocv-1} ocv_{n_ocv-1}
I_0 I_1 ... I_{N-1}
V_0 V_1 ... V_{N-1}
```
`I`, `V` are the recorded drive cycle: current in amps (`I_k`, +discharge)
and the observed terminal voltage in volts (`V_k`, with measurement noise)
at each of `N` one-second steps. `test_id` is for reference only.

## Output (stdout)
```
R0 K M
R_1 C_1
...
R_K C_K
```
`0 <= K <= Kmax`. Each `(R_i, C_i)` is a distinct RC branch (order does not
matter); `tau_i = R_i*C_i`.

## Feasibility
`R0` in `[R0_lo,R0_hi]`; `K` an integer in `[0,Kmax]`; each `R_i` in
`[R_lo,R_hi]`, `C_i` in `[C_lo,C_hi]`, and `tau_i=R_i*C_i` in
`[tau_lo,tau_hi]`; `M` in `[M_lo,M_hi]`; every value finite. A wrong token
count, out-of-range value, non-finite value, or extra trailing tokens all
score `0`.

## Objective (maximize)
Your ECM is simulated forward -- with the *same* recursion above, the same
`tau_h`, `capacity_Ah`, `soc_init` and `OCV` table -- on a held-out drive
cycle from the same cell that you never see. That held-out cycle mixes a
genuinely different set of pulse durations than the one you were given. Let
`F` be the RMS error between your predicted and the true terminal voltage
there, and `B` the RMSE of the checker's own single-generic-branch reference
fit (one fixed time constant, not adapted to the data) evaluated on the same
held-out cycle. The score is
```
Ratio = min(1000, 100*B / max(1e-9, F)) / 1000
```
Reproducing the reference construction scores about `0.1`; a genuinely
better identification raises the ratio; residual measurement noise keeps
even a strong identification below `1.0`.

## Why more branches isn't automatically better
Two RC branches with similar time constants leave nearly the same
fingerprint in the response to any ONE current profile -- their *individual*
`(R,C)` values are not pinned down by that data, only their combined effect
is. A fit that insists on separating them anyway will match the visible
cycle beautifully, apportioning the shared response between the two
branches however training noise happens to break the tie -- and then
mispredicts badly the moment the held-out cycle's different pulse timing
asks the two branches to actually behave as if they were distinct. The
model that survives is the one whose branch count and placement match what
the given drive cycle can actually resolve, not the one with the most
parameters.

## Constraints
`400 <= N <= 4000`, `2 <= Kmax <= 8`. Time limit 2-5s, memory 512MB.

## Example
`testdata/1.in` holds one instance; running the checker against a submitted
`R0 K M` / branch-list artifact prints e.g. `... Ratio: 0.548000`.
