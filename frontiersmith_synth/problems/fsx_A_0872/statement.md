# Petal Press: Dispersion-Tuned Disk Fission

A disk's rim is modeled as `N = 64` points arranged on a ring, holding a scalar
field `u_0, ..., u_{N-1}`. The rim starts almost uniform, disturbed only by a
tiny hidden random noise field you never see. A fixed nonlinear growth process
then runs for `T` steps and settles into a pattern of alternating positive and
negative arcs around the ring — **petals**. Your job is to design the process
so that, reliably, it settles into **exactly `k` petals**, where `k` is given.

You do **not** write the simulator. You submit two things: a short-range
**coupling kernel** (how each point's neighbors push it) and a tiny **bias**
added to the initial state. The evaluator runs the fixed dynamics with your
kernel and bias over many independent hidden noise draws and reports the
fraction that land on exactly `k` petals. The bias budget is deliberately
minuscule (its L2 norm is capped near 2% of the noise field's own L2 norm) —
too small to force any particular wavenumber by itself, even if aimed exactly
at `cos(2*pi*k*i/N)`. The kernel is what has to do the real work.

## The dynamics (run by the evaluator)

Given your kernel `w_0..w_{L}` (`L = L_max`) and bias `b_0..b_{N-1}`, for each
hidden noise draw `eta` (i.i.d. `Uniform(-noise_amp, noise_amp)` per point):

```
u <- (b + eta), then subtract its mean            # zero-mean: no bulk "mode 0" win
repeat T times:
    lin_i <- w_0*u_i + sum_{j=1}^{L} w_j*(u_{i+j} + u_{i-j})     # indices mod N
    u     <- u + 0.1 * (lin - u^3)                               # dt=0.1, alpha=1
    u     <- u with its mean subtracted, then clipped to [-8, 8]
petals(u) <- (# positions i with sign(u_i) != sign(u_{i+1})) // 2
```

The mean is removed every step, so a spatially flat field can never win — some
nonzero wavenumber must dominate. Which one wins is controlled by your
kernel's own Fourier transform (its **dispersion relation**):
`lambda(m) = w_0 + 2*sum_j w_j*cos(2*pi*j*m/N)`. Whichever wavenumber `m` has
the largest `lambda(m)` tends to grow fastest during the early linear stage,
before the cubic term saturates it.

## Candidate program contract

Standalone program: read ONE JSON object from **stdin**, write ONE JSON object
to **stdout**. Runs in an isolated subprocess; sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute kernel and bias ...
print(json.dumps({"kernel": kernel, "bias": bias}))
```

### Public instance (stdin)

```json
{"name": "disk", "N": 64, "k": 10, "L_max": 6, "W_max": 1.2,
 "B": 0.0277, "T": 150, "M": 60, "noise_amp": 0.3}
```

`k` is the target petal count. `L_max` is the max kernel half-width you may
use (submit exactly `L_max+1` taps `w_0..w_{L_max}`, applied symmetrically:
offset `+j` and `-j` both get `w_j`). `W_max` bounds `|w_j|`. `B` bounds the
L2 norm of your length-`N` bias vector. `T`, `M`, `noise_amp` describe the
(hidden) noise ensemble the evaluator runs your kernel over — `M` independent
draws per instance.

### Answer (stdout)

```json
{"kernel": [w_0, ..., w_{L_max}], "bias": [b_0, ..., b_{N-1}]}
```

A submission is **valid** iff `kernel` has exactly `L_max+1` finite numbers
with `|w_j| <= W_max`, and `bias` has exactly `N` finite numbers with L2 norm
`<= B`. Any violation, a crash, a timeout, or non-JSON output scores that
instance `0.0`.

## Scoring (deterministic)

Let `frac_cand` be the fraction of the `M` hidden noise draws where your
kernel+bias yields exactly `k` petals. The evaluator also computes, itself,
two reference fractions per instance:

- `frac_naive`: fraction achieved by the **identity kernel** (`w_0=1`, all
  other taps `0`) with zero bias — no coupling at all.
- `frac_ideal`: fraction achieved by a **fixed wider reference kernel**
  (`L=14` taps, amplitude `0.5`, shaped as `cos(2*pi*k*j/N)`) — more taps than
  any candidate is ever allowed, so this is a strong but not perfectly
  reachable target.

```
r = clamp( 0.1 + 0.82 * (frac_cand - frac_naive) / max(1e-9, frac_ideal - frac_naive), 0, 1 )
```

Doing nothing scores `~0.1`; matching the idealized reference's fraction
scores up to `0.92` — genuine headroom remains since you always have strictly
fewer taps than the reference. The reported **Ratio** is the mean of `r` over
10 seeded instances (varying `k` and `L_max`); **Vector** lists the per-instance
scores.

## Suggested strategies

1. **No-op**: identity kernel, zero bias (the do-nothing baseline).
2. **Diffusive kernel + one big bump**: local-averaging weights plus a strong
   localized bias spike, hoping raw amplitude forces the shape.
3. **Dispersion-shaped kernel**: set `w_j` so the kernel's own Fourier
   transform peaks at wavenumber `k`, letting the *ensemble* of noise draws
   supply the seed and using growth-rate design to pick the winner reliably.
4. **Windowed / budget-aware refinements**: taper or reallocate your
   `L_max` taps and amplitude to sharpen the resonance without instability.
