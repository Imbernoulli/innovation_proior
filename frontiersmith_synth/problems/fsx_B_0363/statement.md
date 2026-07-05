# Telescope-Array Guide-Beacon Fan-Out: Phase-Mask Design

## Setting

A segmented **telescope array** carries a metrology / laser-guide-star
transmitter. Behind the pupil sits a phase-only **Spatial Light Modulator
(SLM)**: an `N x N` grid of pixels, each imposing a phase shift `phi[i][j]`
(radians) on an incoming monochromatic plane wave of **unit amplitude**. The
pupil field is therefore

```
field[i][j] = exp( 1j * phi[i][j] )          (|field| == 1 everywhere)
```

The far field on the sky is the 2-D discrete Fourier transform of the pupil
field. You must sculpt the phase so the far field lights up a prescribed
**array of guide-beacon spots** (a fan-out) that is simultaneously
**power-efficient** and **uniform**.

This is deterministic simulated propagation: the evaluator re-runs the FFT
itself. Nothing about the score depends on wall-clock time, hardware, or the
candidate's own claimed numbers.

## The public instance (stdin, one JSON object)

```json
{
  "N": 64,                       // SLM grid side length; phase array is N x N
  "gamma": 6,                    // uniformity exponent in the figure of merit
  "targets": [[fy, fx], ...]     // K target spots as UNSHIFTED FFT index coords
                                 //   (row fy in [0,N), col fx in [0,N))
}
```

`targets` are the far-field locations (as integer indices into the unshifted
`numpy.fft.fft2` output) that must receive equal, strong intensity. All spots
are distinct and none is the DC term `(0,0)`.

## Your answer (stdout, one JSON object)

```json
{ "phase": [[phi_00, phi_01, ...], ...] }   // an N x N array of finite floats (radians)
```

Exactly `N` rows of `N` finite numbers. Phases may take any real value (they are
applied as `exp(1j*phi)`; only their value mod 2*pi matters). Any wrong shape, a
non-`N x N` array, or any non-finite entry scores **0** on that instance.

## Objective (MAXIMIZE)

Let `F = FFT2(exp(1j*phi))` and `P = |F|^2`. With `I_k = P[fy_k, fx_k]` the
intensity at target spot `k`:

```
eta = ( sum_k I_k ) / ( sum over the whole plane of P )      // diffraction efficiency
u   = 1 - (max_k I_k - min_k I_k) / (max_k I_k + min_k I_k)  // uniformity, 1 == flat
M   = eta * u ** gamma                                        // composite figure of merit
```

You maximize `M`. Efficiency rewards routing light into the wanted spots;
`u ** gamma` heavily penalizes an uneven fan-out. Because the element is
**phase-only**, some light always scatters into non-target orders and the spots
can never be made perfectly equal, so `M < 1` always — there is permanent
headroom and no closed-form optimum.

## Scoring

For each instance the evaluator recomputes `M` from your phase array and
normalizes against its own reference construction `M_ref` (an in-phase
superposition of single-spot gratings):

```
r = min( 1, 0.1 * M / M_ref )
```

The reported score is the mean of `r` over a fixed, seeded battery of
`telescope-array` instances of varying grid size, spot count, and fan-out
geometry (including larger, harder held-out cases). Reproducing the reference
construction scores about `0.1`; genuinely better efficiency and uniformity
score higher.

## Notes / viable strategies

- **Reference:** add one linear grating per target and keep `angle(sum)`.
- **Random-phase superposition:** give each grating an independent random phase
  offset to de-correlate interference — cheap and noticeably flatter.
- **Iterative Fourier-transform (Gerchberg-Saxton):** alternate between pupil
  and far-field planes, re-imposing the phase-only constraint and adaptively
  reweighting the target amplitudes so dim spots are boosted.
- The candidate runs in an isolated sandbox and sees only this public instance;
  the scorer runs in a separate process.
