# Least-squares wavefront reconstruction with closed-loop integrator control

## Problem

Atmospheric turbulence imprints a rapidly varying phase aberration φ(x, y, t) on the wavefront entering a telescope, blurring images far below the diffraction limit. A Shack–Hartmann wavefront sensor measures only **local slopes** (gradients) of the wavefront on a grid of sub-apertures — not the phase itself — and a deformable mirror with a finite set of actuators must be commanded, in a real-time feedback loop, to flatten the wavefront. The task: recover mirror commands from noisy, incomplete slope measurements and apply them stably as turbulence evolves on a millisecond timescale.

## Key idea

1. **Forward map (interaction matrix).** Poke each actuator in turn and record the slope vector it produces. Column *j* is the slope signature of actuator *j*, giving a linear model **s = A c** from commands to measured slopes, calibrated on the real mirror and sensor.
2. **Least-squares reconstructor.** Commands are recovered by minimizing ‖A c − s‖². The solution c = (AᵀA)⁻¹Aᵀs uses the Moore–Penrose pseudo-inverse **R = A⁺**, computed via SVD. The forward map is rank-deficient (piston and the Fried-geometry **waffle** checkerboard produce zero slopes) and ill-conditioned, so the pseudo-inverse is **regularized** — truncate singular values below `rcond·σ_max`, or apply Tikhonov filtering σ/(σ²+α) — discarding the unsensed and noise-dominated modes that would otherwise blow up the noise.
3. **Closed-loop leaky integrator.** Place the corrector before the sensor so it sees the *residual*. Update the running command each frame:

   **c_{k+1} = (1 − ℓ) c_k − g · R s_k**

   - Negative feedback drives the residual slopes to zero.
   - Integral action gives zero steady-state error and high gain at low temporal frequency, where the red turbulence spectrum concentrates its power.
   - Gain **g** is held below ≈ 0.5 for damping against the ≈ 2-frame loop delay (the exact pure-integrator bounds are 0 < g < 2 for one frame and 0 < g < 1 for two frames), and tuned to trade closed-loop bandwidth / servo-lag (σ²_servo = 28.4(f_G Δt)^(5/3), driven by the Greenwood frequency f_G = 0.426 v/r₀) against noise propagation.
   - The leak **ℓ** (small, e.g. 0.01) bleeds off unsensed modes (piston, waffle) that the loop can never correct, and moves the integrator pole from z = 1 to z = 1 − ℓ for strict stability — at the cost of a slight residual DC error.

## Algorithm

```
Calibration (offline):
  for each actuator j:
      command actuator j by a small poke
      A[:, j] = (resulting slope vector) / poke
  R = regularized_pseudo_inverse(A, rcond)      # SVD, drop small singular values

Closed loop (per frame k, ~1 kHz):
  measure residual slopes  s_k       (Shack-Hartmann centroids)
  delta  = R @ s_k                    # least-squares residual command
  c_{k+1} = (1 - leak) * c_k - gain * delta
  apply c_{k+1} to deformable mirror
```

## Code

```python
import numpy as np


class Reconstructor:
    """Poke-calibrated slope-to-command reconstructor."""

    def __init__(self):
        self.interaction_matrix = None
        self.command_matrix = None

    def build_interaction_matrix(self, dm, wfs, poke=1.0, imat_noise=False):
        """Poke each actuator, record its slope response -> A in s = A c."""
        n_act = dm.n_act
        n_slope = wfs.n_measurements             # 2 (sx, sy) per sub-aperture
        A = np.zeros((n_slope, n_act))
        for j in range(n_act):
            cmd = np.zeros(n_act)
            cmd[j] = poke
            mirror = dm.shape(cmd)
            A[:, j] = wfs.measure(mirror, noise=imat_noise) / poke
        self.interaction_matrix = A
        return A

    def build_command_matrix(self, conditioning=1e-3, alpha=None):
        """SVD pseudo-inverse; Tikhonov when alpha is supplied."""
        if alpha is None:
            self.command_matrix = np.linalg.pinv(
                self.interaction_matrix, rcond=conditioning
            )
        else:
            U, sig, Vt = np.linalg.svd(self.interaction_matrix, full_matrices=False)
            filt = sig / (sig**2 + alpha)
            self.command_matrix = (Vt.T * filt) @ U.T
        return self.command_matrix

    def reconstruct(self, slopes):
        return self.command_matrix @ slopes


def close_loop(reconstructor, dm, wfs, atmos, gain=0.4, leak=0.01, n_iter=1000):
    """Leaky integrator: c_{k+1} = (1 - leak) c_k - gain * R @ s_k.

    Sensor sees the residual after correction; the loop nulls it. gain < ~0.5
    keeps the ~2-frame-delay loop stable; leak drains unsensed modes and pins
    the integrator pole at z = 1 - leak.
    """
    commands = np.zeros(dm.n_act)
    for _ in range(n_iter):
        phase = atmos.step()                     # turbulence evolves
        residual = phase - dm.shape(commands)    # post-correction wavefront
        slopes = wfs.measure(residual)           # local gradients on the grid
        delta = reconstructor.reconstruct(slopes)
        commands = (1.0 - leak) * commands - gain * delta
    return commands


reconstructor = Reconstructor()
reconstructor.build_interaction_matrix(dm, wfs)
reconstructor.build_command_matrix(conditioning=1e-3)
final_commands = close_loop(reconstructor, dm, wfs, atmos, gain=0.4, leak=0.01)
```

AO simulators often store the same matrices transposed: Soapy builds an actuator-by-slope interaction matrix by poking actuators, computes `numpy.linalg.pinv(interaction_matrix, conditioning)`, applies it as `control_matrix.T.dot(slopes)`, and accumulates the resulting residual commands with the loop gain. With the positive command-to-slope convention used above, the equivalent matrix is `R = A^+` and the closed-loop feedback subtracts `gain * R @ slopes`.
