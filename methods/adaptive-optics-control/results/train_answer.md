Light from a distant star reaches the top of the atmosphere as an essentially flat wavefront, but after passing through kilometres of turbulent air the phase front over the telescope pupil is crumpled and evolves on millisecond timescales. A crumpled wavefront no longer focuses to the diffraction limit; the image spreads into the much broader seeing disk. The core problem is therefore to infer the time-varying phase aberration from real-time measurements and cancel it with a deformable mirror. The difficulty is that optical detectors record intensity, not phase, so the phase must be recovered indirectly, and the correction must run faster than the turbulence so that the applied shape is not stale before it reaches the mirror.

Existing ideas each leave a gap. A modal Zernike fit gives a smooth low-order reconstruction, but it discards mid- and high-spatial-frequency structure that a real actuator grid can correct, and it does not directly answer what commands this particular mirror should apply against this particular sensor. Direct zonal integration of gradients reconstructs a phase map on a grid, but it is sensitive to geometry-dependent null modes such as piston and the checkerboard waffle, and it still leaves a separate step to convert phase into actuator commands. A one-shot pseudo-inverse correction is open-loop: it inherits calibration errors, amplifies noise through the small singular values of the forward map, and does nothing about the aberration changing between frames. What is needed is a calibrated command-to-slope map, a stable inversion that respects the sensor's blind spots, and a feedback loop that repeatedly drives the residual toward zero.

The method is least-squares wavefront reconstruction with closed-loop integrator control. The first step is to build the interaction matrix by poking each actuator individually and recording the slope vector it produces on the Shack-Hartmann wavefront sensor. Because the mirror subtracts its own phase from the incoming wavefront, the stored row for an actuator is the negative of the raw sensor response to that poke, divided by the poke amplitude. This calibration captures the true end-to-end response of the real hardware, including influence-function overlap and alignment, rather than relying on an idealised analytic stencil.

The interaction matrix is rectangular and rank-deficient. Global piston produces no slopes, so it lies in the null space. In the Fried geometry a checkerboard waffle mode also gives zero slope in every sub-aperture in the ideal local stencil, and in a finite real pupil this waffle-like content leaks into several near-null singular directions. Inverting the matrix directly would therefore blow up on the exact null modes and catastrophically amplify noise on the poorly sensed ones. The cure is to invert with a regularised pseudo-inverse. In the standard actuator-by-slope storage, control_matrix is obtained from numpy.linalg.pinv with a relative conditioning threshold that truncates singular values below rcond times the largest singular value. Alternatively, Tikhonov regularisation replaces the raw inverse of each singular value sigma with sigma divided by sigma squared plus alpha, smoothly suppressing the modes that are dominated by noise while leaving the well-sensed modes nearly untouched.

A single reconstructed command is not enough. Atmospheric turbulence is red in time, with most power at low temporal frequencies, but it is still moving, and the loop has latency. The solution is to place the corrector in front of the sensor and integrate. Each frame the sensor measures the residual slopes, the reconstructor turns them into a residual command increment, and the running actuator command is updated by adding a fraction of that increment. A small leak term pulls every command component gently toward zero each frame. For modes the sensor can see, the loop re-commands them and the leak is negligible; for unsensed modes such as waffle or piston the leak is the only force acting on them, so they drain away rather than growing unchecked. The leak also moves the integrator pole from the unit circle to a stable interior point. The gain is typically kept well below 0.5, because with about two frames of loop delay the pure-integrator stability boundary is near 1, and robust damping requires a comfortable margin. The gain is tuned against the atmosphere: higher gain gives more bandwidth and less servo-lag error, while lower gain keeps noise propagation down; on bright stars one can push closer to the stability ceiling, while faint stars demand a quieter gain.

```python
import numpy as np


class Reconstructor:
    """Poke-calibrated slope-to-command reconstructor for adaptive optics."""

    def __init__(self):
        self.interaction_matrix = None
        self.control_matrix = None
        self.actuator_values = None

    def build_interaction_matrix(self, dm, wfs, poke=1.0, imat_noise=False):
        """Poke each actuator and store signed actuator-by-slope responses."""
        n_act = dm.n_act
        n_slope = wfs.n_measurements
        interaction = np.zeros((n_act, n_slope))
        for j in range(n_act):
            cmd = np.zeros(n_act)
            cmd[j] = poke
            mirror = dm.shape(cmd)
            slopes = wfs.measure(
                phase_correction=mirror,
                noise=imat_noise,
                calibration=True,
            )
            interaction[j] = -slopes / poke
        self.interaction_matrix = interaction
        return interaction

    def build_command_matrix(self, conditioning=1e-3, alpha=None):
        """Regularised SVD pseudo-inverse; Tikhonov when alpha is given."""
        if alpha is None:
            self.control_matrix = np.linalg.pinv(
                self.interaction_matrix, rcond=conditioning
            )
        else:
            U, sig, Vt = np.linalg.svd(
                self.interaction_matrix, full_matrices=False
            )
            filt = sig / (sig ** 2 + alpha)
            self.control_matrix = (Vt.T * filt) @ U.T
        return self.control_matrix

    def reconstruct(self, slopes):
        return self.control_matrix.T @ slopes

    def apply_gain(self, command_increment, gain=0.4, leak=0.0):
        if self.actuator_values is None:
            self.actuator_values = np.zeros(command_increment.size)
        self.actuator_values = (
            (1.0 - leak) * self.actuator_values + gain * command_increment
        )
        return self.actuator_values


def close_loop(reconstructor, dm, wfs, atmos, gain=0.4, leak=0.0, n_iter=1000):
    """Leaky integrator closed loop: c_{k+1} = (1-leak) c_k + gain * R s_k."""
    commands = np.zeros(dm.n_act)
    slopes = np.zeros(wfs.n_measurements)
    for _ in range(n_iter):
        phase = atmos.step()
        delta = reconstructor.reconstruct(slopes)
        commands = reconstructor.apply_gain(delta, gain=gain, leak=leak)
        correction = dm.shape(commands)
        slopes = wfs.measure(phase, phase_correction=correction)
    return commands


reconstructor = Reconstructor()
reconstructor.build_interaction_matrix(dm, wfs)
reconstructor.build_command_matrix(conditioning=1e-3)
final_commands = close_loop(reconstructor, dm, wfs, atmos, gain=0.4, leak=0.01)
```
