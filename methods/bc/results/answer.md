# Behavioral Cloning (BC), distilled

Behavioral Cloning reduces control to supervised learning: collect (state, action) pairs from
an expert demonstrator and train a policy to reproduce the expert's action at each state by
gradient descent on a supervised loss. No reward, no environment interaction during learning —
just function approximation of the expert's state-to-action map. In its original incarnation
(ALVINN, autonomous road following) the state is a reduced camera image, the action is a
steering command, the expert is a human driver, and the contribution is the *training
procedure* that makes naive cloning actually drive: synthesize the off-distribution examples
the expert never produces, and keep the training set diverse and unbiased.

## Problem it solves

Learn a competent controller `pi_theta(a | s)` from expert demonstrations alone, cheaply and
with little hand-engineering, using the freely-available teaching signal (the expert's action
for each observed state).

## Key idea

Maximize the likelihood of the expert's actions under the policy (equivalently, minimize a
supervised loss between policy and expert actions) on the states the expert visited:

```
max_theta  E_{(s, a*) ~ D_expert} [ log pi_theta(a* | s) ]
```

For a Gaussian-policy continuous action this is negative-log-likelihood, usually with small
regularizers such as entropy and L2 terms; for ALVINN's population-coded steering output it is
the squared error to a Gaussian activation hill centered on the correct curvature, read back by
center of mass for sub-unit-resolution steering.

## The two failure modes naive cloning hits, and the fixes

1. **No recovery examples (distribution mismatch).** A competent expert keeps the system near
   the "center," so the demonstrations contain only on-center states; once the learner makes a
   small error it drifts into states the expert never visited and cannot recover. ALVINN's fix:
   **transform the sensor image.** Under a planar-ground assumption the camera-to-ground
   perspective map is known, so each real frame is resampled into ~14 synthetic frames at
   shifted/rotated virtual vehicle poses (shift `~[-0.6, +0.6] m`, rotation `~[-6, +6] deg`) —
   exactly the off-center recovery views the expert never produces.
   - **Edge extrapolation:** shifting reveals unseen ground at the frame edge; fill those
     pixels along the line *parallel to the vehicle heading* (toward the vanishing point), not
     from the nearest pixel — nearest-pixel fill smears road features across rows into an
     artifact the network learns to (wrongly) depend on.
   - **Label transform (pure pursuit):** the expert's command is wrong for the new pose, so
     relabel each synthetic pose with the arc that reaches the same target point T (road center,
     a lookahead `l` ahead). With `r_p` the radius the expert was steering, shift `s`, rotation
     `theta`:
     ```
     d_p = r_p - sqrt(r_p^2 - l^2)               # how far T sits off the original straight-ahead line
     d   = cos(theta) * (d_p + s + l*tan(theta)) # lateral miss of T from the new pose
     r   = (l^2 + d^2) / (2 d)                    # arc radius from the new pose to T
     ```
     derived by inverting the chord-sagitta relation `d = r - sqrt(r^2 - l^2)`. Lookahead `l` =
     distance travelled in ~2-3 s. Transforms whose label is sharper than the output bank can
     represent (~20 m radius) are discarded.

2. **Overlearning recent inputs (catastrophic forgetting).** The demonstration stream is
   temporal; a monotonous stretch (long straight or sustained turn) floods backprop with
   near-identical frames and it forgets the rest. Fix: a **buffer** of ~200 past patterns; insert
   each new pattern, then evict the pattern whose removal leaves the buffer's **mean signed
   steering scalar closest to straight ahead** — not the mean raw radius, which loses the
   left/right sign and treats straight as infinite radius. This locks in diversity and the
   left/right-symmetry prior so the policy never acquires a standing turn bias. (Simpler
   eviction rules fail: replace-oldest fills with one turn direction during a long turn;
   replace-lowest-error lets a momentary expert *mistake* fester forever since the net can
   never fit it.)

## Training details (ALVINN)

14 transforms + 1 real pattern per cycle into a 200-pattern buffer; one forward + backward
backprop pass over the buffer per cycle; learning rate `0.01`, momentum `0.8`; ~100 cycles at
~2.5 s each, i.e. ~4-5 minutes of ordinary human driving; the expert drives at the test speed
(5-55 mph).

## Why these choices

- **Supervised reduction:** the expert's action is a free, dense teaching signal; backprop
  learns its own features, so no hand-built perception is needed.
- **Population code + Gaussian target + center-of-mass readout:** smooth, finely-graded control
  from a coarse discrete output; soft targets give graded credit and better gradients than
  one-hot.
- **Transform, don't swerve:** generates recovery views automatically; no manual learning
  on/off toggling, no dangerous driving; cheap because the resampling pattern for a given
  (shift, rotation) is constant and folds into image reduction.
- **Parallel-to-heading extrapolation:** respects that road features run parallel to the road;
  avoids the row-smear artifact the network would otherwise exploit.
- **Pure pursuit:** a simple, situation-independent model of how the correct action changes with
  pose; matches measured human steering responses.
- **Mean-signed-steering-toward-straight buffer eviction:** prevents both forgetting and turn
  bias by encoding the left/right-frequency-symmetry prior.

## Working code

The original ALVINN-style training procedure (the contribution — the network and backprop
already exist):

```python
import numpy as np


def pure_pursuit_radius(r_person, shift_s, rot_theta, lookahead_l):
    """Steering arc radius for a virtual pose shifted by s, rotated by theta, aiming at
    the same target T the expert aimed at (road center, lookahead l ahead)."""
    d_p = r_person - np.sqrt(r_person ** 2 - lookahead_l ** 2)        # T off original straight-ahead line
    d = np.cos(rot_theta) * (d_p + shift_s + lookahead_l * np.tan(rot_theta))  # T off new pose's line
    return (lookahead_l ** 2 + d ** 2) / (2.0 * d)                    # invert d = r - sqrt(r^2 - l^2)


def behavioral_cloning_alvinn(network, sensor_stream, steering_stream,
                              lr=0.01, momentum=0.8, n_transforms=14,
                              buffer_size=200, max_shift=0.6, max_rot_deg=6.0,
                              lookahead_l=..., sharpest_radius=20.0):
    buffer = []  # (retina, target_profile, signed_steering_scalar)
    for raw_image, human_steering in zip(sensor_stream, steering_stream):
        retina = reduce_to_retina(raw_image)
        r_person = radius_of(human_steering)

        new = [
            (retina, encode_steering_target(human_steering),
             signed_steering_scalar(human_steering))
        ]
        while len(new) < 1 + n_transforms:                            # synthesize off-center poses
            s = np.random.uniform(-max_shift, max_shift)
            theta = np.radians(np.random.uniform(-max_rot_deg, max_rot_deg))
            r = pure_pursuit_radius(r_person, s, theta, lookahead_l)  # transform the label
            if abs(r) < sharpest_radius:                             # too sharp to represent -> redraw
                continue
            img = transform_image(retina, s, theta)                  # resample; fill edge along heading
            t_steering = steer_from_radius(r)
            new.append((img, encode_steering_target(t_steering),
                        signed_steering_scalar(t_steering)))

        for pat in new:                                              # buffer: keep mean signed steering straight
            buffer.append(pat)
            if len(buffer) > buffer_size:
                signed_steering_sum = sum(p[2] for p in buffer)
                victim = min(range(len(buffer)),
                             key=lambda i: abs((signed_steering_sum - buffer[i][2]) / (len(buffer) - 1)))
                buffer.pop(victim)

        for retina_i, target_i, _ in buffer:                        # one backprop pass over the buffer
            network.backprop_step(retina_i, target_i, lr, momentum)
```

The same idea in its minimal modern form — clone a continuous-action policy by minimizing the
negative log-likelihood of expert actions, with the same entropy and L2 regularization terms
used by a standard imitation-learning implementation:

```python
import torch as th


def behavioral_cloning_step(policy, optimizer, expert_obs, expert_acts,
                            ent_weight=1e-3, l2_weight=0.0):
    """One supervised BC update: maximize log pi(a* | s) on expert (state, action) pairs.
    Mirrors the imitation-library loss: neglogp + ent_loss + l2_loss."""
    _, log_prob, entropy = policy.evaluate_actions(expert_obs, expert_acts)
    log_prob = log_prob.mean()
    entropy = entropy.mean() if entropy is not None else None

    l2_norm = sum(th.sum(th.square(w)) for w in policy.parameters()) / 2
    entropy_term = entropy if entropy is not None else th.zeros((), device=log_prob.device)
    neglogp = -log_prob
    ent_loss = -ent_weight * entropy_term
    l2_loss = l2_weight * l2_norm
    loss = neglogp + ent_loss + l2_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```
