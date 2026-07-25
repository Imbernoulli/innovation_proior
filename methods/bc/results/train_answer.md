The problem is to turn a van's camera into a steering controller, in real time, on ordinary roads, without hand-specifying what a road looks like. The prevailing approach is a fixed vision-then-control pipeline: hand-tuned operators find road edges or road-shaped regions, fit a road model, and a geometric law converts that model into a steering command. Systems built this way can drive competently on the road type they were tuned for, but reaching that point takes a vision group months of algorithm development, and a new road surface or a new camera means redoing much of that work — the processing is frozen at design time while the world keeps changing. The alternative I want is to let a network learn the image-to-steering map directly by backpropagation, so the data decides which visual features matter rather than a programmer. Before I had the real sensor loop working, the fallback was to train that network on synthetic road images from a hand-built road generator — this avoids hand-built perception, but it just relocates the hand-specification into the simulator, and a simulator's synthetic statistics never quite match what a real camera sees. So the natural next move is the most direct one: drop the simulator, and while a person drives, pair each live camera frame with their current steering command and train on that stream directly. This is tempting because the teaching signal is free and continuous — every instant the person drives, the steering wheel is the correct answer for that instant's image. But run that naive loop in your head and it walks into two walls. First, a competent driver keeps the van near the center of the lane, so every training frame is an on-center view; the moment the learner makes a small error and drifts off-center, it is looking at a class of image it has never been trained on, its output there is unconstrained, and it has no way to recover — the training distribution is the teacher's distribution, not the learner's, and they diverge the instant the learner is imperfect. Second, the demonstrations arrive as a temporal stream and backprop has no memory beyond its weights: a long monotonous stretch — a sustained turn or a long straight — floods the network with near-identical examples and it overlearns the recent input, forgetting the rest of the task. Both walls are the same underlying disease: the live stream, taken as-is, is missing pieces of the task the network needs to see.

The method is Behavioral Cloning: reduce the whole control problem to supervised maximum-likelihood learning of the expert's action given the state,
$$\max_\theta \; \mathbb{E}_{(s,a^*)\sim D_{\text{expert}}}\big[\log \pi_\theta(a^*\mid s)\big],$$
and then design the training procedure so the examples fed to that objective actually cover the task. The network itself is nothing exotic: a low-resolution retina in, a hidden layer, and a steering output that is not a single regressed scalar but a population code — a bank of output units laid along the steering-curvature axis, trained toward a Gaussian activation hill centered on the correct curvature rather than a one-hot spike, and read back out as the center of mass of the resulting bump. The Gaussian target gives graded credit to near-misses instead of an all-or-nothing gradient, and the center-of-mass readout gives resolution finer than the unit spacing; this is what turns a discrete bank of units into a smoothly steerable command. What makes the method work, though, is not this readout but the two fixes to the naive on-the-fly training loop, and both fixes come from the same source: I know the geometry of the situation even before any learning happens. The camera's height and orientation relative to the ground are known, and on an ordinary stretch of road the ground can be treated as locally planar, so the perspective map between a ground point and its pixel is a fixed, invertible function of the vehicle's pose. That means I do not need to actually drive the van to a shifted or rotated position to know what the camera would see from there — I can compute it, by resampling the one real image I do have. From every real frame I synthesize roughly fourteen additional frames as if the vehicle's pose had been shifted by up to about six tenths of a meter and rotated by up to about six degrees, which manufactures exactly the off-center recovery views a competent human driver never produces, and because the pixel-sampling pattern for a given shift-and-rotation is constant, this synthesis is essentially free, folding straight into the image-reduction step that already runs. Shifting the apparent pose leaves a strip at the image edge with no source pixels, and how that strip is filled matters: filling it from the nearest visible pixel smears near-vertical features like lane lines sideways across rows, in an amount correlated with the shift and hence with the correct steering — an artifact the network will happily learn to read as its steering cue, and then lose the moment it is deployed on real, unsmeared images. So the edge must instead be extrapolated along the line parallel to the vehicle's original heading, toward the vanishing point, which continues a lane line in its true direction and leaves no such artifact.

Synthesizing the image is only half the job, because the label attached to a shifted image cannot be the driver's unchanged command — that command was correct for the original pose, not the new one. I need a model of how the right action changes with pose, and the simplest one that is both geometrically well-founded and matches how people actually steer is pure pursuit: aim at a fixed target point a lookahead distance $l$ down the road (road center, where the person was implicitly aiming) and steer the single circular arc that reaches it. Writing $r_p$ for the radius the person was actually steering, and $s$, $\theta$ for the shift and rotation of the synthetic pose, the sagitta relation for a chord of length $l$ on an arc of radius $r$ gives the lateral miss of the target from the original straight-ahead line as $d_p = r_p - \sqrt{r_p^2 - l^2}$; moving to the shifted, rotated pose adds the shift and a rotation-induced term and rescales by $\cos\theta$, giving $d = \cos\theta\,\big(d_p + s + l\tan\theta\big)$; and inverting the same sagitta relation for the new pose, $d = r - \sqrt{r^2 - l^2}$, yields the corrected steering radius
$$r = \frac{l^2 + d^2}{2d}.$$
Setting $s=\theta=0$ recovers $d = d_p$ and hence $r = r_p$, so the transform is the identity when the pose is not actually moved, which is the least it must do. The lookahead $l$ is set to the distance the vehicle travels in about two to three seconds; at highway speed this lookahead is what makes the formula's corrective radius for a roughly one-meter displacement land in the 500-1200 m range that Reid, Solowka and Billing measured for real human steering corrections, so the synthesized labels are not just internally consistent but physically realistic. Any synthetic pose whose corrected radius is sharper than about 20 m — sharper than the output bank can represent — is discarded and redrawn rather than kept, so the network's limited capacity is not spent on poses it will essentially never occupy.

That handles the missing-recovery-examples wall. The forgetting wall needs a second mechanism: instead of training only on the current cycle's patterns, maintain a buffer of about two hundred past patterns and run one backprop pass over the whole buffer every cycle, so a monotonous stretch cannot dominate what the network is shown. What matters is which pattern gets evicted to make room for each new one. Dropping the oldest pattern turns the buffer into a sliding window that fills with one turn direction during a sustained turn — the same disease the buffer was meant to cure. Dropping whichever pattern currently has the highest training error also fails, in a different way: an occasional lapse by the human driver produces a mislabeled pattern the network can never fit well, so it always has high error and, under this rule, never gets evicted — it festers in the buffer indefinitely. The rule that actually targets the real hazard — a standing left/right bias — is to track the buffer's mean *signed* steering scalar (not the raw radius, which loses the left/right sign and treats straight-ahead as an infinite radius) and, on each insertion, evict whichever pattern's removal leaves that mean closest to zero, i.e. closest to straight-ahead. Since left and right turns occur about equally often over a long drive, this eviction rule keeps encoding that left-right symmetry into the training set irrespective of what the current stretch of road happens to look like, so the network never acquires a standing turn bias.

Putting the whole cycle together: each pass grabs the live frame, reduces it to the retina, generates the fourteen transformed poses with pure-pursuit-corrected labels (redrawing any that come out too sharp), inserts those fourteen plus the one real pattern into the two-hundred-pattern buffer with the mean-signed-steering eviction rule, and then runs one forward and backward pass over the entire buffer with learning rate 0.01 and momentum 0.8. Roughly a hundred such cycles, at a couple of seconds each, are enough to learn a new road — about four to five minutes of a person simply driving at the speed the network will later be tested at. Here is the training procedure; the network's forward pass, its center-of-mass readout, and its backprop update already exist, and the image-reduction step gains the transform as an extra, nearly-free resampling pass:

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

The same idea, stripped to its minimal modern form, is to clone a continuous-action policy by directly minimizing the negative log-likelihood of expert actions, with the entropy and L2 regularization terms of a standard imitation-learning implementation:

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
