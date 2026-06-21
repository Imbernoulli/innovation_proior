# Context: learning to control a robot from a human teacher (late 1980s / early 1990s)

## Research question

A mobile robot has to produce a continuous control signal — a steering command — from raw
sensor input, frame after frame, fast enough to drive in real time. The setting is the real
world: outdoor scenes are noisy and variable; a road can be paved or dirt, single- or
multi-lane, lined or unlined, lit or shadowed, and the same physical road looks different under
different weather and camera orientations. Whatever maps the image to a steering command must
be cheap enough to run many times a second on the modest on-board compute of the time, and it
must keep working as the conditions change rather than being re-engineered by hand for every
new road type.

The desired output is *observable*. When a person drives the vehicle, the steering wheel
records, for every sensor image, the action a competent controller should have produced. So a
teaching signal is available continuously, in exactly the form a learner would need. The goal
is to convert that stream of (sensor image, human steering) pairs into a controller that,
deployed on its own, drives the vehicle competently across the range of situations it will
encounter, with a procedure that needs little hand-engineering and can be retrained quickly for
a new road.

## Background

**Backpropagation and supervised function approximation.** By this time the dominant tool for
learning an input-output map from examples is the multilayer feedforward network trained by
error backpropagation (Rumelhart, Hinton & Williams 1986). Given a differentiable network
`f_theta`, a set of input-target pairs `(x, y)`, and a differentiable loss `L(f_theta(x), y)`,
one repeatedly presents examples, propagates activation forward to get the network's response,
compares it to the target, and propagates the error backward to adjust every weight by a small
step down the gradient. Across handwritten-character (Jackel et al. 1988) and speech (Waibel et
al. 1988) recognition, the hidden units *learn their own features*: the data, not the
programmer, determines which aspects of the input matter for the task. Backprop is a
batch/online procedure over a *fixed, representative* set of examples; its guarantees assume
the training examples are drawn from the same distribution the trained network will be asked to
handle. It has no memory beyond the weights: the weights at any moment reflect the recent
stream of examples presented to it. And its output on inputs unlike its training examples is
unconstrained.

**Hand-built autonomous navigation.** The prevailing approach to road following is a fixed
pipeline: detect road edges or road-shaped regions with hand-tuned image operators, fit a road
model, and feed a geometric controller. Systems in this family (Thorpe et al. 1987; Dickmanns &
Zapp 1986; Kuan et al. 1988; Dunlay & Seida 1988) drive in the conditions they were tuned for,
performing the same fixed processing across situations. Reaching competent performance on one
road type took the vision groups of the day many months of algorithm development and parameter
tuning, and a change of road type or sensor meant much of that work over.

**Population codes and graded targets.** A standard way to read a scalar quantity (here,
steering curvature) out of a network is not a single linear output but a *bank* of output units
laid along the quantity's axis, with a localized bump of activation marking the value — a
population code. Reading the quantity back as the location of the bump (e.g. its center of
mass) gives a resolution finer than the unit spacing, and training each unit toward a smooth
(e.g. Gaussian) profile rather than a hard one-hot target gives partial credit to near-misses,
which yields better-behaved gradients than an all-or-nothing target.

**Geometry of a driving correction.** Two pieces of physical knowledge about driving are
available before any learning happens. (i) A vehicle-mounted camera has a known height and
orientation relative to the road surface, and on ordinary road-following stretches that surface
can be treated as locally planar. (ii) Empirical studies of human steering (Reid, Solowka &
Billing) report that a driver displaced laterally by about a meter at ~50 km/h responds with a
steering arc of radius roughly 500-1200 m; simple geometric steering laws can approximate these
corrective responses. Both facts are about the world, not about any controller.

## Baselines

**Hand-engineered road followers (Thorpe et al. 1987; Dickmanns & Zapp 1986; Kuan et al.
1988).** Core idea: a fixed perception-then-control stack — segment the image into road and
non-road using designed operators, estimate the road's geometry, and command a steering arc
from that estimate. On the conditions they were built for, several reached solid driving
performance (Thorpe's group drove the Navlab testbed). The perception stage is hand-specified
and fixed across circumstances.

**Supervised network trained on simulated road images.** Core idea: avoid hand-built perception
by training a feedforward network to map a low-resolution road image to a steering direction,
using a *road generator* that synthesizes labeled road snapshots under a variety of
orientations, widths, and lighting. The network is a single hidden layer over a 30x32 retina,
with a bank of direction output units carrying a Gaussian activation hill centered on the
correct turn curvature; driving reads a steering direction from that output profile. The
training distribution is whatever the simulator synthesizes.

**Naive online supervised imitation.** Core idea: drop the simulator and instead train the same
network online, directly on live sensor frames, pairing each current image with the human's
current steering command as the target, while the person drives. This is the most direct use of
the teaching signal: a competent human keeps the vehicle near the lane center, so the live
examples are the near-center views the driver produces, arriving as a temporal stream in the
order they are driven.

## Evaluation settings

The natural yardstick is real driving on the CMU Navlab, a modified van carrying the camera, a
scanning laser rangefinder, and on-board computers, on real roads of several types (single-lane
paved and dirt access roads, lined and unlined two-lane roads, on- and off-road). Inputs are
reduced to a 30x32 retina before the network. The standard accuracy measure for a steering
network on held-out images is *steering error*: fit the network's output activation profile
with its bump, take the bump location as the network's commanded direction, and measure its
distance (in output units) from the reference direction the human steered. On the road, driving
quality is measured as the vehicle's lateral displacement from the road center as the network
drives a test stretch (mean and standard deviation in cm), and as the speed and distance over
which it can drive without intervention; a human driver over the same stretch is the reference.
Training cost is measured in minutes of human driving needed before the network can take over.

## Code framework

The existing substrate is a feedforward network trained by backpropagation, plus the vehicle's
real-time loop that grabs a sensor frame, reduces it to the retina, runs the network forward to
get a steering command, and drives. The scaffold below exposes only what already exists (the
network, the backprop update, the image-reduction step, the human teaching signal) and leaves
one generic training-procedure stub.

```python
import numpy as np


class Network:
    """Existing feedforward net trained by backprop (RHW86). Maps a reduced
    sensor retina to a steering command. Architecture and backprop update exist;
    how the network is shown data is what we will design."""

    def forward(self, retina):
        # existing forward pass -> output-layer activation profile
        ...

    def steering_from_output(self, output_profile):
        # existing readout: locate the activation bump -> a scalar steering command
        ...

    def backprop_step(self, retina, target_profile, lr, momentum):
        # existing supervised weight update toward target_profile
        ...


def reduce_to_retina(raw_image):
    """Existing preprocessing: downsample the raw sensor frame to the small retina
    the network consumes (a fixed pixel-sampling pass)."""
    ...


def encode_steering_target(steering_command):
    """Existing: turn a scalar steering command into the network's output-layer
    target profile (the graded bump used as the supervised target)."""
    ...


def train_while_human_drives(network, sensor_stream, human_steering_stream,
                             lr, momentum):
    """Online supervised training from a human teacher. Each loop: a live sensor frame and
    the human's current steering command are available; the missing piece is the training
    procedure that decides how backprop should use this stream."""
    for raw_image, human_steering in zip(sensor_stream, human_steering_stream):
        retina = reduce_to_retina(raw_image)
        # TODO: fill in the training procedure.
        pass
```

The single empty slot is the training procedure that makes the live teaching stream useful to
the backprop update.
