## Research question

How do you make a computer animation of a *flock of birds* — or a school of fish, or a herd of land animals — that looks right, without scripting the path of every individual? The aggregate motion of a flock is one of nature's most familiar sights and one of its most paradoxical: it is made of discrete birds yet flows like a fluid; it looks randomly arrayed yet is magnificently synchronized; and most puzzling of all, it gives a strong impression of centralized, intentional control even though no bird is in charge. A solution would have to reproduce that polarized, non-colliding, fluid group motion — including the way a flock splits to pass an obstacle and re-forms on the far side — and it would have to do so cheaply enough to animate dozens of birds for hundreds of frames.

The difficulty is concrete. Scripting the path of a large number of individual objects by the standard keyframe-and-interpolate techniques of the time is tedious; given how complex the paths birds actually trace, it is doubtful the specification could even be made without error; and even if a plausible set of paths were drawn by hand, the *constraints* of flock motion could not be maintained — in particular, preventing collisions between every pair of birds at every single frame. Worse, a flock scripted this way would be nearly impossible to edit: changing the course of the whole flock for a few seconds would mean re-drawing every bird's path. What is wanted instead is a way to *generate* the motion: describe how one bird behaves, instance it many times, and let the flock fall out of the interaction.

## Background

**Scripting versus simulation in animation.** Traditional cel animation used a completely inert medium; the animator specified every motion directly. Computer animation runs on an active medium but, with the tools of the day, animators still work at almost the same low level — they tell the story by directly describing each character's motion, using helpers only to interpolate between keyframes. Little had been done to *automate* motion description. The alternative on the table is behavioral animation: model not just the character's shape and physics but its *behavior*, so the simulated character handles the details of its own motion. The animator becomes a meta-animator, a director of behavior rather than a designer of every frame — at the cost of giving up tight control ("these darn boids seem to have a mind of their own").

**Particle systems (Reeves, 1983).** The closest existing primitive. A particle system is a collection of a large number of individual particles, each carrying its own state — color, opacity, position, velocity — and its own simple behavior; particles are created, age, and die. They had been used for fire, smoke, clouds, and ocean spray. Two things are missing for flocking. First, a particle is a dot: it has no orientation, no body. Second, and more fundamentally, particles as described *do not interact with one another* — each evolves on its own. A flock member's behavior must depend not only on its internal state but strongly on the *external* state of its neighbors.

**A prior force-field foreflock (Amkraut, Girard, Karl, "Eurythmy," 1985).** Birds were driven by force fields: a 3×3 matrix operator mapping a point in space to an acceleration vector, with rejection forces around each bird and around static objects; the birds trace paths along the field's phase portrait. It produced a real flock-like film, but the underlying mechanism has known failure modes. A force field surrounding an obstacle does not reliably turn a bird aside: if the bird approaches exactly opposite to the field direction, the field only decelerates it head-on and provides no lateral thrust — and failing to turn is the worst possible response to an impending collision. Force fields are also too strong close up and too weak far away, so avoidance becomes a panicky last-minute correction rather than long-range planning, and they cannot express "peripheral vision" (turn from a wall you are flying *toward*, ignore one you are flying *alongside*).

**Geometric flight and turtle geometry (Logo; Papert 1980; 3D-turtle work of Abelson & diSessa).** The locomotion layer. Geometric flight is incremental translation along the object's own forward axis, interleaved with small steering rotations (pitch and yaw) that re-aim that axis — a discrete approximation of a continuously curving flight path. This is exactly the turtle geometry of Logo, lifted into 3D: an object that unites position and heading and moves forward / turns relative to its own frame. The flight model carries simple physics: conservation of momentum (an object in flight tends to stay in flight, which smooths abrupt changes of goal), viscous speed damping that imposes a maximum speed, and a maximum acceleration — expressed as a fraction of maximum speed — that truncates over-anxious acceleration requests, modeling a creature with a finite amount of available energy.

**The actor model (Hewitt).** The computational abstraction that fits a flock: an actor encapsulates process, procedure, and internal state, and communicates with other actors only by passing messages. It had been proposed as a natural structure for animation control and is especially apt for many interacting characters; in the distributed-systems literature, flocks and schools are themselves cited as examples of robust self-organizing distributed systems. Each bird is an actor running its own behavior program over its own state.

**What zoology says about how birds actually flock.** Natural flocks seem to consist of two balanced, opposing urges: a desire to stay close to the flock and a desire to avoid collisions within it (Shaw). The drive to join a flock is read as evolutionary — predator protection, larger effective foraging search, social and mating advantage. Crucially, flocks show *no* upper bound on size: herring schools run 17 miles long and contain millions of fish, and flocks operate the same way across a huge range of populations. That argues that an individual bird cannot be attending to every flockmate; it is doing something like a *constant-time* computation — aware of roughly three categories, *itself*, its *two or three nearest neighbors*, and *the rest of the flock* as an undifferentiated mass. Quantitative studies of schooling fish (Partridge) found that a fish is influenced far more by its near neighbors than its distant ones, with each neighbor's contribution falling off roughly as the inverse square or cube of distance — consistent with the way a visual silhouette's area and a pressure wave's intensity fall off with distance. And maneuver waves propagate across a flock faster than any individual's startle reaction time would allow, suggesting birds anticipate an oncoming turn and time their own to match it (Potts' "chorus line" hypothesis).

**The diagnostic failure of a global model.** An early flocking attempt used a *central-force* model — every bird pulled toward the flock's global centroid. This produces unnatural effects: all members of a widely scattered flock simultaneously converge on the single centroid, and the flock cannot *split* to flow around an obstacle and re-merge, because every bird is bound to one global center rather than to whoever happens to be near it. The observed lesson is that the motion we recognize as flocking *depends on each bird having only a limited, localized view of the world* — give a simulated bird perfect global information and the behavior model visibly fails.

## Baselines

- **Hand-scripted / keyframed flocks.** Draw and interpolate each bird's path individually. Gap: tedious and error-prone at flock scale, cannot maintain pairwise collision-freedom every frame, and is nearly impossible to edit globally.
- **Reeves particle systems (1983).** Many independent dot-particles with per-particle state and life cycle. Gap: particles have no orientation (no body) and, as used, do not interact — so they cannot express the neighbor-dependent coordination a flock needs.
- **Force-field flocks ("Eurythmy," 1985).** Birds and obstacles carry repulsion fields; birds trace the field's phase portrait. Gap: head-on approach to an obstacle field yields no turning (only deceleration); fields are too strong near and too weak far; cannot express peripheral-vision discrimination.
- **Central-force / global-centroid flock model.** Every bird is attracted to the flock's global center of mass. Gap: a scattered flock collapses to one point and the flock cannot bifurcate around obstacles; it presumes non-local information birds do not have.
- **Leader-following model.** All birds track a designated leader. Gap: like the central-force model, it cannot split, and it imposes a centralized structure foreign to real, leaderless flocks.

## Evaluation settings

The natural yardstick is qualitative and comparative: run the simulation and judge whether viewers spontaneously recognize the aggregate motion as *flocking*. The validity of such a simulation is hard to measure objectively, so the strongest available evidence is that observers immediately read the motion as a natural flock. Beyond that, the simulated motion can be checked against reported zoological criteria and statistical properties of real flocks and schools — the near-neighbor spacing distributions of schooling fish, the existence of maneuver-wave propagation, and the qualitative repertoire (polarization, flash expansion under crowding, coalescence of small flockettes into one flock, and bifurcation around obstacles). The simulation is deterministic and repeatable (a restartable random-number generator seeds initial positions within an ellipsoid, headings, and velocities), so behaviors can be reproduced and compared. The performance setting of record is a single-processor run: with 80 birds and the naive all-pairs algorithm (6400 bird-to-bird comparisons per frame), one frame takes on the order of a minute and a half, and a ten-second test runs overnight — so per-bird cost and its scaling with flock size are themselves part of the protocol.

## Code framework

A flight/locomotion substrate already exists: an oriented object that can do geometric flight (incremental forward translation plus pitch/yaw steering), carrying momentum, a maximum speed, and a maximum acceleration, with operators to convert between the object's local "bird's-eye" frame and the global frame. Each bird is an actor with its own state and a per-bird update called once per frame. What does not yet exist is the *behavior* that decides where to steer; that is the single empty slot below.

```python
import numpy as np

class Boid:
    """An oriented point-mass flyer: position + velocity, with speed/force caps
    inherited from the geometric-flight substrate."""
    def __init__(self, position, velocity):
        self.position = np.asarray(position, float)
        self.velocity = np.asarray(velocity, float)

def truncate(vector, max_magnitude):
    """Cap a vector's magnitude without rotating it (kinematic limit)."""
    m = np.linalg.norm(vector)
    if m > max_magnitude and m > 0:
        return vector / m * max_magnitude
    return vector

def neighbors(boid, flock):
    """The boid's local perception: which flockmates it is aware of.
    TODO: select by a distance radius and a view-angle (field of view)."""
    raise NotImplementedError

def steer(boid, flock):
    """The behavior to be designed: turn local perception of neighbors into a
    single steering force, respecting the max-acceleration budget.
    TODO: this is the contribution."""
    raise NotImplementedError

def step(flock, max_speed, max_force, dt):
    """One simulation frame: each actor computes its own steering, then the
    flight model integrates it (bounded acceleration, bounded speed)."""
    forces = [steer(b, flock) for b in flock]      # decide first, then move all
    for b, f in zip(flock, forces):
        acceleration = truncate(f, max_force)
        b.velocity = truncate(b.velocity + acceleration * dt, max_speed)
        b.position = b.position + b.velocity * dt
```
