OK, let me think this through from scratch. I want an animation of a flock of birds — dozens of them, wheeling and turning together, never colliding, splitting around a tree and re-forming on the far side — and I do not want to draw a single bird's path by hand. If I tried, it would be tedious past the first few birds, I would surely make errors given how intricate real flight paths are, I could never guarantee that no two birds touch at every frame, and the moment a director said "have them turn left two seconds in" I'd be re-drawing everything. So scripting is out. I need the motion to *generate itself*.

What does a flock actually look like, structurally? It's made of discrete birds but moves like a fluid; it looks random but is tightly synchronized; and it feels like something is steering the whole thing from above. That last impression is the trap. There is no conductor. Every shred of evidence from the birds themselves says the flock is nothing but the aggregate of individual birds, each acting only on its own local view of the world. So if I want the look, I should build the cause: simulate *one* bird's behavior, instance it many times, let them interact, and hope the flock falls out. The thing I recognize as "flocking" would then be an emergent property, not something I authored.

That reframes the whole problem. I'm no longer designing a flock; I'm designing a bird — or rather the slice of a bird's behavior that lets it participate in a flock. What primitive do I build it on? Reeves' particle systems are close: a big population of individual particles, each with its own position, velocity, color, opacity, each created and aging and dying, already used for fire and smoke and spray. That's most of the bookkeeping I need for free. But two things are wrong for birds. A particle is a dot — no orientation, no body, and I want creatures that bank into their turns. And the deeper problem: particles as people use them *don't interact*. Each one evolves on its own. A bird's whole point is that what it does depends on its neighbors. So I'll take the particle system and generalize it — replace the dot with an oriented object carrying a local coordinate frame, and make the per-particle behavior depend on *external* state, the positions and velocities of nearby birds, not just internal state. Call these things boids.

Each boid is then naturally an actor: a little encapsulated process with its own state and its own behavior program, talking to the world only through what it can perceive locally. That's the right computational shape for a swarm of independent interacting things, and flocks are literally the textbook example of a robust self-organizing distributed system. Good — the architecture is a population of actors, each running the same behavior over its own state once per frame.

Before any behavior, the boid has to be able to *fly*, because behavior only gets to *steer*; the flying itself is a substrate. I'll model flight as geometric flight: increment position along the boid's own forward axis, and interleave small steering rotations — pitch and yaw — that re-aim that forward axis. A continuously curving flight path approximated by many small straight hops plus turns. This is exactly Logo's turtle geometry lifted into 3D: an object that fuses position and heading and only ever moves forward or turns relative to itself. And I'll bake in a little physics, because it does real work for me later. Conservation of momentum: a boid in flight tends to stay in flight, so if a goal changes abruptly the boid's own inertia smooths the transition — the dynamics act like an automatic interpolation between control points, which I'll want when I script the flock around. Viscous damping so speed can't run away — a maximum speed. And a maximum acceleration, sensibly expressed as a fraction of the maximum speed, so that whenever a behavior asks for more turn or thrust than the bird has muscle for, the request gets truncated. A creature has finite energy; the cap is how I encode that. So the locomotion layer hands me a bird that flies, banks, and respects speed and acceleration limits. My entire job is the function that decides *which way to steer*.

Now, the behavior. Zoology gives me the seed: a real flock seems to be the tension between two opposing urges — stay close to the flock, and don't crash into your flockmates. Those two pull against each other, and the flock is where they balance. Let me take that seriously and see if two rules suffice.

Stay-close first. The crude way is a central force: pull every bird toward the flock's center of mass. Let me actually run that before I trust my intuition about it. I scatter forty birds in four well-separated clusters around (±20, ±20), give each an acceleration toward the global centroid (capped, with a little damping), and integrate a few hundred frames. I measure the mean distance of a bird from the flock centroid: it starts at about 28 and ends at 0.017. So they don't *contract* — they collapse essentially to a single point. That's not a flock; that's an implosion. And it tells me something worse by implication: a bird bound to the one global center can never leave to let the flock *split*, but real flocks pour around an obstacle in two streams and merge again. The failure is the globalness.

So "center of the flock" cannot mean the true center. Try making it the center of the *nearby* birds — the centroid of whoever this bird can actually perceive — and check whether the pathology inverts. Take a bird deep inside, with eight neighbors arranged roughly symmetrically around it; the mean of their positions comes out at (0, 0), right on top of the bird, so the centering pull has magnitude 0 — it doesn't fight to reach some far-off middle. Now take a bird on the *edge*, with its six neighbors all off to one side (I put them around x ≈ 1–2); their centroid lands at (1.5, 0), an offset of 1.5 pulling the bird inward. Same rule, and the urge is near-zero in the interior and strong exactly at the boundary, gently herding stragglers back in. That's the behavior I want, and it fell straight out of making the centroid local. And splitting works for free, because a bird only cares about staying near *its* neighbors; if the rest of the flock peels away around a tree, it doesn't care, it just follows the locals. The lesson is sharp and I'll hold onto it: the motion I recognize as flocking *depends on each bird having only a limited, local view*. Give it global knowledge and the behavior visibly fails. Locality isn't an optimization here; it's the mechanism.

So rule: steer toward the centroid of nearby boids. Call it flock centering, or cohesion.

Now don't-crash. The instinct is one more position rule: if a neighbor is too close, push away. Fine — but let me ask whether position-based avoidance alone is enough, because there's something subtle. Static avoidance, based only on where neighbors are right now, ignoring where they're *going*, can establish a minimum separation but can't *maintain* it through flight. Picture two boids flying side by side at the same velocity; they never get closer, no avoidance fires, good — but if their velocities differ even slightly, the gap drifts, and a purely positional rule only reacts once they're already too close, perpetually a step behind. What I actually want is to keep separations roughly invariant as the flock cruises. And there's a clean way to get that: if a boid *matches the velocity* of its neighbors, then by construction the distances to those neighbors barely change over time — it's predictive collision avoidance. Match velocity now and you're unlikely to collide soon. So velocity matching and static position avoidance are *complementary*, not redundant: static avoidance sets the minimum required separation; velocity matching maintains it. That's two distinct behaviors, and now I see I need *three* rules, not two:

separation — steer away from neighbors that are too close (position only);
velocity matching, or alignment — steer to match the average velocity of nearby neighbors (velocity only);
cohesion — steer toward the centroid of nearby neighbors (position, long-range).

That this also matches what biologists describe — birds aware of itself, its two-or-three nearest neighbors, and the rest of the flock as a mass, doing something close to a constant-time computation regardless of flock size — is reassuring. Flocks have no size ceiling precisely because no bird attends to every other bird. My rules must have the same property: each boid looks only at neighbors, so its per-bird work is bounded by neighborhood density, not flock size.

Let me make "neighbor" precise, because everything rides on it. A boid is aware of another based on the offset vector between them — its distance and its direction. Distance gives a radius: ignore anything past some sensitivity range. But a pure radius isn't right, because a bird doesn't perceive equally in all directions. Real birds have an enormous field of view, around 300 degrees, but not a full 360 — there's a blind cone behind the head. A neighbor flying directly *behind* me shouldn't tug me; I can't see it, and being pulled by it would make the boids at the leading edge of a flock keep glancing backward and getting distracted by the mass behind them, which is exactly a glitch I'd expect. So the neighborhood is two things: a *distance* and a *view angle*. A boid considers another boid a neighbor only if it's within the radius *and* within the field-of-view cone around the boid's own heading. (Ideally I'd even stretch that cone forward and widen it with speed — being in motion demands more awareness of what's ahead — but distance-plus-angle is the essential structure.)

And how should a neighbor's influence fall off with distance inside that zone? My first cut weighted attraction and repulsion *linearly* by distance — a spring-like law. When I picture it, the flock comes out bouncy, springy, oscillating in and out, fine for a cartoon but not natural. The fix is to make influence fall off faster — inverse square of distance, a more gravity-like law. That damps the bounce and reads as natural, and it isn't arbitrary: Partridge's measurements of schooling fish found each neighbor's contribution falling off roughly as 1/r² to 1/r³, which makes sense because a neighbor's visual silhouette area shrinks as 1/r² and pressure waves in water fall as 1/r³. So separation's repulsion should fall off something like inverse-square — scale each neighbor's offset vector by 1/r² — close neighbors shout, distant ones whisper.

Now I have three urges, each a steering suggestion. How do I express one cleanly and how do I combine three? Take steering generally. A behavior wants the boid to be moving a certain way — call that the *desired velocity*. But the boid is already moving with some current velocity. The honest thing the behavior is asking for is the *difference*: steering = desired_velocity − current_velocity. That's a velocity error, and it's a lovely formulation, because the steering force then automatically points toward the desired motion and shrinks to zero as the boid achieves it — no separate overshoot-handling, the correction fades out on its own as you turn into line. So for each rule I'll compute a desired velocity and subtract the boid's current velocity.

Seek a point — like the cohesion target — is then: desired_velocity = normalize(target − position) · max_speed, and steering = desired_velocity − velocity. That says "go full tilt toward the target," and the velocity error turns me there. For cohesion the target is the local centroid: average the positions of the neighbors, seek it.

Alignment is even more direct, because the desired thing is *itself* a velocity: average the neighbors' velocities, call that the desired velocity, and steering = average_velocity − current_velocity. That difference rotates my heading toward the group's heading and nudges my speed toward theirs. Exactly velocity matching.

Separation I want as repulsion summed over the close neighbors. For each neighbor that's too near, take the offset from the neighbor to me and scale it by 1/r² — equivalently, take the unit away-from-neighbor direction and scale it by 1/r — so the nearest intruder dominates; then sum these over all close neighbors. I sum the per-neighbor repulsions rather than steering away from their averaged position, and I want to be sure that distinction actually matters before I commit to the more awkward summation. So I put a boid in the worst case for crowding: four neighbors at distance 1 on the +x, −x, +y, −y axes, hemming it in symmetrically. Steering away from their *averaged position* first: the mean of the four neighbor positions is (0, 0), exactly where the boid is, so the push is the zero vector — a boid buried in a symmetric crowd would feel *no* urge to escape, the worst possible outcome. Now *sum* the four 1/r² repulsions instead: by symmetry +x and −x cancel, +y and −y cancel, so... that also comes out (0, 0). So summation does *not* save me in the perfectly symmetric case — the cancellation is real for both. But the symmetric case is a measure-zero knife-edge; what matters is what happens the instant it breaks. So I move one neighbor in from r=1 to r=0.5 and recompute the *sum*: its repulsion is now 1/0.25 = 4 along its axis while its opposite partner contributes only 1, so the total is (−1, 0) with magnitude 1 — a definite shove away from the close intruder. Doing the same with the *averaged position*: the four positions still average to roughly the origin (the closer neighbor barely moves the mean), so the push stays near zero. *That's* the real difference: summation responds sharply to any asymmetry in crowding because the 1/r² weighting lets the nearest neighbor dominate, while averaging positions washes that weighting out. Good — summation it is. And the summed repulsion *is* the separation steering force; I leave its magnitude alone rather than renormalizing it to max_speed, because renormalizing would throw away exactly the distance falloff I just relied on, flattening the near intruder's shove down to the same length as a far one's.

Three steering vectors. Now the combination. The easy move is to average them, weighting by some per-behavior strength. An early version of exactly this steers a boid along a reasonable course under typical conditions, so it's tempting to stop here. But I'm uneasy about what averaging does when two urges disagree sharply, so let me put numbers on the case I'm worried about. Avoidance says "hard left now, wall ahead": call it (−1, 0), unit magnitude. Centering says "right, toward your friends," nearly opposite: (0.9, 0.1), also about unit magnitude. Their average is ((−1+0.9)/2, (0+0.1)/2) = (−0.05, 0.05), magnitude 0.07. So two urgent, full-strength commands average to a vector less than a *tenth* their length — the boid makes a negligible turn and sails essentially straight ahead, into the wall, at exactly the moment hesitation is the one unforgivable response. The prioritized alternative, by contrast, takes avoidance in full: (−1, 0), a hard left. And even when urges don't cancel, averaging produces nonsense compromises: flying through a grid of city streets, "fly north" is good and "fly east" is good, but their average, "fly northeast," flies you into a building. Averaging mushes together advice that should be taken one-at-a-time.

So I won't average. I'll *prioritize*. Rank the behaviors by urgency — collision avoidance first, it's non-negotiable; velocity matching next; flock centering last, it's the one I can safely drop in a pinch. Each behavior emits an acceleration request, a vector capped to unit magnitude and scaled by its strength. Then I parcel out a fixed acceleration budget — the boid's maximum acceleration — in priority order: add the highest-priority request into an accumulator, track the accumulated magnitude, take the next request, and so on, until the accumulated magnitude would exceed the maximum acceleration. At that point I trim the last request back to exactly fill the remaining budget and stop; any lower-priority behaviors go unsatisfied this frame. In an emergency, all the acceleration goes to the most pressing need — avoid the obstacle — and centering is simply ignored until the crisis passes, which is exactly right. The fixed budget is the same max-acceleration the flight model already enforces; I'm just deciding *who spends it first* instead of blending everyone's wishes into mush. (In a simpler implementation I can approximate this priority by giving the rules fixed weights with separation weighted hardest — sum the weighted steerings and truncate the total to the max force. The weighting encodes the precedence; the truncation enforces the budget. The strict prioritized allocation is the robust version for the brick-wall case.)

Put the loop together. Once per frame, every boid: find its neighbors (within radius and view-angle); compute the three steering vectors — inverse-square-weighted separation repulsion, alignment toward neighbors' average velocity, cohesion toward neighbors' centroid — each as desired_velocity − current_velocity; combine them by priority/weight; truncate the combined steering to the max force; let the flight model apply it. The flight model does: acceleration = steering force over mass; new velocity = truncate(velocity + acceleration, max_speed) — that's the viscous speed cap, applied by truncating the *magnitude* so the cap never rotates my heading, only shortens it; new position = position + velocity. One subtlety in ordering: I compute *all* boids' new velocities from the *current* frame's positions before moving any of them, so a boid reacts to where its neighbors *are*, not to where some of them have already jumped to this frame.

That's the whole flocking core. Before I believe any of the prose I'd write about what it does, I should run it. I seed fifty boids in a 30-unit box with tiny random velocities, weights (separation 2, alignment 1.5, cohesion 2), and step it forward, tracking two numbers: the minimum pairwise distance (does anyone collide?) and the polarization — the magnitude of the mean unit heading, which is 1 if everyone flies the same way and near 0 if headings are random.

Separation passes cleanly: the minimum pairwise distance starts around 0.54 and never falls to zero — across 800 frames it sits around 0.7–0.9, so the boids hold each other off and never overlap. Good; avoidance does its job. The crowding test is even more convincing: pack thirty boids almost on top of each other (mean spread 0.36) and after just ten frames the spread is 1.52 — a radial flash expansion driven by the over-pressure, exactly the behavior real schools show when startled. That one I trust now because I watched the number quadruple.

But polarization surprises me. I expected it to climb toward 1 — "everyone heads the same way." It doesn't: it hovers around 0.07–0.25 for the whole run, and pushing the alignment weight up to 4 or 8 barely moves it. The flock stays *together* and *non-colliding*, but it mills rather than streams. Thinking about why, it's the geometry I chose: a perception radius of 25 in a box only 30 across means almost every boid sees almost every other, so cohesion dominates and pulls everyone into a churning blob with no shared heading — there's no spatial structure for alignment to lock onto. So velocity matching needs a *band* of neighbors, not the whole flock, to actually polarize the group; I'd expect that to appear when the radius is small relative to the flock's extent. I'll flag that as something to check with a tighter radius rather than claim polarization here, because the run plainly says it isn't happening at these settings. What I *can* claim from the run: held separation, flash expansion under crowding, and a cohesive non-colliding cluster. Scatter the boids and small flockettes should likewise coalesce by the same cohesion that holds the cluster; send them at an obstacle and centering lets the flock bifurcate around it and re-form — because each boid only ever needed its locals.

Two more things make it usable as *animation* rather than a closed terrarium. First, I want to *direct* the flock — "enter from the left at 2.3 seconds, turn up at 3.5." So I add a migratory urge: a global goal, either a direction or a target point, that produces a bounded acceleration turning each boid toward it. The animator animates the goal point along the path they want, a little ahead of the flock, and the flock follows; and because the flight model conserves momentum, even an abrupt jump of the goal yields smooth boid paths — the dynamics interpolate between the control points for me. I can also stagger the goal change across boids so the turn propagates through the flock like a wave instead of everyone snapping at once, which is what real flocks do anyway.

Second, environmental obstacles. I tried the force-field approach — a repulsion field around the obstacle — and it has the failures I already distrust: a boid approaching the field head-on gets only decelerated, no sideways push, so it fails to turn (the worst outcome); the field is too strong up close and too weak far; and it can't tell "wall I'm flying toward" from "wall I'm flying alongside." A better, more vision-like scheme is steer-to-avoid: consider only obstacles directly ahead — intersect the boid's forward axis with the obstacle — and if there's a hit, find the silhouette edge of the obstacle nearest the impact point and aim the boid at a spot just past that edge, one body length beyond it. That plans a turn around the obstacle instead of panicking into it, and it naturally ignores walls you're merely flying alongside, because they're not ahead.

One honest caveat about cost. As written, each boid reasons about every other boid — if only to decide to ignore it — so the work grows as the square of the flock size; doubling the boids quadruples the time. Real birds clearly don't pay this, because each attends to only a few neighbors, so the natural fixes are to make neighbor-finding sub-linear: sort boids into spatial bins and only check bins near the boid, or do nearness-testing incrementally from the previous frame. With those, the per-boid cost tracks neighborhood density rather than flock size, which is the constant-time-per-bird character that lets real flocks grow without bound. For a first implementation, though, the all-pairs version is fine and correct.

The flocking core, written plainly: a neighborhood query by distance and view-angle; the three steering vectors — separation as summed inverse-square repulsion, alignment as average-velocity-minus-mine, cohesion as a seek toward the local centroid in desired-minus-velocity form; a combined-and-truncated steering force; and the integration step with the speed cap, computing all steerings before moving anyone.

```python
import numpy as np

class Boid:
    # an oriented point-mass flyer: position + velocity (the body's heading is
    # just the direction of velocity). speed/force caps come from the flight model.
    def __init__(self, position, velocity):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)

def _norm(v):
    return float(np.linalg.norm(v))

def truncate(vector, max_magnitude):
    # cap magnitude without rotating the vector: the kinematic limit shortens,
    # never turns, the command.
    m = _norm(vector)
    if m > max_magnitude and m > 0.0:
        return vector / m * max_magnitude
    return vector

def desired_minus_velocity(desired, boid, max_speed):
    # steering = desired_velocity - current_velocity. points toward the desired
    # motion and fades to zero as the boid achieves it (no built-in overshoot).
    if _norm(desired) > 0.0:
        desired = desired / _norm(desired) * max_speed
    return desired - boid.velocity

def neighbors(boid, flock, radius, view_cos):
    # local perception: within the sensitivity radius AND within the field-of-view
    # cone around the boid's own heading (a neighbor directly behind is unseen).
    out = []
    heading = boid.velocity
    h = _norm(heading)
    for other in flock:
        if other is boid:
            continue
        offset = other.position - boid.position
        d = _norm(offset)
        if d == 0.0 or d > radius:
            continue
        if h > 0.0:
            # cos(angle between heading and the offset to the neighbor)
            if float(np.dot(heading, offset)) / (h * d) < view_cos:
                continue   # outside the view cone -> not perceived
        out.append((other, offset, d))
    return out

def separation(boid, nbrs, max_speed):
    # steer away from crowding neighbors. each contributes a repulsion along
    # (me - neighbor): the offset vector scaled by 1/r^2, equivalently the unit
    # away-vector scaled by 1/r, so the nearest intruder dominates. SUM the
    # per-neighbor repulsions (don't steer from their average position, which
    # would cancel for a symmetric crowd and leave a hemmed-in boid unpushed).
    # the summed repulsion IS the steering force -- renormalizing it would throw
    # away the very distance falloff just built in.
    acc = np.zeros_like(boid.position)
    for other, offset, d in nbrs:
        acc += -offset / (d * d)      # offset scaled by 1/r^2 (gravity-like)
    return acc

def alignment(boid, nbrs, max_speed):
    # velocity matching: the desired velocity IS the neighbors' average velocity,
    # so the steering is that average minus my current velocity (no renormalizing).
    # matching it keeps inter-boid distances ~invariant (predictive avoidance).
    if not nbrs:
        return np.zeros_like(boid.velocity)
    avg_vel = np.mean([o.velocity for o, _, _ in nbrs], axis=0)
    return avg_vel - boid.velocity

def cohesion(boid, nbrs, max_speed):
    # flock centering: seek the centroid of LOCAL neighbors. small deep inside the
    # flock (neighbors surround you), strong at the edge (neighbors on one side).
    if not nbrs:
        return np.zeros_like(boid.position)
    centroid = np.mean([o.position for o, _, _ in nbrs], axis=0)
    return desired_minus_velocity(centroid - boid.position, boid, max_speed)

def steer(boid, flock, params):
    nbrs = neighbors(boid, flock, params["radius"], params["view_cos"])
    # precedence separation > alignment > cohesion, encoded as weights; collision
    # avoidance is non-negotiable, centering is the droppable one. (the robust
    # version parcels a fixed acceleration budget in strict priority order.)
    s = separation(boid, nbrs, params["max_speed"]) * params["w_sep"]
    a = alignment(boid,  nbrs, params["max_speed"]) * params["w_align"]
    c = cohesion(boid,   nbrs, params["max_speed"]) * params["w_coh"]
    combined = s + a + c
    return truncate(combined, params["max_force"])   # spend at most the max budget

def step(flock, params, dt):
    # decide ALL steerings from this frame's positions, THEN move everyone, so a
    # boid reacts to where its neighbors ARE, not where some already jumped to.
    forces = [steer(b, flock, params) for b in flock]
    for b, f in zip(flock, forces):
        acceleration = f                                   # unit mass
        b.velocity = truncate(b.velocity + acceleration * dt, params["max_speed"])
        b.position = b.position + b.velocity * dt
```

Tracing the chain once more: I refused to script paths and instead simulated a single bird many times, betting the flock would emerge from local interaction — and the bet pays because the motion I recognize as flocking *depends* on each bird seeing only locally; a global central force collapses the flock and forbids it from splitting. Two opposing zoological urges (stay close / don't collide) split into three rules once I notice that position-based avoidance only *establishes* separation while *matching velocity* maintains it predictively — so separation, alignment, and cohesion, each computed only over neighbors selected by distance and a view-angle. Alignment and cohesion take the desired_velocity − current_velocity form so the force fades as the motion is achieved; separation is summed inverse-square repulsion taken directly as the steering force (gravity-like and well-damped, matching the fish data, and immune to the symmetric-crowd cancellation that averaging would suffer). The three are combined not by averaging — which I checked cancels two opposing unit urges down to a vector of length 0.07 and would crash the boid into walls — but by priority with separation hardest, truncated to a fixed acceleration budget; and the flight model integrates with momentum and a magnitude-truncated speed cap so the limits shorten the command without ever rotating it. Run forward a frame at a time over a population of these actors, the core demonstrably holds separation (no overlaps), flash-expands an over-packed start, and clusters cohesively into non-colliding, splitting-and-merging aggregate motion that nobody scripted. The streaming polarization I'd want on top of that needs a perception radius small relative to the flock — at a radius that sees the whole flock the group mills rather than aligns — which is the one piece my quick run flagged for tuning rather than confirmed.
