Let me start from the thing that has bothered me longest, because I think the bother is the whole problem in disguise. Take a magnet and a conductor. Hold the conductor still, move the magnet: the changing magnetic field makes an electric field, and that electric field pushes the charges in the wire around — a current. Now hold the magnet still and move the conductor the other way, same relative motion: there is no electric field near the magnet at all, and yet the charges in the wire, because *they* are now moving through the magnetic field, feel a force and make the very same current, same path, same strength. The current you measure depends only on the relative motion of the two. But the story the theory tells depends on which one I declare to be "really" moving — an electric field in one telling, a magnetic force in the other. The phenomenon is symmetric; the description is not. That asymmetry is not in nature. It is in the bookkeeping. And the only thing that lets the bookkeeping distinguish the two cases is a notion of absolute rest — somebody's idea that there is a fact about which body is "truly" in motion.

I keep coming back to a second picture from when I was younger. Suppose I run after a light beam and catch up to it, moving at `c` alongside it. What do I see? By the ordinary rule for adding velocities, the wave's speed relative to me is zero, so I should see a frozen electromagnetic field — a static pattern, periodic in space, sitting still. But there is no such thing. Nobody has ever seen frozen light, and Maxwell's equations have no solution like that — no static, spatially-oscillating field standing in vacuum. So one of two things is wrong: either Maxwell's equations stop being true for the fast runner, or the picture of "catching up to light" is itself broken. And here is the part that feels like bedrock to me: if I am moving uniformly, I should not be able to tell. Uniform motion is exactly the kind of motion that no experiment inside the lab is supposed to reveal — that is the whole content of the relativity principle for mechanics, and I see no reason electrodynamics should be exempt. If the fast runner's electrodynamics differed from the slow observer's, he could detect his own steady motion just by watching light go wrong. That can't be allowed. So whatever the resolution is, the runner's laws must be the *same* laws.

So let me write down the two things I refuse to give up and see what it costs me.

First: the laws of physics — all of them, mechanics *and* electrodynamics — take the same form in every frame moving uniformly relative to another. No frame is privileged; no experiment picks out absolute rest. Call this the principle of relativity.

Second: light is always propagated in empty space with one definite speed `c`, and that speed does not depend on the motion of the source. I believe this because I believe Maxwell's electrodynamics is correct, and Maxwell's equations hand me `c` as a constant built out of `ε₀` and `μ₀` — a property of the medium-less field, not of any emitter's motion. The source can be rushing toward me or away; the wave it launches still goes at `c`.

Now put the two together and watch them collide. By the relativity principle, `c` is the same in my frame and in a frame gliding past me at `v`. But by the ordinary addition of velocities — the Galilean rule, where if you move at `v` then everything's speed shifts by `v` and time is the same shared `t` for everybody — the speed I measure for that same light pulse should be `c − v`, not `c`. The two postulates seem to collide. One says the number is `c` in both frames; the other says it has to differ by `v`.

This is the wall. I have spent the better part of a year here. My instinct was to keep the postulates and fix the electrodynamics — to follow Lorentz. Lorentz had already done something beautiful and unsettling. To explain why the moving Earth never reveals an ether wind, he showed that if you take the equations in the supposed rest frame of the ether and rewrite them for a system moving through it at `v`, then to first order in `v/c` they keep their form *provided* you feed them not the true time `t` but a doctored time
```
t' = t − (v/c²) x,
```
together with the shifted space coordinate `x' = x − vt`. He called `t'` the "local time." With it, a moving observer's optics looks exactly like a stationary observer's optics, so first-order ether-drift experiments come out null — as they do. And to kill the second-order Michelson–Morley null he added a separate hypothesis: bodies moving through the ether physically shrink along their motion by `√(1 − v²/c²)`. By 1904 he had the whole transformation, `γ = 1/√(1−v²/c²)` and all, with some leftover scale factor he couldn't pin down.

I tried for months to make this work — to take Lorentz's electron equations and simply demand that they hold in the moving frame too, the way the relativity principle would want. And I kept failing, because Lorentz's picture has a splinter in it I cannot remove. For him `t'` is a *trick*. It is an auxiliary mathematical time; the *real* time is still `t`, the ether's time, the one true time that everyone secretly shares. The contraction is a *dynamical accident* — a story about how electric forces squeeze matter when it plows through the ether. So he has two separate patches, both reaching toward the same algebra, both hanging on an ether frame that no experiment can find. I want the relativity principle to be exact and the ether to be *gone*, not hidden. As long as there is a true time `t` underneath, I have a privileged frame, and the magnet-and-conductor asymmetry comes right back. The splinter is the word "true."

So let me stop trying to modify the electrodynamics and ask a more dangerous question. The contradiction I derived — `c` in both frames versus `c − v` — rested on a step I never examined: that time is the same in both frames, that `t' = t`. Where did I get that? It is sitting in the Galilean rule as an unstated axiom. The absolute, universal, frame-independent character of *simultaneity*. Two events that happen "at the same time" — I have been assuming that is a fact about the world, the same fact for everybody. What if it isn't? What if the whole conflict is not between the two postulates at all, but between the two postulates and this one buried prejudice about time?

I need to be honest about what "time" even means before I can ask whether it's absolute. And the honesty has to be operational, because every judgment I make involving time is really a judgment about *coincidence*: "the train arrives at seven" means "the small hand pointing at seven and the train arriving are the same event, here, now." A single clock at one place dates only the events right next to it. The instant I want to talk about *one* time shared by two distant places — an event here and an event over there happening "at the same time" — I have said nothing yet, because I have given no procedure. There is no meaning to "simultaneous at a distance" until I lay down a rule for setting a clock at B against a clock at A.

What rule? I could station an observer with the A-clock and have him read remote events by the light that reaches him — but the time he assigns then depends on his distance and standpoint, and that won't do; it isn't a property of the events. The clean way uses the one thing both postulates make universal: light. Put a clock at A and an identical clock at B. Send a flash from A at A-time `t_A`; let it reflect at B at B-time `t_B`; let it return to A at A-time `t'_A`. I *define* the two clocks to be synchronized when
```
t_B − t_A = t'_A − t_B,
```
i.e. the light takes the same time out as back. That's it — that is what "the B-clock reads the same as the A-clock" is going to mean. It is a definition, not a discovery; there is nothing to discover until I make it. I'll assume it's consistent — symmetric (if B is synced to A then A is synced to B) and transitive (if A syncs with B and with C, then B and C are synced) — which it is. And it builds in that the one-way speed of light is `c`: the round trip `2·AB` over the elapsed `t'_A − t_A` is `c`.

Now I have the move. Simultaneity is *defined* by light-synchronization, and light's behavior is governed by the two postulates — so simultaneity will inherit whatever the postulates say, and there is no guarantee it survives a change of frame. Let me actually check, because this is the hinge of everything.

Take a rod of rest length `l` lying along `x`, moving at `v` in my frame, with clocks pinned to its two ends A and B. Suppose someone riding the rod synchronizes those two clocks by the light rule — flash from A, reflect at B, back to A. I, standing in the frame the rod moves through, watch the same flash, and for me light still goes at `c` (that's postulate two), while the rod's far end B is *running away* from the outgoing flash and its near end A is *running into* the returning flash. So in my frame:
```
forward leg (A → B):   t_B − t_A = r_AB / (c − v),
back leg   (B → A):    t'_A − t_B = r_AB / (c + v),
```
where `r_AB` is the rod's length as I measure it. And `c − v ≠ c + v`. The two intervals are *unequal*. So by the very definition of synchrony, the rod-rider's two clocks — which *he* set equal — are, *to me*, not synchronized. The events I call "the two clocks reading the same instant" are not simultaneous for him, and the events he calls simultaneous are not simultaneous for me.

There it is. Simultaneity is not absolute. It is frame-relative. Two events simultaneous in one inertial frame are not simultaneous in another moving relative to it. The buried axiom `t' = t` is simply false, and it was false in a way nobody noticed only because `c` is so enormous that for everyday speeds the discrepancy `r·v/c²` is invisible — light *looks* like it gives instantaneous, absolute simultaneity. The two postulates were never in contradiction. They only contradicted each other *through* the assumption of absolute time. Remove that assumption and the apparent conflict evaporates. The price of holding both postulates honestly is exactly this: give up the universal "now."

And Lorentz's local time `t' = t − vx/c²` stops being a trick the moment I see this. That `−vx/c²` is *precisely* the position-dependent offset you need so that clocks the moving observer considers synchronized are the ones that satisfy my light rule in his frame. It is not an auxiliary time. It *is* his time. There is no other, truer time underneath. The ether frame had nothing to stand on, because "true simultaneity" had nothing to stand on.

Now I can derive the transformation, and I expect it to be forced — no freedom left once I demand both postulates plus the synchronization definition. Let me set it up cleanly. My frame is `K` with coordinates `(x, y, z, t)`. The moving frame `k` glides at `v` along the shared `x`-axis, with coordinates `(ξ, η, ζ, τ)`. At `t = τ = 0` the origins coincide. A point sitting still in `k` has a fixed value of `x' ≡ x − vt`, so let me carry `x'` around as the natural co-moving coordinate.

First, the transformation must be linear in the coordinates. Space and time are homogeneous — no special point, no special instant — so a uniform motion has to map to a uniform motion and equal intervals to equal intervals; only linear functions do that. Good. So `τ` is some linear function of `(x', y, z, t)`.

Next I pin down `τ` by *demanding it be the time the `k`-observer gets from synchronizing his own clocks by the light rule I just defined*. From the origin of `k` shoot a ray along `+x'` to a point `x'`, reflect it, bring it back to the origin, arriving at `τ₂`, having reflected at `τ₁`, having left at `τ₀`. Synchrony says the reflection event is dated at the midpoint: `½(τ₀ + τ₂) = τ₁`. Now I write those `k`-times as the function `τ` evaluated at the right `K`-coordinates. The light leaves the `k`-origin (which sits at `K`-coordinate moving with the frame) and chases the point `x'`, which is receding, so in `K` the outbound trip takes `x'/(c − v)`; the return, with the origin now approaching, takes `x'/(c + v)`. So:
```
½ [ τ(0,0,0,t) + τ(0,0,0, t + x'/(c−v) + x'/(c+v)) ] = τ(x', 0, 0, t + x'/(c−v)).
```
Let `x'` be infinitesimal and expand to first order. The left side picks up `∂τ/∂t` times half of `x'/(c−v) + x'/(c+v)`; the right side picks up `∂τ/∂x'` times `x'` plus `∂τ/∂t` times `x'/(c−v)`:
```
½ ( 1/(c−v) + 1/(c+v) ) ∂τ/∂t  =  ∂τ/∂x' + 1/(c−v) ∂τ/∂t.
```
Now `½(1/(c−v) + 1/(c+v)) = ½ · (2c)/(c²−v²) = c/(c²−v²)`, and `1/(c−v) = (c+v)/(c²−v²)`. Subtract:
```
[ c/(c²−v²) − (c+v)/(c²−v²) ] ∂τ/∂t = ∂τ/∂x',
```
and `c − (c+v) = −v`, so the bracket is `−v/(c²−v²)`, giving
```
∂τ/∂x' + (v/(c²−v²)) ∂τ/∂t = 0.
```
The choice of the origin as the ray's launch point was arbitrary, so this holds at every point — it's a genuine constraint on `τ` everywhere.

What about `y` and `z`? Shoot the synchronizing ray along the `y`-axis of `k`. Viewed from `K`, that ray has to keep up with the transverse motion, so its `K`-frame speed along `y` is `√(c² − v²)` (it spends some of its speed `c` going sideways at `v`). Running the same midpoint argument with no `x`-dependence gives simply
```
∂τ/∂y = 0,   ∂τ/∂z = 0.
```
So `τ` doesn't depend on `y` or `z`, and being linear it must be
```
τ = a ( t − (v/(c²−v²)) x' ),
```
with `a = a(v)` an overall factor I don't yet know.

Now the spatial coordinates, and here the *second* postulate does the work — `ξ` has to be such that light measured in `k` also goes at `c`. A ray sent at `τ = 0` along `+ξ` satisfies `ξ = cτ` by definition of `k`'s own measurement. Substitute the `τ` I just found:
```
ξ = c · a ( t − (v/(c²−v²)) x' ).
```
But this same ray, watched from `K`, advances relative to `k`'s origin at speed `c − v`, so `x' = (c − v) t`, i.e. `t = x'/(c − v)`. Put that in:
```
ξ = a c ( x'/(c−v) − (v/(c²−v²)) x' ) = a c x' ( 1/(c−v) − v/(c²−v²) ).
```
Common denominator `c²−v²`: `1/(c−v) = (c+v)/(c²−v²)`, so the bracket is `(c + v − v)/(c²−v²) = c/(c²−v²)`, and
```
ξ = a c · x' · c/(c²−v²) = a (c²/(c²−v²)) x'.
```
The same exercise along `y` is cleaner if I just track the transverse light time. A ray that advances the transverse distance `y` in `k` has `x' = 0` and, as seen from `K`, needs `t = y/√(c²−v²)`. Then `τ = a t = a y/√(c²−v²)`, and since `k` measures that ray as `η = cτ`, I get `η = a c y/√(c²−v²)`. Likewise `ζ = a c z/√(c²−v²)`.

Now substitute `x' = x − vt` back into `τ` and `ξ`, and clean up the factors. Define
```
β ≡ 1/√(1 − v²/c²).
```
Watch what `c²/(c²−v²)` becomes: `c²/(c²−v²) = 1/(1 − v²/c²) = β²`. And `c/√(c²−v²) = 1/√(1 − v²/c²) = β`. And `v/(c²−v²) = (v/c²)/(1 − v²/c²) = β² v/c²`. So collecting, and folding the bookkeeping factors into a single function `φ(v)` (it'll absorb `a` and the various `β` powers so each equation comes out with a clean `φ(v)` in front of the relativistic core):
```
τ = φ(v) β ( t − v x / c² ),
ξ = φ(v) β ( x − v t ),
η = φ(v) y,
ζ = φ(v) z.
```
I still have this one unknown stretch factor `φ(v)` — exactly the leftover scale Lorentz couldn't fix. Let me kill it, because the postulates should leave nothing free.

Two arguments together do it. First, reciprocity. Introduce a third frame `K'` moving at `−v` relative to `k` — that is, apply my transformation twice: `K → k` at velocity `v`, then `k → K'` at velocity `−v`. Composing the equations (and noting `β(−v) = β(v)`), the `y` and `z` relations give `y' = φ(v)φ(−v) y`, with no time dependence in the spatial relation between `K` and `K'`. But a frame reached by going `+v` then `−v` is at rest relative to `K`; the round trip must be the *identity*. So
```
φ(v) φ(−v) = 1.
```
Second, isotropy. Look at a rod lying along the `η`-axis of `k`, from `η = 0` to `η = l` — a rod moving *sideways*, perpendicular to its own length, at speed `v`. Its `K`-length comes out `l/φ(v)`. But a rod moving perpendicular to its length can't care about the *sign* of `v`: flip the direction of motion and, by symmetry, nothing about a transverse length should change. So `l/φ(v) = l/φ(−v)`, i.e.
```
φ(v) = φ(−v).
```
Combine the two: `φ(v)² = 1` and `φ > 0`, so `φ(v) = 1`. The scale is forced to unity. (And notice this also says the transverse lengths `η = y`, `ζ = z` are completely unchanged — sideways motion doesn't contract anything. Only the longitudinal direction will.) The transformation is now rigid, nothing left to choose:
```
τ = β ( t − v x / c² ),
ξ = β ( x − v t ),
η = y,
ζ = z,        β = 1/√(1 − v²/c²).
```

Before I trust it I owe myself the consistency check I have not yet earned — the proof that the two postulates really are compatible, that I haven't smuggled a contradiction into the algebra. Send out a spherical light pulse from the common origin at `t = τ = 0`. In `K`, after time `t`, the wavefront is the sphere
```
x² + y² + z² = c² t².
```
Is it a sphere of speed `c` in `k` too? Substitute the transformation and grind it out. Take `ξ² + η² + ζ² − c² τ²` and express it in `K`-coordinates:
```
ξ² − c²τ² = β²(x − vt)² − c² β²(t − vx/c²)²
        = β² [ (x − vt)² − c²(t − vx/c²)² ]
        = β² [ x² − 2vxt + v²t² − c²t² + 2vxt − v²x²/c² ]
        = β² [ x²(1 − v²/c²) − t²(c² − v²) ]
        = β² (1 − v²/c²) [ x² − c² t² ]
        = x² − c² t²,
```
since `β²(1 − v²/c²) = 1`. The cross terms `−2vxt` and `+2vxt` cancelled cleanly — that cancellation is the whole point. Adding `η² + ζ² = y² + z²`:
```
ξ² + η² + ζ² − c²τ² = x² + y² + z² − c²t² = 0,
```
so `ξ² + η² + ζ² = c² τ²`. The pulse is *still* a sphere expanding at `c` in the moving frame. The two postulates are compatible; I built a kinematics in which light has the same speed in every inertial frame and no frame is special. (And I see now I could have run the derivation the other way around — *demand* that `x²+y²+z²−c²t²` be invariant and read off the transformation that does it. Same equations. But I prefer the route through the clock-synchronization definition, because that is where the physics lives: the invariance is a *consequence* of how I was forced to redefine time, not a postulate I pulled from the air.)

Now I want the physical consequences, because these are the things you can actually take to a lab — and they should drop out by just reading the transformation, no new assumptions.

Length contraction. Take a rigid sphere of radius `R` at rest in `k`: `ξ² + η² + ζ² = R²`. What shape is it in `K` at a single instant `t = 0`? Set `t = 0` in the transformation: `ξ = β(x − v·0) = βx`, `η = y`, `ζ = z`. So `β²x² + y² + z² = R²`, i.e.
```
x²/(1 − v²/c²) + y² + z² = R².
```
That is an ellipsoid with semi-axes `R√(1 − v²/c²)` along `x` and `R`, `R` along `y`, `z`. The moving sphere is squashed along its motion by the factor `√(1 − v²/c²)`, untouched sideways. A body that is a sphere at rest is, in motion, flattened in the direction of travel — and the faster it goes the flatter, vanishing into a disk as `v → c`. This is exactly Lorentz's contraction — but I did not *postulate* it as a dynamical squeezing by the ether. It fell out of the kinematics of measurement: "the length of the moving rod" *means* where its two ends are *at the same time in my frame*, and "the same time" is no longer what the rod-rider means by it. The contraction is a statement about simultaneity, not about forces in matter.

Time dilation. Watch a clock sitting at the origin of `k`, so its position in `K` is `x = vt`. Its own ticking is `τ`. From the transformation, with `x = vt`:
```
τ = β ( t − v(vt)/c² ) = β t (1 − v²/c²) = t (1 − v²/c²)/√(1 − v²/c²) = t √(1 − v²/c²).
```
So `τ = t√(1 − v²/c²) ≤ t`. The moving clock reads *less* elapsed time than the stationary frame's clocks — it runs slow. Per second of `t`, it falls behind by `1 − √(1 − v²/c²)`, which for small `v` is `≈ ½ v²/c²`. And there's a sharper consequence: take two clocks synchronized and sitting at A and B in `K`. Carry one of them from A to B at speed `v`; on arrival it lags the stationary one by `≈ ½ (v²/c²) t`, with `t` the travel time. The lag accumulates along the path, so at this small-speed order it holds for any polygonal route, and — if I grant the continuous limit — for any closed loop bringing the clock back to its start: the travelled clock returns *behind* the one that stayed. A clock carried around a closed curve at constant speed is slow on its return by `≈ ½(v²/c²)t`. (Which says, for instance, a clock at the equator, swept around by the Earth's rotation, should run a hair slower than an otherwise identical clock at a pole.)

Velocity addition. This is the piece that has to rescue the original collision — `c` in both frames yet velocities seeming to add to `c − v` — by replacing the Galilean rule with whatever the transformation actually implies. Let a point move uniformly in `k`: `ξ = w_ξ τ`, `η = w_η τ`, with `w_ξ`, `w_η` constant. What's its motion in `K`? Use the inverse transformation to write `x, y, t` in terms of the `k`-coordinates, or just push the `k`-velocities through. The `x`-velocity in `K` is `dx/dt`. From `ξ = β(x − vt)` and `τ = β(t − vx/c²)`, a co-moving point with `ξ = w_ξ τ` gives, after substitution,
```
x = ( (w_ξ + v) / (1 + v w_ξ / c²) ) t,
y = ( √(1 − v²/c²) / (1 + v w_ξ / c²) ) w_η t,
```
so the longitudinal composition law is
```
V_x = (w_ξ + v) / (1 + v w_ξ / c²),
```
and the transverse component carries an extra `√(1 − v²/c²)` and the same denominator. For the collinear case — `w` along `x` — the resultant is simply
```
V = (v + w) / (1 + v w / c²).
```
There is the replacement for the Galilean `v + w`. To first order in the velocities it *is* `v + w`, which is why ordinary life never noticed — the parallelogram of velocities is only a first approximation. But the denominator changes everything at the top end. Check that it cannot break the light barrier: set `v = c − κ`, `w = c − λ` with `κ, λ > 0` and less than `c`. Then
```
V = (2c − κ − λ) / (1 + (c−κ)(c−λ)/c²)  = c (2c − κ − λ) / (2c − κ − λ + κλ/c) < c,
```
because the denominator exceeds the numerator-over-`c` by the positive `κλ/c`. Two sub-light velocities always compose to something still below `c`. And the limiting case — compose `c` with anything `w < c`:
```
V = (c + w) / (1 + cw/c²) = (c + w) / (1 + w/c) = c (c + w)/(c + w) = c.
```
The speed of light, composed with any ordinary velocity, stays `c`. Exactly as it must — that's the second postulate reappearing as a fixed point of the composition law, and it is what makes "chasing a light beam" impossible: you can never gain on it, because adding your own speed to it just returns `c`. The frozen-light picture was never a real option. And `v` and `w` enter the general formula symmetrically, and composing two transformations of this kind gives a third one of the same kind with the velocity `(v + w)/(1 + vw/c²)` — the transformations form a group. Nice. That closure is the structural guarantee that "no privileged frame" is consistent: there's no end of the chain, no frame the composition law singles out.

So the whole thing hangs together from the two postulates and one honest definition. I held the principle of relativity and the constancy of light side by side; they looked irreconcilable; the irreconcilability lived entirely in the unexamined assumption that "now" is the same everywhere; I forced myself to *define* simultaneity operationally with light signals (`t_B − t_A = t'_A − t_B`), which immediately showed simultaneity is frame-relative; demanding that this definition give light the speed `c` in *both* frames forced the transformation `ξ = β(x − vt)`, `τ = β(t − vx/c²)` with `β = 1/√(1 − v²/c²)`, the scale factor pinned to one by reciprocity and transverse isotropy; that transformation leaves `x² + y² + z² − c²t²` invariant, so the postulates are compatible; and reading it off gives length contraction by `√(1 − v²/c²)`, clocks slowed to `t√(1 − v²/c²)`, and the velocity-addition law `(v + w)/(1 + vw/c²)` that keeps `c` unsurpassable. The ether is gone; there was never a true time to host it. The magnet and the conductor now tell one story — relative motion, one current — because there was never a fact about which one was "really" moving.

I can encode the final kinematics directly and check the identities it forced:

```python
import numpy as np

C = 1.0  # units with the speed of light equal to 1

def gamma(v):
    # beta = 1/sqrt(1 - v^2/c^2); the factor forced by the two postulates
    return 1.0 / np.sqrt(1.0 - (v / C) ** 2)

def synchronize_event_time(t_emit, t_return):
    # operational simultaneity-at-a-distance: date a remote event at the midpoint
    # of light emission and reflected return  ->  t_B - t_A = t'_A - t_B
    return 0.5 * (t_emit + t_return)

def lorentz_transform(x, t, v):
    # forced by: laws identical in both frames (P1) + light speed c in both (P2),
    # once simultaneity is defined by the light-signal rule above.
    #   xi  = beta (x - v t),     tau = beta (t - v x / c^2)
    b = gamma(v)
    xp = b * (x - v * t)
    tp = b * (t - v * x / C ** 2)
    return xp, tp

def length_contraction(rest_length, v):
    # the moving rod's ends, located at one instant of the rest-through frame:
    # only the longitudinal dimension shrinks, by sqrt(1 - v^2/c^2).
    return rest_length * np.sqrt(1.0 - (v / C) ** 2)

def moving_clock_time(t, v):
    # clock at the moving origin (x = v t):  tau = t sqrt(1 - v^2/c^2)  -> runs slow
    return t * np.sqrt(1.0 - (v / C) ** 2)

def add_velocities(w, v):
    # composition law from the transformation; replaces the Galilean v + w.
    # collinear case; never exceeds c for sub-c inputs, and returns c when w = c.
    return (v + w) / (1.0 + v * w / C ** 2)

# self-consistency checks the derivation guarantees:
if __name__ == "__main__":
    v = 0.6
    # (1) the interval x^2 - c^2 t^2 is invariant, so a light pulse
    #     (x^2 = c^2 t^2 in K) stays a pulse of speed c in the moving frame:
    for (xx, tt) in [(0.6, 1.0), (-0.3, 0.5), (0.9, 0.95)]:
        xp, tp = lorentz_transform(xx, tt, v)
        lhs = xx**2 - (C*tt)**2
        rhs = xp**2 - (C*tp)**2
        assert abs(lhs - rhs) < 1e-12
    # (2) light composed with any velocity is still light:
    assert abs(add_velocities(C, 0.3) - C) < 1e-12
    # (3) two sub-c velocities stay sub-c:
    assert add_velocities(0.9, 0.9) < C
    # (4) contraction and dilation share the same sqrt factor:
    assert abs(length_contraction(1.0, v) - moving_clock_time(1.0, v)) < 1e-12
    print("Lorentz transform consistent with both postulates.")
```
