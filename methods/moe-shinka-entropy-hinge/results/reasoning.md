The global-batch loss landed where I expected — imbalance in the same good band as the Switch loss,
cross-entropy no worse — and the previous rung ended by naming exactly what it leaves undone. The
smooth `f·P` penalty equalizes the *average* usage, but it treats an expert at a tenth of its fair
share the same as one at nine-tenths: both are just small terms in the sum, so the gradient that
would resurrect a nearly-dead expert is weakest precisely in the tail where I need it most.
Balancing the mean is not resurrecting the dying. I want a term that singles out the under-utilized
experts and pushes specifically on them — and I want to be careful about it, because the obvious
ways to do that are exactly the ways to wreck specialization.

Let me think about what a targeted under-use penalty should and should not do. It should fire on an
expert that has fallen below some floor of usage, and it should leave alone every expert that is at
or above its fair share. That immediately suggests a one-sided penalty — a hinge: penalize
`floor − f_i` only when it is positive, zero otherwise. An expert comfortably above the floor
contributes nothing; an expert below it contributes in proportion to how far below. That is the
shape I want: it ignores the healthy experts entirely and concentrates all its force on the cold
tail, which is the opposite of the smooth `f·P` term that spreads its attention evenly. The floor
itself should be a small fraction of the uniform share — I do not want to demand that every expert
hit `1/N` exactly, only that none be allowed to wither below a minimum. Something like a few percent
of `1/N` as the per-expert usage floor.

But a hinge by itself is dangerous, and I have to see the danger before I trust it. If the hinge
fires hard whenever an expert is below the floor, it will fire even when the router is already
healthy and the under-use is just benign variation — a momentary dip, the natural roughness of a
finite batch. Pushing hard in that situation would do exactly what the micro-batch loss did:
flatten legitimate structure, raise the cross-entropy. So the hinge needs a gate that asks *is this
a real collapse, or just normal variation?* And there is a clean signal for that already in the
router: the entropy of its probability distribution over experts. When the router is peaked — low
entropy, most mass on a few experts — that is the collapse regime, and a cold expert really is being
starved; the rescue should be strong. When the router is near-uniform — high entropy — the system is
healthy, any momentary under-use is noise, and the hinge should barely fire. So I want to weight the
hinge by something that is large when entropy is low and small when entropy is high: a complement of
the normalized entropy.

Concretely, normalize the router entropy by its maximum (the log of the number of experts, the
entropy of the uniform distribution), so it runs in zero-to-one; take one minus that, so peaked
routers score near one and uniform routers near zero; and offset it by a half so the weight is never
quite zero and the floor is always at least gently enforced. That gives a peakedness weight that is
roughly one-and-a-half when the router has collapsed and around a half when it is uniform — exactly
the modulation I want. The hinge, scaled by this weight, becomes a collapse-triggered rescue: it
waits, mostly idle, and only when the router peaks and experts start dying does it surge to pull the
cold ones back. It is the targeted, self-limiting term the global-batch loss was missing.

Now, I should be candid about the status of this construction. I did not derive these exact pieces —
the half-offset, the particular floor of about six percent of `1/N`, the tenth weighting on the
hinge relative to the global term — from first principles. They are the artifacts of an
*evolutionary search* over the loss function itself: ShinkaEvolve evolved the Python of the
balancing loss, scored by the very fitness I am using here, the negative of cross-entropy plus
imbalance, on real MoE pretraining, and this is the loss it converged to. So my reasoning above is
the reconstruction of *why* the discovered form makes sense, not the path that found it; the search
found a specific, slightly unusual set of constants, and what I can verify is that each piece plays
the role the mechanism needs — the hinge for targeting, the floor for the threshold, the entropy
complement for the collapse gate, the small coefficient for not overwhelming the global term. The
two parts together are: the global-batch `f·P` term, averaged over layers with the experts-times-one
scale, plus a tenth-weighted, entropy-modulated hinge on the under-floor experts, also averaged over
layers.

One implementation point I have to get right or the term does nothing: the hinge is written on the
count `f_i`, which is non-differentiable, so as written its gradient is zero just like the bare
count penalty of the first balancing rung. The count can only *select* which experts are under the
floor; the gradient that actually raises a cold expert's usage has to flow through the differentiable
probability `P_i` of those selected experts. So I let `f` decide membership in the under-used set and
apply the differentiable pressure to the `P` of that set — push the router's probability mass up on
exactly the experts the floor test flagged as dying. That is what makes the hinge a real training
signal rather than a decorative zero.

So the rung is the discovered endpoint: keep the global-batch term unchanged, add the
entropy-weighted under-utilization hinge on top, with the floor and weights as the search fixed them,
and the hinge gradient routed through the under-used experts' probabilities. I expect this to hold
the cross-entropy where the global-batch loss had it — the hinge is idle when the router is healthy,
so it should not cost specialization — while pushing the imbalance lower than the global term alone
managed, because now the cold tail is actively rescued rather than merely averaged over. The best
joint point of cross-entropy and imbalance, the best fitness, should be here. And this is where the
ladder stops: this is the loss a dedicated program-evolution search discovered against this exact
fitness, on a 556M-active-82M MoE over two-billion-plus FineWeb tokens; my run is a small
reproduction of its mechanism, not its scale, and there is no further hand-designed rung above it
that I have reason to believe does better on the joint objective.
