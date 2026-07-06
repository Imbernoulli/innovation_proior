The continuous-filter floor told me exactly what its ceiling is, and it told me in three numbers I want to
read carefully rather than at a glance. EC came in at 0.589 accuracy — respectable, because the enzyme
class correlates with broad active-site geometry and with the amino-acid composition the 28-dim node
features already carry, so a radial encoder has plenty to read there. GO-BP landed at 0.245 f1_max — a real
but middling number, which is what I expected from a coarse threshold-swept multilabel metric that even a
modest graph embedding can move. But Fold classification came in at 0.184, far the lowest of the three, and
that is the tell. The accuracies alone already say a lot: against a `1/384` chance rate EC is running about
`226×` chance, and against a `1/1195` chance rate Fold is running about `220×` chance, so both carry
genuine signal — this is not an encoder failing to train — yet the *ratio* of the two accuracies is
`0.589 / 0.184 ≈ 3.2`, meaning the same encoder is three times better at the task that leans on
radial-plus-sequence structure than at the task that leans on directional fold discrimination. The loss
column sharpens it further. Uniform guessing over 384 EC classes costs `ln 384 = 5.95` nats and SchNet's EC
test loss is `2.43`, a full `3.5` nats below uniform — the encoder is *confidently* right on EC. Uniform
guessing over 1195 folds costs `ln 1195 = 7.09` nats and SchNet's Fold test loss is `6.60`, only `0.49`
nats below uniform — so even where its top-1 guess lands, its probability mass is diffuse, barely
distinguishable from chance in likelihood terms. That is the radial ceiling made visible twice over: Fold is
the task that most demands fine discrimination among local geometries, and two folds can present the same
multiset of neighbor distances and differ only in *direction* — whether a residue's contacts go off
together or apart — and a distance-only encoder cannot tell them apart no matter how deep the stack, so its
Fold predictions stay smeared. SchNet's whole signal is scalar-radial; the diagnosis is sharp and it is not
a width or a depth problem, it is that the encoder has no channel that carries and *transforms* directional
information through its layers. The fix has to add exactly that channel.

What does "carry directional information" mean precisely, and how do I add it without throwing away the
invariance that bought SchNet its honesty? Let me return to the two facts the rigid-motion constraint gives
me. The relative difference `pos_i - pos_j` is translation-invariant and rotates with an orthogonal `Q` — it
transforms as a type-1 vector, exactly the way I would want a directional output to transform. SchNet
collapsed that difference immediately to its norm, the invariant scalar `d_ij`, and from that moment the
directional, type-1 structure was gone forever — which is *why* it is capped on Fold. So the move is to stop
collapsing. Keep a vector channel alive through the layers and update it equivariantly, while keeping the
feature channel invariant the way SchNet did. I want both regimes on the same graph: invariant features `h`
(rotate the protein, `h` is unchanged) and an equivariant coordinate channel (rotate the protein, the
coordinates rotate with it).

There is more than one way to carry type-1 information, so I lay the options out before committing. The
expensive one is the steerable route the ancestors took — represent everything in higher-order steerable
types and build layers from spherical harmonics and Clebsch–Gordan coefficients so the layer commutes with
rotation by construction. It works, but the harmonics are heavy to recompute per geometry and the apparatus
is welded to three dimensions, and having just learned that pure radial already runs `220×` chance on Fold,
I do not believe I need that much machinery to add the missing directional degree of freedom. The cheap and
wrong one is to bolt the raw direction vector back onto the edge feature, the way the scaffold default did —
but that is precisely the "invariance by hope" I refused at the floor: a direction rotates with `Q`, so the
readout would once again have to learn to average over pose, and nothing forces it to. So I want something
equivariant the way the relative-difference vector itself is equivariant, with no harmonics and no
pose-dependent readout. Stare at the difference vector again. If I take a weighted sum `Σ_j w_ij (pos_i -
pos_j)` and rotate the input, the sum becomes `Σ_j w_ij Q(pos_i - pos_j) = Q Σ_j w_ij (pos_i - pos_j)` —
`Q` factors cleanly out of the sum *provided the weights `w_ij` do not themselves change under rotation*.
That proviso is the entire trick. If each weight is an invariant scalar, then the combination of difference
vectors is equivariant. And I already have an invariant scalar sitting on every edge: the message `m_ij`,
built (SchNet-style) from the invariant features and the invariant distance. So let the coordinate weight be
a scalar function of the message, `φ_x(m_ij)`, and update the coordinate by `pos_i ← pos_i + Σ_j (pos_i -
pos_j) φ_x(m_ij)`. Equivariance lives in the difference vectors, learning lives in the invariant scalar
weight, and they meet only through a product (vector)·(scalar). The weight *must* be a scalar — if `φ_x`
emitted anything with directional structure, that thing would itself have to transform under `Q` and the
clean factoring would break, dragging me back toward the steerable machinery I am avoiding.

So the layer is a message-passing layer with two updates. The message is the SchNet-style invariant edge
function: concatenate the two endpoint features `h_i, h_j` with the invariant distance and pass through an
edge MLP, `m_ij = φ_e(h_i, h_j, ‖pos_i - pos_j‖)`. The feature update aggregates the messages and runs a
node MLP with a residual, `h_i ← h_i + φ_h(h_i, Σ_j m_ij)` — invariant in, invariant out, so the feature
channel stays invariant exactly as in SchNet. The coordinate update is the new equivariant piece. One
detail in the message is worth pausing on, because it looks like a step backward from the floor and is not:
the message MLP reads the *bare* scalar distance `‖pos_i - pos_j‖`, concatenated as a single number onto
the two endpoint features, rather than SchNet's fifty-Gaussian expansion of it. At the floor I argued a
filter net fed a single scalar collapses to one effective degree of freedom at init — a plateau — so why is
a bare scalar acceptable here? Because the geometry is no longer carried *only* by that scalar. Here the
distance is one input among the `1024` feature inputs to a full MLP, not the sole driver of a dedicated
filter net whose entire job is to turn one number into many decorrelated channels, and, more to the point,
the directional information that the floor lacked now arrives through the coordinate channel and re-enters as
*changing* distances layer to layer. So the message MLP does not need the RBF's resolution to be the only
window onto geometry; the plateau argument bit the filter net precisely because that net saw nothing but the
scalar, and that condition no longer holds. I want to
be sure the two channels stay mutually consistent through a whole stack, so I run the induction explicitly.
Suppose `h` is invariant and `pos` is equivariant entering a layer. Then `‖pos_i - pos_j‖` is invariant
(distances survive `Q` and translation), so `m_ij = φ_e(h_i, h_j, d_ij)` is invariant, so `φ_x(m_ij)` is an
invariant scalar, so the coordinate update is `Q`-equivariant and the aggregated messages are invariant, so
the new `h` is again invariant and the new `pos` is again equivariant. The base case holds — `h^{(0)}` is a
linear map of the invariant node features and `pos^{(0)}` is the raw coordinate, which is equivariant by
definition — so by induction a whole stack is invariant on features and equivariant on coordinates. That is
the channel SchNet was missing, and it costs one scalar MLP and a weighted sum of vectors, not a basis of
spherical harmonics.

Two stability details, because the difference vectors can misbehave. First, a bare sum over neighbors grows
with the number of neighbors, so the per-step displacement scale would depend on graph size; with the kNN
graph fixing `k = 16` neighbors, a plain sum of 16 difference vectors would be about `16×` the magnitude of
a single displacement, and a residue in a differently-connected region would move on a different scale. I
aggregate the coordinate displacements with a *mean* rather than a sum, which keeps the displacement `O(1)`
regardless of degree. (The feature messages keep the plain sum aggregation — only the coordinate channel is
mean-aggregated.) Second, the raw difference `(pos_i - pos_j)` can be large for distant pairs, so I
normalize it by its own length before scaling, `(pos_i - pos_j) / (‖pos_i - pos_j‖ + 1)`, so what the
learned weight scales is essentially a bounded direction and the magnitude is governed by the weight, not by
how far apart the residues happen to be. Dividing by a scalar function of the invariant distance does not
touch equivariance — the numerator is `Q` times a difference and the denominator is an invariant scalar, so
the quotient is still `Q` times a direction times an invariant scalar. And the nonlinearities all live on
the invariant channels (inside `φ_e`, `φ_x`, `φ_h`, all consuming and producing invariant quantities); I
never apply a pointwise nonlinearity to the coordinate, and I want to be concrete about why, because it is
the kind of thing that silently breaks. Take a coordinate `x = (1, -1, 0)` and a ninety-degree rotation
about `z` sending `(x,y,z) ↦ (-y,x,z)`, so `Qx = (1, 1, 0)`. Apply ReLU: `Q·ReLU(x) = Q·(1,0,0) = (0,1,0)`,
while `ReLU(Qx) = ReLU(1,1,0) = (1,1,0)`. These are different vectors, so ReLU does not commute with `Q` —
a pointwise nonlinearity on the coordinate would break equivariance outright. That is why the only thing
that ever touches the coordinate is a scalar multiply and a difference; all the bending happens on the
invariant side.

It is worth doing the *positive* check too, so I trust the update rather than only its guardrails. Take a
single edge with `pos_i = (0,0,0)` and `pos_j = (2,0,0)`, so `d_ij = 2` and the normalized difference is
`(pos_i - pos_j)/(d_ij + 1) = (-2,0,0)/3 = (-2/3, 0, 0)`; with an invariant scalar weight `φ_x = 0.5` the
displacement is `(-1/3, 0, 0)`. Now rotate the whole configuration by the same ninety-degree turn about
`z`: `pos_i = (0,0,0)`, `pos_j = (0,2,0)`, the distance is still `2` and the weight `φ_x` is unchanged
because it reads only invariants, so the displacement is `(0,-2,0)/3 · 0.5 = (0,-1/3,0)` — which is exactly
`Q` applied to the original `(-1/3,0,0)`. The displacement rotated with the coordinates, using a weight that
did not move. That is equivariance in a single concrete arithmetic step, and it is the whole reason the
scalar-weighted difference is the right primitive.

The asymmetry between how I aggregate the two channels — sum for the feature messages, mean for the
coordinate displacements — is deliberate and worth stating as a claim about what degree *means* in each
channel. For the features, a residue's degree is signal: a residue packed among many neighbors should
accumulate a larger, richer message than one on a loose loop, because dense packing is itself a fact about
the fold, so summing lets the count of neighbors carry information into `h`. For the coordinate channel,
degree is a nuisance scale: I do not want a residue to lurch further just because the kNN graph happened to
hand it more edges, because the *direction* of the induced motion, not its raw magnitude, is what should
inform the geometry. So mean-aggregation strips the degree factor out of the displacement while sum keeps it
in the messages. I can put rough numbers on the displacement magnitude to check it is neither negligible nor
wild: the normalized difference `(pos_i - pos_j)/(d_ij + 1)` has norm `d_ij/(d_ij + 1)`, which is `0.83` at
a five-Ångström contact and bounded above by `1`, so with an `O(1)` learned weight and the mean over sixteen
neighbors the per-layer displacement is on the order of one Ångström. Over six layers that is a few
Ångströms of accumulated, equivariant coordinate motion — comparable to the very contact scale the distances
live on — so the induced changes in `d_ij` that the next layer's features read are physically meaningful,
not a rounding-error perturbation. The harness omits the `tanh` that would hard-bound the weight, so nothing
caps `φ_x` a priori; the residual anchoring and the mean aggregation are what keep this controlled rather
than letting the coordinates run away early in training.

Now make it concrete in this task's edit surface, and here I have to be careful about what the harness
actually exposes versus the generic equivariant net, because the differences are load-bearing. This task
builds its own **kNN graph** from `pos` (`k = max_neighbors = 16`), so the equivariant layer runs on kNN
edges, not a fully-connected point cloud and not a provided adjacency. The widths are overridden to the
reference encoder configuration — `emb_dim = 512`, six layers, ReLU activation, dropout 0.1, **batch
normalization inside every MLP** (`norm='batch'`), message aggregation `sum`, residual on both the feature
channel and the coordinate channel. Several pieces of the general equivariant machinery are *omitted* by the
harness and I should not import their story: there is no soft attention edge gate, no velocity channel, no
tanh bound on the coordinate weight, and no special tiny initialization called out for the coordinate MLP —
the layer is the plain message-passing port, with the coordinate displacement weighted by `mlp_pos(msg)` on
the `(pos_i - pos_j)/(dist+1)` direction and mean-aggregated. A budget check confirms this is a real
step up in size from the floor: each layer runs about `1.84M` parameters (a message MLP `Linear(1025,512)`
then `Linear(512,512)`, a coordinate MLP `Linear(512,512)` then `Linear(512,1)`, and an update MLP
`Linear(1024,512)` then `Linear(512,512)`, each with its batch-norms), so six layers are about `11M`, close
to four times SchNet's `2.8M`. Tracing the widths through one layer confirms the wiring: the message MLP
reads `concat(h_i, h_j, d_ij)` at `512 + 512 + 1 = 1025` and returns `512`; the summed messages at `512`
feed the coordinate MLP, which squeezes them to the single scalar weight `1`; and the update MLP reads
`concat(h_i, Σ_j m_ij)` at `512 + 512 = 1024` and returns `512`, which the residual adds onto the incoming
`h`. The coordinate residual `pos ← pos + Σ_mean` matters for a subtle reason: keeping the original
coordinate present means the distances a deep layer reads are the raw geometry plus an accumulation of
equivariant corrections, not a coordinate field that has drifted off to somewhere arbitrary, so the geometry
the features consume stays anchored to the real structure while still being enriched. The single biggest
thing to be honest about: although the coordinates are
updated equivariantly layer by layer, the **readout uses only the invariant feature channel** — the final
node embedding is `out_proj(h)` and the graph embedding is the *mean* pool of those node embeddings
(`pool='mean'`, not SchNet's sum). The updated coordinates are computed and propagated but are not
themselves read out as the embedding. So the role of the equivariant coordinate channel here is *not* to
emit a vector at the end; it is to let geometry and features exchange information richly inside the stack —
the coordinate update feeds the next layer's distances, which feed the next messages, which feed the
features that are finally read out. That is the mechanism by which directional structure reaches the
invariant embedding that SchNet's pure-radial layer could never give it: the moving coordinate channel
injects directional, relative-geometry information back into the distances the feature channel sees. The
whole scaffold module is in the answer.

So the delta from SchNet is precise. SchNet collapsed each pair to a single invariant distance and never
carried direction; on Fold that radial blindness capped it at 0.184 with a test loss barely below uniform.
EGNN keeps the invariant feature channel and adds an equivariant coordinate channel — a weighted sum of
relative-difference vectors with invariant scalar weights — so directional, relative-geometry information
lives inside the stack and feeds back into the distances the features read, all while the readout stays
invariant. The cost is one extra scalar MLP per layer and a coordinate update, and roughly four times the
parameters; the gain should be exactly the directional discrimination SchNet lacked.

Here is what I expect against SchNet's numbers, stated so it can be falsified. The clearest prediction is
**Fold**: equivariance should help most where directional fold discrimination matters most, so I expect Fold
to rise substantially above SchNet's 0.184 — if it does not move, the equivariant channel is not actually
injecting usable directional information through the kNN distances and my whole diagnosis is wrong. **EC**
should also rise above 0.589 — the active-site geometry that EC reads benefits from directional structure
too — but by less, because SchNet was already doing well there on radial-plus-sequence signal and is
already `3.5` nats below uniform, so there is less headroom to take. **GO-BP** is the one I am least sure
about: it is a coarse multilabel metric, and the extra coordinate machinery with batch-norm-heavy MLPs could
even *not* help, or marginally regress, if the GO signal is dominated by the sequence-composition node
features rather than fine geometry — batch normalization couples examples within a batch and adds
optimization noise, and piling it on a task whose useful signal is the cheap sequence-correlated one could
drown it. The mechanism I have in mind is specific: `BatchNorm1d` here normalizes each hidden channel over
*all residues in the batch* — every node across the thirty-two proteins pooled together — so a per-protein
compositional offset, exactly the kind of signal GO-BP leans on, is partly subtracted out against the batch
mean, and at test time the running statistics stand in for a distribution of proteins that may not match any
single one. On EC and Fold that normalization is a fair price for stabilizing the deep equivariant stack,
but on a coarse metric whose useful signal is the cheap compositional one, it is plausibly a net loss. So I will watch whether GO-BP holds near or slightly below SchNet's 0.245 rather than assuming it
climbs. And even in the good case, I can already name what this layer structurally cannot do. The
coordinate update applies one scalar weight `φ_x(m_ij)` to each edge
independently and moves `pos_i` along the sum of those single-edge displacements; there is no term anywhere
that reads *two* edges of the same residue jointly. So the angle between two contacts of `i` — the very
directional quantity SchNet was blind to — is still not a first-class object here; the equivariant channel
gives me a richer *distance* field for the features to read, since the moving coordinates change the
next layer's `d_ij`, but the relation *between* a residue's edges is only recovered the same lossy multi-hop
way SchNet had, not represented directly. There is a second thing the layer ignores: it runs one shared
transform over a single kNN graph, and a kNN neighborhood blends two categorically different things — the
residues adjacent along the backbone and the residues far away in sequence but close in space, the tertiary
contacts that folding is *about* — treating a backbone step and a long-range contact with the same weights
as if they carried the same kind of information. If EGNN clears SchNet on Fold and EC but leaves signal on
the table exactly there, the structural error to fix is already exposed: a single shared transform is being
forced onto edges that carry categorically different information, and a residue's contacts are never read
jointly.
