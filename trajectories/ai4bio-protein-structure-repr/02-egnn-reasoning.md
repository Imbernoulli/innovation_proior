The continuous-filter floor told me exactly what its ceiling is, and it told me in three numbers. EC came
in at 0.589 accuracy — respectable, because the enzyme class correlates with broad active-site geometry
and with the amino-acid composition the 28-dim node features already carry, so a radial encoder has
plenty to read there. GO-BP landed at 0.245 f1_max — a real but middling number, which is what I expected
from a coarse threshold-swept multilabel metric that even a modest graph embedding can move. But Fold
classification came in at 0.184, far the lowest of the three, and that is the tell. Fold is the task that
most demands fine discrimination among local geometries, and 0.184 over ~1200 folds is a radial encoder
hitting precisely the blindness I predicted: two folds can present the same multiset of neighbor
distances and differ only in *direction* — whether a residue's contacts go off together or apart — and a
distance-only encoder cannot tell them apart no matter how deep the stack. SchNet's whole signal is
scalar-radial; the Fold number is the radial ceiling made visible. So the diagnosis is sharp and it is
not a width or a depth problem: it is that the encoder has no channel that carries and *transforms*
directional information through its layers. The fix has to add exactly that channel.

What does "carry directional information" mean precisely, and how do I add it without throwing away the
invariance that bought SchNet its honesty? Let me return to the two facts the rigid-motion constraint
gives me. The relative difference `pos_i - pos_j` is translation-invariant and rotates with an orthogonal
`Q` — it transforms as a type-1 vector, exactly the way I would want a directional output to transform.
SchNet collapsed that difference immediately to its norm, the invariant scalar `d_ij`, and from that
moment the directional, type-1 structure was gone forever — which is *why* it is capped on Fold. So the
move is to stop collapsing. Keep a vector channel alive through the layers and update it equivariantly,
while keeping the feature channel invariant the way SchNet did. I want both regimes on the same graph:
invariant features `h` (rotate the protein, `h` is unchanged) and an equivariant coordinate channel
(rotate the protein, the coordinates rotate with it).

The expensive way to carry type-1 information equivariantly is the steerable route the ancestors took —
represent everything in higher-order steerable types and build layers from spherical harmonics and
Clebsch–Gordan coefficients so the layer commutes with rotation by construction. It works, but the
harmonics are heavy to recompute per geometry and the apparatus is welded to three dimensions. I do not
need that much machinery; I need something equivariant the way the relative-difference vector itself is
equivariant, with no harmonics. Stare at the difference vector again. If I take a weighted sum
`Σ_j w_ij (pos_i - pos_j)` and rotate the input, the sum becomes `Σ_j w_ij Q(pos_i - pos_j) = Q Σ_j
w_ij (pos_i - pos_j)` — `Q` factors cleanly out of the sum *provided the weights `w_ij` do not themselves
change under rotation*. That proviso is the entire trick. If each weight is an invariant scalar, then the
combination of difference vectors is equivariant. And I already have an invariant scalar sitting on every
edge: the message `m_ij`, built (SchNet-style) from the invariant features and the invariant distance. So
let the coordinate weight be a scalar function of the message, `φ_x(m_ij)`, and update the coordinate by
`pos_i ← pos_i + Σ_j (pos_i - pos_j) φ_x(m_ij)`. Equivariance lives in the difference vectors, learning
lives in the invariant scalar weight, and they meet only through a product (vector)·(scalar). The weight
*must* be a scalar — if `φ_x` emitted anything with directional structure, that thing would itself have
to transform under `Q` and the clean factoring would break, dragging me back toward the steerable
machinery I am avoiding.

So the layer is a message-passing layer with two updates. The message is the SchNet-style invariant edge
function: concatenate the two endpoint features `h_i, h_j` with the invariant distance and pass through an
edge MLP, `m_ij = φ_e(h_i, h_j, ‖pos_i - pos_j‖)`. The feature update aggregates the messages and runs a
node MLP with a residual, `h_i ← h_i + φ_h(h_i, Σ_j m_ij)` — invariant in, invariant out, so the feature
channel stays invariant exactly as in SchNet. The coordinate update is the new equivariant piece. Both
the invariance of `h` and the equivariance of `pos` are mutually consistent through a stack: if `h` is
invariant entering the layer, the distance is invariant, so `m_ij` is invariant, so `φ_x(m_ij)` is an
invariant scalar, so the coordinate update is equivariant and the new `h` is again invariant — the
induction closes, and a whole stack is invariant on features and equivariant on coordinates. That is the
channel SchNet was missing, and it costs one scalar MLP and a weighted sum of vectors, not a basis of
spherical harmonics.

Two stability details, because the difference vectors can misbehave. First, a bare sum over neighbors
grows with the number of neighbors, so the per-step displacement scale would depend on graph size; I
aggregate the coordinate displacements with a *mean* rather than a sum, which keeps the displacement
O(1) regardless of degree. (The feature messages keep the plain sum aggregation — only the coordinate
channel is mean-aggregated.) Second, the raw difference `(pos_i - pos_j)` can be large for distant pairs,
so I normalize it by its own length before scaling, `(pos_i - pos_j) / (‖pos_i - pos_j‖ + 1)`, so what
the learned weight scales is essentially a bounded direction and the magnitude is governed by the weight,
not by how far apart the residues happen to be. Dividing by a scalar function of the invariant distance
does not touch equivariance — it is still `Q` times a direction times an invariant scalar. And the
nonlinearities all live on the invariant channels (inside `φ_e`, `φ_x`, `φ_h`, all consuming and
producing invariant quantities); I never apply a pointwise nonlinearity to the coordinate, because a
pointwise nonlinearity does not commute with `Q` and would silently break equivariance.

Now make it concrete in this task's edit surface, and here I have to be careful about what the harness
actually exposes versus the generic equivariant net, because the differences are load-bearing. This task
builds its own **kNN graph** from `pos` (`k = max_neighbors = 16`), so the equivariant layer runs on kNN
edges, not a fully-connected point cloud and not a provided adjacency. The widths are overridden to the
reference encoder configuration — `emb_dim = 512`, six layers, ReLU activation, dropout 0.1, **batch
normalization inside every MLP** (`norm='batch'`), message aggregation `sum`, residual on both the
feature channel and the coordinate channel. Several pieces of the general equivariant machinery are
*omitted* by the harness and I should not import their story: there is no soft attention edge gate, no
velocity channel, no tanh bound on the coordinate weight, and no special tiny initialization called out
for the coordinate MLP — the layer is the plain message-passing port, with the coordinate displacement
weighted by `mlp_pos(msg)` on the `(pos_i - pos_j)/(dist+1)` direction and mean-aggregated. The single
biggest thing to be honest about: although the coordinates are updated equivariantly layer by layer, the
**readout uses only the invariant feature channel** — the final node embedding is `out_proj(h)` and the
graph embedding is the *mean* pool of those node embeddings (`pool='mean'`, not SchNet's sum). The updated
coordinates are computed and propagated but are not themselves read out as the embedding. So the role of
the equivariant coordinate channel here is *not* to emit a vector at the end; it is to let geometry and
features exchange information richly inside the stack — the coordinate update feeds the next layer's
distances, which feed the next messages, which feed the features that are finally read out. That is the
mechanism by which directional structure reaches the invariant embedding that SchNet's pure-radial layer
could never give it: the moving coordinate channel injects directional, relative-geometry information back
into the distances the feature channel sees. The whole scaffold module is in the answer.

So the delta from SchNet is precise. SchNet collapsed each pair to a single invariant distance and never
carried direction; on Fold that radial blindness capped it at 0.184. EGNN keeps the invariant feature
channel and adds an equivariant coordinate channel — a weighted sum of relative-difference vectors with
invariant scalar weights — so directional, relative-geometry information lives inside the stack and feeds
back into the distances the features read, all while the readout stays invariant. The cost is one extra
scalar MLP per layer and a coordinate update; the gain should be exactly the directional discrimination
SchNet lacked.

Here is what I expect against SchNet's numbers, stated so it can be falsified. The clearest prediction is
**Fold**: equivariance should help most where directional fold discrimination matters most, so I expect
Fold to rise substantially above SchNet's 0.184 — if it does not move, the equivariant channel is not
actually injecting usable directional information through the kNN distances and my whole diagnosis is
wrong. **EC** should also rise above 0.589 — the active-site geometry that EC reads benefits from
directional structure too — but by less, because SchNet was already doing well there on radial-plus-
sequence signal. **GO-BP** is the one I am least sure about: it is a coarse multilabel metric, and the
extra coordinate machinery with batch-norm-heavy MLPs could even *not* help, or marginally regress, if
the GO signal is dominated by the sequence-composition node features rather than fine geometry — so I will
watch whether GO-BP holds near or slightly below SchNet's 0.245 rather than assuming it climbs. If EGNN
clears SchNet on Fold and EC but the equivariant coordinate channel still leaves something on the table on
the relational structure of contacts — because a single scalar weight per edge cannot model how *two*
edges of the same residue relate — then the next rung is already implied: type the edges by what they
mean and let the relational structure, not just per-edge geometry, into the encoder.
