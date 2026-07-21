The D-MPNN numbers are the breakthrough I was watching for on BBBP — and they are disappointing
everywhere else in a way that names the next move exactly. Line the two feedback tables up and take the
differences, because the differences are the whole argument. BBBP moved off chance: GIN's **0.510** to
D-MPNN's **0.599**, a `+0.089` jump — so the diagnosis was right in direction, the global RDKit descriptor
prior pulled the single-target task whose answer is global physicochemistry up off the coin flip. But look
at the other two rows: BACE *fell* `0.726 → 0.688`, a `−0.038`, and Tox21 *fell* `0.747 → 0.713`, a
`−0.034`. Average the three deltas and the mean ROC-AUC barely moved — `0.661 → 0.667`, `+0.006` — which
is the tell that matters: I did not raise the ceiling, I *redistributed* error, buying `+0.089` on BBBP by
paying it back almost exactly on the two tasks the 2D graph was already handling. Rearranging the message
passing — atom-centered to bond-centered, mean readout to sum — bought nothing net on the tasks where the
2D graph was the whole story, and slightly hurt them. Two quite different 2D architectures now bracket what
2D can do: BACE lives in `[0.688, 0.726]` and Tox21 in `[0.713, 0.747]` across the atom-centered and
bond-centered encoders, a spread of only ~`0.04`, and *neither* arrangement broke above GIN's marks. When
two structurally different ways of passing messages along the same bonds land in the same narrow band and
neither clears the first rung, the ceiling is not in the *architecture* — it is in the *input*. The
information the 2D graph throws away is the *geometry*, and geometry is not a thing I can recover by
tweaking aggregation, because it was never in the input to begin with.

And read BBBP's `0.599` a second time, because it also carries information about how far I actually got.
Fixed-descriptor classifiers — a random forest over exactly the kind of RDKit panel I concatenated — land
in the high-`0.60`s to low-`0.70`s on BBBP's scaffold split; that is roughly the ceiling the descriptor
axis alone can buy. I landed at `0.599`, well short of that, which reads as the descriptor branch arriving
only *partially* — consistent with the exposure risk I flagged at rung two, that the branch only
contributes where `batch._smiles` is present and otherwise degenerates to a head-absorbed constant. So even
the one clear win is a fraction of what its own prior could deliver, and pushing on more 2D descriptor
plumbing would, at best, recover the missing part of a lift that is itself capped in the low-`0.70`s. The
lever is elsewhere.

So step back to what the property actually depends on. Whether a molecule crosses the blood-brain barrier,
inhibits an enzyme, is toxic in an assay — every one of these is decided by how the atoms sit in *space*:
which groups are close enough to touch, the shape that fits or doesn't fit a binding site. A 2D bond graph
has thrown that away before the encoder ever sees it. Two atoms can be many bonds apart yet sit right next
to each other in folded 3D space and interact strongly — a hydrogen bond bridging across a ring, two
aromatic groups stacking face-to-face — and a message-passing GNN that only talks along bonds *structurally
cannot* see that contact. Put a number on it: a donor and an acceptor on opposite sides of a six-membered
ring are six or more bonds apart on the graph but can sit `~2.8 Å` apart in space, close enough for a real
hydrogen bond; a three-hop message passer never connects them, and no amount of rearranging the three hops
will, because the edge simply is not in the graph. That is the real ceiling under BACE and Tox21. And there
is a second wall I keep hitting: data. These sets are a few hundred to a few thousand molecules; an encoder
learning its whole representation from scratch overfits the artifacts of a tiny training set, exactly the
`400:1` and `220:1` parameter-to-label ratios the 2D rungs were already straining against. The fix for
*that* is the recipe that made BERT and ViT work — pretrain a representation on an essentially unlimited
pile of unlabeled molecules, then finetune a light head. The scaffold even hands me a pretrained checkpoint
inside the container. So the next rung is not more 2D graph machinery; it is an encoder whose input *is* the
3D structure, pretrained at scale, finetuned here. Both walls — geometry and data-hunger — fall to the same
move.

There is a real fork in how to consume 3D, though, and I want to walk it rather than jump. One option is a
3D *graph* net — SchNet/DimeNet style, message passing over spatial neighbors within a cutoff. But locality
is exactly the property that just failed me: a spatial-cutoff GNN reintroduces a receptive-field wall, only
now measured in Ångströms instead of bonds, and the contacts that decide the property are precisely the
long-range ones. A second option is a fully *equivariant* tensor network — SE(3)-Transformers, tensor-field
networks — that carries geometric tensors transforming correctly under rotation. That is the principled
heavy machinery, but it is the wrong *price* here for two reasons I can make precise. First, I only need the
model's *output* — a molecule-level property — to be invariant to rotating and translating the molecule; I
do *not* need the intermediate features to be equivariant. Equivariance is a strictly stronger, more
expensive property than the invariance the task actually demands, so paying for it is over-engineering the
symmetry. Second, I am not training this from scratch — I am consuming a checkpoint pretrained at enormous
scale — and the checkpoint is for an *invariant* Transformer, so equivariant tensor machinery is not even
the object I can load. The third option is a plain Transformer over atoms with geometry injected as an
invariant, and once I see that invariance is all the target needs, that is where the walk lands.

So the backbone is a Transformer, not a local 3D GNN: it treats its input as a fully-connected set, every
atom attending to every other, so the long-range contacts that decide the property are visible — flipping
the bonded-only message passing that capped the 2D encoders into full connectivity. A set is the right
object anyway: atoms have no canonical order, attention is permutation-equivariant over them, and a `[CLS]`
readout gives an order-invariant molecule vector.

But a Transformer is position-blind: `softmax(QKᵀ/√d)V` depends only on the set of token features, not on
where any atom is. In language you add a positional encoding indexed by a *discrete* integer slot. My
position is a *continuous* 3D coordinate, and one I cannot use raw: rotate or translate the whole molecule
— which changes nothing about its chemistry — and every absolute coordinate changes, so the model would
waste capacity learning that a molecule and its rotated copy are the same. So I need to inject 3D position
in a way that is continuous *and* invariant to global rotation and translation (SE(3)). The clean way to
put structure into a Transformer without breaking full connectivity is the Graphormer move: don't touch the
architecture, add a learnable per-pair bias to the attention logits, `A_ij = QᵢKⱼᵀ/√d + b_{φ(i,j)}`. The
slot is right — geometry enters as a number added to `QKᵀ` before the softmax — but Graphormer's bias is
keyed by shortest-path distance and discrete bond features, all 2D topology, with nowhere to put a
continuous Euclidean distance and nothing rotation-invariant in it. So the question sharpens: what
continuous, SE(3)-invariant per-pair quantity goes in that bias slot?

Stare at the coordinates and do the algebra. A rigid motion sends every atom `xᵢ → R xᵢ + t` with `R`
orthogonal (`RᵀR = I`) and `t` a translation. An *absolute* coordinate obviously moves. But the pairwise
difference loses the translation, `(R xᵢ + t) − (R xⱼ + t) = R(xᵢ − xⱼ)`, and its length loses the
rotation too: `‖R(xᵢ − xⱼ)‖² = (xᵢ − xⱼ)ᵀ RᵀR (xᵢ − xⱼ) = (xᵢ − xⱼ)ᵀ(xᵢ − xⱼ) = ‖xᵢ − xⱼ‖²`. So the
Euclidean distance `d_ij = ‖xᵢ − xⱼ‖` is exactly invariant under every rotation and translation, checked
in three lines, while any absolute coordinate is not. Build the per-pair bias purely out of pairwise
distances and I get SE(3)-invariance *for free*, with none of the heavy equivariant-tensor machinery I just
ruled out — which is the concrete payoff of choosing to be invariant to the coordinate rather than
equivariant to it: the cost stays close to a vanilla Transformer.

What function of `d_ij`? One scalar per pair, fed through layers and eventually backprop targets, so I want
a smooth, expressive featurization — evaluate the distance against a bank of `K` learnable Gaussians with
means `μᵏ` and widths `σᵏ`: the vector `(G(d_ij,μ¹,σ¹),…,G(d_ij,μᴷ,σᴷ))` is a soft one-hot of where on the
distance axis the pair lives. Gaussians over bins because the encoding must be differentiable in `d` (so
coordinates can themselves be a target, which matters for the pretraining that produced my checkpoint) and
free of bin-boundary artifacts. And a bare distance kernel says a C–C pair at `1.5 Å` and a C–O pair at
`1.5 Å` are encoded *identically* — clearly wrong, the chemistry differs. So make the kernel
pair-type-aware: apply a per-pair-type affine `a_t·d + b_t` to the distance *before* the Gaussian basis,
with `a_t, b_t` learnable scalars indexed by the unordered pair of element types. The same number of
Ångströms then lands at a different position on the shared Gaussian ruler for different pair types, so the
model reads `1.5 Å` between two carbons as a different geometric event than `1.5 Å` between a carbon and an
oxygen. Now trace the shapes so I know it drops into the attention slot cleanly. Distances arrive as
`[B, N, N]`; the pair type is `edge_type = token_i · dict_size + token_j`, and with a dictionary of `31`
tokens that is `31·31 = 961` distinct pair types each carrying its own `(a_t, b_t)`. The affine-then-Gaussian
step lifts `[B, N, N] → [B, N, N, K]`; a small MLP `Linear(K → heads)` projects the `K` Gaussian channels
down to one bias per attention head, giving `[B, N, N, heads]`, which permutes to `[B, heads, N, N]` and
adds straight onto the attention logits, themselves `[B, heads, N, N]`. Dimensions line up, and geometry is
now steering which atoms each atom attends to — the pair-to-atom channel. `961` pair types is generous:
an organic molecule draws from a handful of elements, so `C–C`, `C–N`, `C–O`, `C–H`, `C–S`, `N–O` and a
few more cover the overwhelming majority, and the rare affines see almost no gradient here. That is fine
because these `(a_t, b_t)` come pretrained — exercised across hundreds of millions of conformers — so I
inherit calibrated per-pair rulers rather than trying to fit `961` of them on a few hundred labels, which
the finetune set never could.

The readout has its own symmetry bookkeeping, and it is easy to reintroduce an order dependence that
quietly breaks the invariance I paid for. Attention is permutation-*equivariant* over the atom tokens:
permute the input atoms by any `π` and every output token permutes by the same `π`, since `softmax(QKᵀ)V`
carries no positional index and the distance bias is indexed symmetrically by the pair. So the per-atom
outputs are equivariant, but I need the molecule vector *invariant* — order-free. The `[CLS]` token
supplies that: a fixed extra position that attends to all atoms and is attended-to, so its output is a
permutation-invariant pooling of the atom set, one vector per molecule independent of atom numbering.
Equivariant trunk, invariant readout — the composition is invariant end to end, which with the
SE(3)-invariant distance bias makes the whole model invariant to both atom relabeling and rigid motion.

The pair feature should not stay frozen, though. The distance is fixed by the conformer, but the *meaning*
of a pair's interaction is what the model figures out as it processes the molecule — and that figuring-out
lives in the atom representations while the pair channel sits at its initial geometric value. There is a
per-pair, per-layer quantity that already captures "how strongly do these two atoms interact right now":
the query-key product `QᵢKⱼᵀ`. So let the atoms write back into the pairs — accumulate the current per-head
`QᵢKⱼᵀ` into the pair representation each layer, residually. Now the two channels genuinely talk: the pair
feature biases attention (pair→atom) and the attention's affinity refines the pair feature (atom→pair),
layer after layer, and the residual accumulation makes `q^L − q^0` the *learned correction* to the raw
geometry. The cost stays small because the `QK` product is already computed for the attention itself.

Optimization stability is not a detail at this scale — the pretraining ran a very long schedule over
hundreds of millions of conformers, and I am finetuning a 15-layer, 512-dim, 64-head, ~86M-parameter
encoder — so the block has to be the stable variant. Post-LN normalizes *after* the residual add, which
leaves the identity path passing through a LayerNorm and produces large gradients near the output at init,
forcing a fragile warmup to keep early training from diverging; Pre-LN puts the norm *inside* the residual
branch, so the identity path is clean and the block is well-behaved at initialization and through a long
schedule. So each block is: normalize, self-attend with the pair bias, add; normalize, GELU feed-forward,
add. The encoder tracks and updates the pair representation through the stack, so the attention bias
entering each layer is the running pair feature, and the delta `q^L − q^0` falls out by subtraction.

The scale is why loading the checkpoint is forced, not a nicety. `86M` parameters against BBBP's `~1.6k`
training molecules is ~`52,000` per labeled example — two orders past the `~400:1` GIN and `~220:1`
D-MPNN ratios that were already over-parameterized. This trunk cannot learn a transferable representation
from a few hundred scaffolds; the pretrained weights, learned on an essentially unlimited pile of unlabeled
conformers, are the entire mechanism by which it escapes the data-hunger that capped the 2D encoders.

Now make it concrete in the editable region, since the fixed pipeline exposes only the
`MoleculeModel` class, narrower than the full framework. I do *not* build the
pretraining loop, the masked-atom / noised-coordinate recovery objective, or the equivariant coordinate
head here — those produced the checkpoint I am loading, but for property prediction I only consume the
encoder and read out a molecule vector. So the fill is: a token embedding sized to the pretrained dictionary
(map each atom's element to its dictionary index from the 136-dim feature vector, prepend `[CLS]`, append
`[SEP]`, pad with `[PAD]`); extend the distance matrix and mask for those two special tokens; form edge
types as `token_i · dict_size + token_j`; run the edge-type Gaussian layer and project to per-head attention
bias; embed tokens; run the Pre-LN encoder-with-pair; and read out the `[CLS]` token through a
classification head. One simpler choice here: the head is a plain dropout-then-linear on the
`[CLS]` representation — not a dense+tanh+dropout pooler — because at finetune the heavy pretrained trunk
does the work and the simplest head transfers most cleanly and adds the fewest fresh, un-pretrained weights
to overfit on a few hundred labels. And the load step matters: I read the checkpoint, remap the
fairseq/unicore key names to this flat module layout, copy every key whose shape matches (`embed_tokens`,
the Gaussian layer, all 15 encoder layers), and skip the pretraining-only heads (LM head, distance head,
coordinate projection) and any mismatch — falling back to from-scratch only if the checkpoint is absent, a
path the budget arithmetic above says I never want to take.

So the delta from D-MPNN is total, not incremental: instead of passing messages along bonds in 2D with a
bolted-on descriptor prior, I feed the model atoms-in-space, attend fully-connected with an SE(3)-invariant
pair-type Gaussian distance bias, let geometry and attention co-refine through a Pre-LN encoder, read out
`[CLS]`, and start from weights pretrained on hundreds of millions of conformers.

The cleanest falsifiable test is BACE and Tox21: both *fell* from GIN to D-MPNN (`0.726→0.688`,
`0.747→0.713`) because 2D message passing had hit its ceiling, so if geometry plus pretraining is the real
lever they should now *exceed* even GIN's marks — not merely recover D-MPNN's. BBBP is the harder claim:
the descriptor prior got it to `0.599`, and 3D geometry plus a pretrained trunk should push it higher
still, comfortably off chance, though its severe single-target scaffold shift means I expect the least
absolute headroom there. If instead BACE and Tox21 fail to clear GIN's marks, my story — that the 2D
*input*, not the 2D *architecture*, was the ceiling — is wrong, and the bracket I read off the two feedback
tables was coincidence rather than a wall.
