The D-MPNN numbers are the breakthrough I was watching for on BBBP — and they are disappointing
everywhere else in a way that names the next move exactly. BBBP moved off chance: from GIN's **0.510** to
**0.599**. So the diagnosis was right in direction — the global RDKit descriptor prior pulled the single-
target task whose answer is global physicochemistry up off the coin flip. But 0.599 is a weak win, not
the decisive one I expected if the descriptor branch were fully delivering its lipophilicity/polarity/
size signal, which is exactly the risk I flagged: the descriptor branch only contributes where
`batch._smiles` is exposed, and a BBBP that climbs to only ~0.60 rather than ~0.70 reads as the global
prior arriving partially, with the directed-bond encoder carrying most of the load. Worse, read the other
two rows: BACE *fell* from 0.726 to **0.688** and Tox21 *fell* from 0.747 to **0.713**. The directed-bond
fix and the sum readout did not buy anything on the tasks where the 2D graph was already doing fine —
they slightly hurt. That is the tell. I have now spent two rungs entirely inside the 2D bond graph, and
the 2D bond graph has a ceiling: BACE and Tox21 do not move up no matter how I rearrange message passing,
and BBBP creeps but does not leap. The information the 2D graph throws away is the *geometry* — and that
is not a thing I can recover by tweaking aggregation, because it was never in the input.

So step back to what the property actually depends on. Whether a molecule crosses the blood-brain
barrier, inhibits an enzyme, is toxic in an assay — every one of these is decided by how the atoms sit in
*space*: which groups are close enough to touch, the shape that fits or doesn't fit a binding site. A 2D
bond graph has thrown that away before the encoder ever sees it. Two atoms can be many bonds apart yet
sit right next to each other in folded 3D space and interact strongly — a hydrogen bond across a ring,
two aromatic groups stacking — and a message-passing GNN that only talks along bonds *structurally
cannot* see that contact. That is the real ceiling under BACE and Tox21. And there is a second wall I
keep hitting: data. These sets are a few hundred to a few thousand molecules; an encoder learning its
whole representation from scratch overfits the artifacts of a tiny training set. The fix for *that* is
the recipe that made BERT and ViT work — pretrain a representation on an essentially unlimited pile of
unlabeled molecules, then finetune a light head. The scaffold even hands me a pretrained checkpoint
inside the container. So the next rung is not more 2D graph machinery; it is an encoder whose input *is*
the 3D structure, pretrained at scale, finetuned here. Both walls — geometry and data-hunger — fall to
the same move.

The first instinct is a 3D graph net, but locality is exactly wrong: I want *every* atom able to look at
*every* other atom, because the interactions that decide the property are long-range in bond-distance.
That is what self-attention gives for free — a Transformer treats its input as a fully-connected set,
every atom token attending to every other. And a set is the right object: atoms have no canonical order,
attention is permutation-equivariant over them, and a `[CLS]` readout gives an order-invariant molecule
vector. So the backbone is a Transformer, not a local GNN — which also flips the very thing that capped
the 2D rungs (bonded-only message passing) into full connectivity.

But a Transformer is position-blind: `softmax(QKᵀ/√d)V` depends only on the set of token features, not on
where any atom is. In language you add a positional encoding indexed by a *discrete* integer slot. My
position is a *continuous* 3D coordinate, and one I cannot use raw: rotate or translate the whole
molecule — which changes nothing about its chemistry — and every absolute coordinate changes, so the
model would waste capacity learning that a molecule and its rotated copy are the same. So I need to inject
3D position in a way that is continuous *and* invariant to global rotation and translation (SE(3)). The
clean way to inject structure into a Transformer without breaking full connectivity is the Graphormer
move: don't touch the architecture, add a learnable per-pair bias to the attention logits,
`A_ij = QᵢKⱼᵀ/√d + b_{φ(i,j)}`. The slot is right — geometry enters as a number added to `QKᵀ` before the
softmax — but Graphormer's bias is keyed by shortest-path distance and discrete bond features, all 2D
topology, with nowhere to put a continuous Euclidean distance and nothing rotation-invariant in it. So
the question sharpens: what continuous, SE(3)-invariant per-pair quantity goes in that bias slot?

Stare at the coordinates. Anything involving an *absolute* coordinate moves under a rigid motion. What
survives any rotation and translation of the whole molecule is the *relative* geometry, and the simplest
invariant of a pair of points is the Euclidean distance `d_ij = ‖xᵢ − xⱼ‖`. Build the per-pair bias
purely out of pairwise distances and I get SE(3)-invariance *for free*, with none of the heavy
equivariant-tensor machinery (tensor field networks, SE(3)-Transformers) that would be the wrong price at
pretraining scale. By choosing the *invariant* of the coordinate rather than trying to be equivariant to
it, I keep the cost close to a vanilla Transformer.

What function of `d_ij`? One scalar per pair, fed through layers and eventually backprop targets, so I
want a smooth, expressive featurization — evaluate the distance against a bank of `K` learnable Gaussians
with means `μᵏ` and widths `σᵏ`: the vector `(G(d_ij,μ¹,σ¹),…,G(d_ij,μᴷ,σᴷ))` is a soft one-hot of where
on the distance axis the pair lives. Gaussians over bins because the encoding must be differentiable in
`d` (so coordinates can be a target, which matters for pretraining) and free of bin-boundary artifacts.
And a bare distance kernel says a C–C pair at 1.5 Å and a C–O pair at 1.5 Å are encoded *identically* —
clearly wrong, the chemistry differs. So make the kernel pair-type-aware: apply a per-pair-type affine
`a_t·d + b_t` to the distance *before* the Gaussian basis, with `a_t, b_t` learnable scalars indexed by
the unordered pair of element types. The same number of Ångströms then lands at a different position on
the shared ruler for different pair types. (Pair type from the atom identities, not the bond — I am
living without the 2D bond.) The `K` Gaussian channels project down to one bias per head with a small
MLP, and that bias is added into the attention logits exactly in Graphormer's slot — continuous and
SE(3)-invariant now, not a discrete shortest-path scalar. This is the pair-to-atom communication:
geometry steers which atoms each atom attends to.

The pair feature should not stay frozen, though. The distance is fixed by the conformer, but the
*meaning* of a pair's interaction is what the model figures out as it processes the molecule — and that
figuring-out lives in the atom representations while the pair channel sits at its initial geometric value.
There is a per-pair, per-layer quantity that already captures "how strongly do these two atoms interact
right now": the query-key product `QᵢKⱼᵀ`. So let the atoms write back into the pairs — add the current
per-head `QᵢKⱼᵀ` into the pair representation each layer, residually. Now the two channels genuinely talk:
the pair feature biases attention (pair→atom) and the attention's affinity refines the pair feature
(atom→pair), layer after layer, and the residual accumulation makes `q^L − q^0` the *learned correction*
to the raw geometry. The cost stays small because the QK product is already computed.

Optimization stability is not a detail at this scale — the pretraining ran for a very long schedule over
hundreds of millions of conformers, and I am finetuning a 15-layer, 512-dim, 64-head, ~86M-parameter
encoder, so the block has to be the stable variant. Post-LN gives large gradients near the output at init
and forces fragile warmup; Pre-LN puts the norm inside the residual branch, well-behaved at init. So each
block is: normalize, self-attend with the pair bias, add; normalize, GELU feed-forward, add. The encoder
tracks and updates the pair representation through the stack, so the attention bias entering each layer is
the running pair feature, and the delta `q^L − q^0` falls out by subtraction.

Now make it concrete *in this task's edit surface*, because the harness fixes everything but the
`MoleculeModel` class, and what it exposes is narrower than the full framework. I do *not* build the
pretraining loop, the masked-atom / noised-coordinate recovery objective, or the equivariant coordinate
head here — those produced the checkpoint I am loading, but for property prediction I only consume the
encoder and read out a molecule vector. So the fill is: a token embedding sized to the Uni-Mol dictionary
(map each atom's element to its dictionary index from the 136-dim feature vector, prepend `[CLS]`, append
`[SEP]`, pad with `[PAD]`); extend the distance matrix and mask for those two special tokens; form edge
types as `token_i · dict_size + token_j`; run the edge-type Gaussian layer and project to per-head
attention bias; embed tokens; run the Pre-LN encoder-with-pair; and read out the `[CLS]` token through a
classification head. One faithful-but-thinner choice here: the head is a simple dropout-then-linear on the
`[CLS]` representation — not the dense+tanh+dropout pooler — because at finetune the heavy pretrained
trunk does the work and the simplest head transfers most cleanly. And the load step matters: I read the
checkpoint, remap the fairseq/unicore key names to this flat module layout, copy every key whose shape
matches (`embed_tokens`, the Gaussian layer, all 15 encoder layers), and skip the pretraining-only heads
(LM head, distance head, coordinate projection) and any mismatch — falling back to from-scratch only if
the checkpoint is absent. The pretrained weights are the entire point: they are how the encoder escapes
the data-hunger that capped the 2D rungs.

So the delta from rung two is total, not incremental: where D-MPNN passed messages along bonds in 2D and
bolted on a global descriptor prior, I feed the model atoms-in-space, attend fully-connected with an
SE(3)-invariant pair-type Gaussian distance bias, let geometry and attention co-refine through a Pre-LN
encoder, read out `[CLS]`, and — crucially — start from weights pretrained on hundreds of millions of
conformers. Reading the 2D rungs' shape, here are the falsifiable claims. BACE and Tox21 are the cleanest
tests: both *fell* from GIN to D-MPNN (0.726→0.688, 0.747→0.713) because 2D message passing had hit its
ceiling, so if geometry plus pretraining is the real lever, both should now *exceed* even GIN's numbers —
BACE clearly above 0.726 and Tox21 above 0.747 — not merely recover D-MPNN's. BBBP is the harder claim:
the descriptor prior got it to 0.599, and 3D geometry plus a pretrained trunk should push it higher
still, comfortably off chance and above D-MPNN — though BBBP's severe scaffold shift on a single binary
target means I expect the *smallest* absolute headroom there, so a BBBP around the low-0.70s with BACE in
the mid-0.80s and Tox21 in the mid-0.70s would confirm the diagnosis: the win comes from the geometry the
2D graph never had and the pretraining the small sets could never substitute for, and it is largest
exactly where local-2D message passing was most starved. If instead BACE and Tox21 fail to clear GIN's
marks, my story — that the 2D *input*, not the 2D *architecture*, was the ceiling — is wrong.

The causal chain in one breath: D-MPNN's measured failure is a *2D-input* ceiling — BBBP crept off chance
(0.510→0.599) on the descriptor prior but BACE and Tox21 *fell* (0.726→0.688, 0.747→0.713), so
rearranging 2D message passing buys nothing where geometry is the missing signal → the property depends on
how atoms sit in space and the small sets are data-starved, so move to a 3D-input encoder pretrained at
scale → use a fully-connected Transformer (every atom sees every atom, long-range contacts visible),
inject 3D position as a per-pair attention bias built from the SE(3)-invariant Euclidean distance through
a pair-type-aware learnable Gaussian basis (Graphormer's slot, geometric contents) → let `QKᵀ` write back
into the pair channel so geometry and attention co-refine, Pre-LN for stable finetuning, `[CLS]` readout,
and load the pretrained checkpoint → expecting BACE and Tox21 to exceed even GIN's marks and BBBP to push
past D-MPNN, with the largest gains where 2D message passing was most starved of geometry.
