The EGNN numbers landed exactly on the split I predicted, and the split is the whole instruction for
this rung. Let me put the two rows side by side and actually difference them, because the pattern is
too clean to read casually. Against SchNet, EGNN lifted 2019 the most: RMSE 1.5624 → 1.4422, a drop of
0.1202 (about 7.7%), and Rp 0.5699 → 0.6175, a rise of 0.0476 that clears the 0.60 bar I set. It
improved 2013 across the board too: RMSE 1.4765 → 1.4114 (down 0.0651) and Rp 0.7921 → 0.8093 (up
0.0172). So on the temporal holdout and the small core set — the two places where *reasoning* beats
*recall* — the node-pair-aware message bought real generalization, and the biggest single move on the
whole table is the 2019 RMSE gain. But on 2016 it did exactly what I feared: RMSE went *up*, 1.3702 →
1.4213 (worse by 0.0511, about 3.7%), and Rp *down*, 0.7767 → 0.7646 (worse by 0.0121). Count the
cells: EGNN won four of the six metric slots and lost two, and the two it lost are both on 2016, the
near-training core set. That is not a wash and it is not noise; it is the precise trade I named. EGNN
fed the distance raw as one channel — the rank-one-at-init `mlp_e` — where SchNet expanded it into 60
decorrelated RBF channels, so on the familiar CASF-2016 motifs, where fine distance resolution pays
off, EGNN's coarser geometry cost it even as its node-side richness won everywhere else. One more
number is worth extracting, because it is the strongest evidence that the node-pair move bought
*generalization* rather than fit: SchNet's RMSE gap between the near-training 2016 set and the
temporally distant 2019 set was 1.5624 − 1.3702 = 0.192, a wide train-test chasm, whereas EGNN's is
1.4422 − 1.4213 = 0.021 — almost gone. EGNN gave up a little on 2016 and pulled 2019 up so far that the
two sets nearly meet, which is exactly what "reasoning about unseen chemistry instead of recalling seen
chemistry" should do to a generalization gap. I want to keep that property, not trade it away, as I add
geometry back.

So I now have two diagnoses stacked, and they point at the same culprit from opposite sides: SchNet was
geometry-resolved but node-poor and orientation-blind; EGNN is node-rich but geometry-thin. *Both* are
still feeding their messages essentially one scalar — the contact distance — while the batch carries an
11-dim geometric description of every edge (angle max/sum/mean, triangle-area max/sum/mean,
neighbour-distance max/sum/mean, and pairwise L1 and L2) and a 17-dim description of every covalent edge
that adds bond type, conjugation, and ring membership. Between them, the first two rungs threw almost
all of that away. And the way they failed is complementary in a way that is itself informative: SchNet's
geometry resolution helped 2016 and its node-poverty hurt 2019; EGNN's node richness helped 2019 and
its geometry-thinning hurt 2016. Neither axis alone dominates, which means the right move is not to pick
a side but to hold *both* — recover the resolution SchNet had while keeping the node-pair expressiveness
EGNN bought. The instruction is unambiguous: stop discarding the full edge geometry and let it into the
message.

Before I settle on how, let me lay out the ways I could obey that instruction, because they are not
equivalent. The most literal is to fuse the two prior rungs directly: keep EGNN's additive endpoint
MLPs but feed the geometry side the full 11-dim edge vector (or its RBF expansion) instead of the bare
distance — one homogeneous message mechanism carrying both node richness and geometry resolution. That
is tempting and cheap, but it leaves two things unaddressed that I now think matter. First, it keeps a
*single* message mechanism for physically distinct edge types, and the covalent-versus-non-covalent
distinction is exactly the structure the batch's separate 17-dim and 11-dim feature sets are inviting
me to exploit — a stiff low-degree bond and a soft variable-degree contact want different aggregation,
not one shared rule. Second, it keeps the pool-and-regress readout form untouched, and I am about to
argue that the readout *form*, not just its inputs, is a second axis worth moving. A third option is
more ambitious: pass messages over the *angles* between edges, the way an angle-aware geometric network
does, by building a line graph on the edges and conditioning each message on the angle it makes with
its neighbours. That would capture orientation directly rather than through node identity — genuinely
the thing SchNet was blind to. But I have to check whether the edit surface even permits it, and it
does not: the harness hands me precomputed invariant *edge features*, one vector per edge, but not the
edge-adjacency structure — which edges share an atom, and at what angle — that a line-graph angle
message requires. The angle *statistics* are baked into the 11-dim vector as summary numbers
(angle max/sum/mean), but the triangle-level adjacency I would need to pass a message from one edge to
another is not exposed, and I cannot reconstruct it without coordinates. So the angle-message route is
unimplementable here, and the angle information is available to me only in the aggregated form already
sitting in `edge_attr`. That settles it toward the second option — distinct covalent and non-covalent
mechanisms, each consuming the full edge feature — and pushes me to earn the rest of the gain from the
readout form rather than from an angle-adjacency I do not have.

But I want to push past "use more features," because there is a second, sharper diagnosis hiding in how
all three fills so far read out the affinity, and I nearly missed it because all three share the same
readout. They all pool an interface representation and regress it — a triple-product-and-bias head, yes,
but still a learned scalar function of pooled statistics. Let me reconsider what binding affinity *is*
as an output, not just what geometry feeds it. The free energy of binding is, to first order, a *sum* of
pairwise contributions — each non-covalent contact between a ligand atom and a pocket atom contributes
some favourable or unfavourable amount, and the total binding free energy is their sum. That is a strong
claim about the *form* of the prediction, not just its inputs, and none of my fills enforce it: they let
the affinity be any learned function of the interface embedding, including functions that could not
possibly be a sum of local contact terms. If I instead make the output *literally* a sum over interface
contacts of a per-contact atom-atom affinity, I restrict the hypothesis class to functions that already
look like the physics. Restricting the hypothesis class in a direction the truth actually lies is the
cleanest kind of inductive bias — it shrinks the space the model can overfit into, which should help
exactly on the held-out split where overfitting is punished. I should be careful not to oversell the
physics, though: the sum-of-pairwise picture is only a *first-order* truth. Real binding free energy
carries genuinely non-additive terms — cooperativity between nearby contacts, desolvation of the buried
interface, conformational entropy — that a strict sum of independent per-contact scalars cannot
represent. The escape hatch is that my per-contact score is not a fixed physical potential but a
learned function of the *post-message-passing* atom features, and after three rounds of message passing
each atom's representation already encodes its local neighbourhood, so a "per-contact" score is really a
per-contact-in-context score that can absorb some of the low-order cooperativity. The bias-correction
head absorbs more. So the form is a physically-motivated scaffold with learned slack for the
higher-order structure, not a claim that binding is exactly pairwise — and that is the honest version of
the inductive bias I am buying. And I get interpretability for free: I can read off which contacts contributed. So this rung
commits to two inductive biases at once — a genuinely heterogeneous covalent/non-covalent graph, and an
output that is a sum of pairwise atom-atom affinities. The dual interface readout the ladder has been
using is already shaped this way — a per-contact triple-product score summed over contacts, minus a bias
correction — so what changes here is not the readout's *form* but what *feeds* it and how the two
molecules' messages are built.

Start with the messages, relation by relation, now that I am committing to use the full edge features,
because the two edge types are physically different and should not share a mechanism. The covalent edges
inside each molecule are stiff, low-degree, and chemically meaningful, and I want the bond chemistry and
geometry to shape the message. I project each covalent edge's 17 features to the hidden width and inject
it into the message *additively before the nonlinearity*, `m_ij = ReLU(h_src + e_ij)`. Adding the edge
feature to the source feature before ReLU means a double bond in a ring sends a genuinely different
message than a single acyclic bond, and a sharp angle differs from a straight one — the geometry SchNet
and EGNN discarded is now shaping the message rather than being ignored. Aggregate by *sum* over
neighbours, not mean, and the reason is chemical: covalent degree is small and real. Trace it — a carbon
with four bonds should carry more incident structure into its update than a carbon with two; under a sum
the aggregate is roughly 4 message-units versus 2, preserving the valence distinction, whereas a mean
would return roughly the same per-neighbour average in both cases and wash the valence signal away. So
sum. Then a residual `rst = h + agg` so the atom keeps its own identity and the convolution only learns
the *update*, and a post-MLP of `Linear → Dropout(0.1) → LeakyReLU → BatchNorm1d`. The BatchNorm is
doing real work here and not by reflex: residual message passing lets activation magnitudes drift
upward across three stacked layers (each layer *adds* to the running representation), and normalizing
after each convolution keeps the scale controlled so the later layers see inputs in a stable range.
Separate weights for the ligand's covalent graph and the pocket's, because drug-like ligand chemistry
and protein-pocket chemistry are not the same distribution of bonds.

The non-covalent contacts across the interface are different in character, and the difference dictates a
different convolution. A pocket atom can contact many ligand atoms, the contact degree varies a lot from
atom to atom, and a contact's *strength* should depend on its geometry — a close, well-aligned contact
should transmit more than a grazing one. So here I do not sum raw source features; I let the projected
contact geometry *gate* the message multiplicatively, `m = h_src ⊙ e`, so a strong contact (its
geometric feature large in the right components) passes more of the source atom's signal and a marginal
one passes less. Then aggregate by *mean*, not sum, precisely because the degree is so variable, and this
is the mirror image of the covalent decision, so let me trace it too. Suppose pocket atom A sits in a
crowded shell with twenty weak contacts each gate-passing about 0.1 of a unit, while pocket atom B has
three strong, specific contacts each passing about 0.9. Under a *sum*, A aggregates ≈ 2.0 and B ≈ 2.7 —
nearly tied, so A's twenty incidental brushes almost match B's three real interactions purely by count.
Under a *mean*, A returns ≈ 0.1 and B ≈ 0.9, a nine-fold gap that reflects the *typical* contact
strength rather than the crowd size. Binding is set by strong specific contacts, not by how many atoms
happen to fall inside the 5 Å shell, so the mean is the right summary for the non-covalent side exactly
as the sum was right for the covalent side. A linear map of the mean aggregate plus a separate linear
map of the destination atom's own feature and a bias, `rst = fc_self(h_dst) + fc_neigh(mean) + bias`,
with `fc_neigh` applied *after* the mean (cheaper, and when in- and out-widths match, equivalent to
applying it per-message before averaging), Xavier-uniform init on both maps. Run it in both directions,
ligand→pocket and pocket→ligand, separate weights, because a contact summarized from the ligand side and
from the pocket side are two different views. Each layer runs all four convolutions in parallel from the
same inputs and sums per destination type — `lig_out = CIG_lig(lig) + NIG_{p→l}(poc, lig)`, `poc_out =
CIG_poc(poc) + NIG_{l→p}(lig, poc)` — because covalent and non-covalent influences on an atom are
additive. Three layers, hidden width 256, same depth reasoning as before: the contact graph is a local 5
Å shell and three rounds carry a protein atom's influence a few bonds into the ligand and back without
oversmoothing the complex into an indistinguishable soup.

Now the readout that *is* the second inductive bias. After three layers I score each non-covalent contact
with a low-rank triple product of the projected source atom, the projected destination atom, and the
projected *contact geometry*: `i_lp = prj_edge(e) ⊙ prj_src(lig_h)[src] ⊙ prj_dst(poc_h)[dst]`, collapsed
by a final `Linear(H, 1)` to one scalar per contact, summed over a complex's contacts via the
`inter_batch` index. The elementwise triple product is a low-rank trilinear scorer — it fires where the
source atom, the destination atom, and the contact geometry all agree in the same latent components,
which is exactly "this kind of atom meeting that kind of atom at this geometry is a favourable contact,"
and crucially it is *multiplicative*, so it supplies the src⊗dst cross term that the additive messages
never form. Do it in both directions for two affinity estimates per complex. But a raw, unweighted sum
over every contact within the 5 Å cutoff carries a systematic offset, and I should quantify it rather
than wave at it. If a complex has `k` contacts and each per-contact score has mean μ, the raw directional
sum runs about `kμ` — an *extensive* quantity that grows with contact count, and `k` grows with the size
of the complex and with how many incidental pairs happen to fall inside the cutoff. But the label
`-logKd/Ki` is not systematically larger for a bigger complex; a large ligand is not automatically a
tighter binder. So the raw sum has a spurious size term riding on top of the real binding signal. I hit
the wall the additive form always hits: the interpretability I bought comes with a size-dependent
nuisance. The fix is a learned, complex-specific bias correction that knows not all contacts deserve
equal weight — an attention over the contact edges. Compute a logit per contact from its source,
destination, and edge projections through a small head, softmax it *over each complex's contacts*, form
an attention-weighted triple-product aggregate, push it through a 2-layer `FC(H, 200)`, and subtract:
`pred_lp = atompairs_lp − bias_lp`, likewise `pred_pl`, per direction with separate weights.

Let me be honest about what that subtraction actually accomplishes, because it is easy to overstate. The
softmax weights sum to one, so the attention-weighted aggregate is an *intensive*, scale-stable quantity —
a convex combination of per-contact terms, not a `k`-growing sum. Subtracting an intensive correction
from an extensive sum does not literally cancel the `kμ` growth term by term; what it does is give the
model a learned, count-aware, per-complex reference level to subtract off, a soft baseline it can tune to
counteract the systematic size drift. It is a de-biasing prior, not an exact normalization, and whether
it is enough to keep contact count from leaking into the score is something the numbers will have to
tell me. I keep it because it is the principled shape of the correction even if it is not a guarantee.

That leaves the two corrected directional estimates, and here is the move that distinguishes this rung's
*training* from everything before it. The two views — ligand→pocket and pocket→ligand — summarize the
*same* interface from opposite molecules, so they ought to agree, and a disagreement is itself a signal
that the model is being internally inconsistent. So beyond fitting each head to the label, I add a term
that drives the two toward each other. The loss becomes three MSE terms averaged: fit `pred_lp` to truth,
fit `pred_pl` to truth, and a consistency term `MSE(pred_lp, pred_pl)` forcing the two views to agree.
The third term needs no labels — it is free supervision extracted from the symmetry of the interface —
and it acts like a two-member ensemble with a consistency prior: two estimators of the same quantity,
tied together so their average has lower variance than either alone and neither is allowed to wander into
a regime the other rejects. This is the one rung where the harness's plain-MSE default is not enough: a
single `forward` output cannot express a two-head, three-term objective. The harness anticipates exactly
this — it calls `compute_loss(batch, labels)` if the model exposes it, falling back to plain MSE
otherwise — so I expose `compute_loss`, compute both heads, and return `(MSE(lp, y) + MSE(pl, y) +
MSE(lp, pl)) / 3`. The third term costs nothing at inference; `forward` simply averages the two heads.

A shape and invariance check before I trust it. The two heads each reduce to `[B]` — `pred_lp` and
`pred_pl` are per-complex scalars after the `index_add_` over `inter_batch` and the bias subtraction — so
the three MSE terms are all between `[B]`-shaped tensors and `compute_loss` returns a scalar, which is
what the harness backpropagates. Invariance survives end to end exactly as before: every input is
frame-independent — the 35-dim atom one-hots, the 17-dim covalent features, the 11-dim contact features,
all of them built from angles, triangle areas, and distances, which are rigid-motion invariants — so
every message (additive-then-ReLU or gated), every aggregate (sum or mean), the triple product, the
attention softmax, and the bias subtraction are functions of invariant quantities, and coordinates never
appear. The prediction is invariant by construction, no augmentation.

So the falsifiable expectations against EGNN's numbers, benchmark by benchmark. The full 17/11-dim edge
geometry in the messages should recover the distance resolution EGNN lost on 2016 while keeping its
node-pair richness, so I expect 2016 to be where this rung gains *most* over EGNN — RMSE should drop well
below EGNN's 1.4213, and Rp well above its 0.7646, plausibly back past SchNet's 0.7767 since I now have
*both* resolution and node expressiveness, and on top of that the multiplicative triple-product scorer
supplies the cross term the additive message lacked. A clear 2016 win is the headline I am predicting. On
2013 I expect a modest gain over EGNN (1.4114 / 0.8093) — but I temper it with the calibration caution
from last rung: 2013's error was more about dynamic-range compression than comprehension, so unless the
sum-of-contacts form also sharpens the predicted range, 2013 RMSE may move only a little. The genuinely
open question is 2019. The consistency-regularized sum-of-contacts form is a hypothesis-class restriction
toward the physics, which *should* generalize at least as well as EGNN's 0.6175 there; but it is entirely
possible that the richer covalent-edge features — bond type, conjugation, ring membership — help less on
the temporally distant set than on the near-training core, because those covalent motifs are exactly the
kind of training-era chemistry that may not transfer, and the extra covalent capacity could even overfit
to it. So 2019 may move only a little, and I would not be shocked to see its RMSE sit a touch *below*
EGNN's while Rp holds. If that is the pattern — strong 2016 and 2013 gains but flat or slightly worse
2019 — it would say the contribution is real but concentrated where the chemistry is familiar, and it
sets up the final rung's question directly: whether all this dual-head, bias-corrected, consistency-
trained machinery is actually *necessary*, or whether its advantage is confined to the familiar core
set and does not survive the temporally distant split it was supposed to help. (The full scaffold
module is in the answer.)
