The EGNN numbers landed exactly on the split I predicted, and the split is the whole instruction for
this rung. Against SchNet, EGNN lifted 2019 the most — RMSE 1.4422 (down from 1.5624) and Rp 0.6175
(up from 0.5699), clearing the 0.60 bar I set — and improved 2013 across the board (1.4114 / 0.8093 vs
1.4765 / 0.7921). That confirms the node-pair-aware message buys real generalization on the temporal
holdout, where reasoning beats memorization. But on 2016 it did exactly what I feared: RMSE went *up*
to 1.4213 (from SchNet's 1.3702) and Rp down to 0.7646 (from 0.7767). That is the trade I named — EGNN
fed the distance raw as one channel where SchNet expanded it into 60 RBF channels, so on the
near-training CASF-2016 core set, where fine distance resolution on familiar motifs pays off, EGNN's
coarser geometry cost it even as its node-side richness won everywhere else. So I now have two diagnoses
stacked, and they point at the same culprit from opposite sides: SchNet was geometry-resolved but
node-poor and orientation-blind; EGNN is node-rich but geometry-thin. *Both* are still feeding their
messages essentially one scalar — the contact distance — while the batch carries an 11-dim geometric
description of every edge (angle max/sum/mean, triangle-area max/sum/mean, neighbour-distance
max/sum/mean, pairwise L1 and L2) and a 17-dim description of every covalent edge that adds bond type,
conjugation, and ring membership. Both rungs threw almost all of that away. The instruction is
unambiguous: stop discarding the full edge geometry and let it into the message, recovering the
resolution SchNet had while keeping the node-pair expressiveness EGNN bought.

But I want to push past "use more features," because there is a second, sharper diagnosis hiding in how
all three fills so far read out the affinity. They all pool an interface representation and regress it.
Let me reconsider what binding affinity *is* as an output, not just as an input. The free energy of
binding is, to first order, a *sum* of pairwise contributions — each non-covalent contact between a
ligand atom and a pocket atom contributes some favourable or unfavourable amount, and the total is
their sum. That is a strong claim about the *form* of the prediction, and none of my fills enforce it:
they let the affinity be any learned function of pooled node statistics. If I instead make the output
literally a sum over interface contacts of a per-contact atom-atom affinity, I restrict the model to
functions that look like the physics, which should generalize better, and I get interpretability for
free — I can read off which contacts contributed. So this rung commits to two inductive biases at once:
a genuinely heterogeneous covalent/non-covalent graph (which the skeleton already gives me), and an
output that is a sum of pairwise atom-atom affinities. The dual interface readout the ladder has been
using is already shaped this way — a per-contact triple-product score summed over contacts, minus a
bias correction — so what changes at this rung is not the readout's *form* but what *feeds* it and how
the two molecules' messages are built.

Start with the messages, relation by relation, now that I am committing to use the full edge features.
The covalent edges inside each molecule are stiff, low-degree, and chemically meaningful, and I want
the bond chemistry and geometry to shape the message. I project each covalent edge's 17 features to the
hidden width and inject it into the message *additively before the nonlinearity*: `m_ij = ReLU(h_src +
e_ij)`. Adding the edge feature to the source feature and passing through ReLU means a double bond in a
ring sends a different message than a single bond, a sharp angle differs from a straight one — the
geometry SchNet and EGNN discarded is now in the message. Aggregate by *sum* over neighbours, not mean,
because covalent degree is small and chemically real: a carbon with four bonds genuinely has more
incident structure than one with two, and averaging would wash that away. Then a residual `rst = h +
agg` so the atom keeps its identity and the conv only learns the update, and a post-MLP of
`Linear → Dropout(0.1) → LeakyReLU → BatchNorm1d` — the BatchNorm doing real work here, because
residual message passing lets activation magnitudes drift across three layers and normalizing after
each conv keeps the scale controlled. Separate weights for the ligand's covalent graph and the pocket's,
because drug-like ligand chemistry and protein-pocket chemistry are not identical.

The non-covalent contacts across the interface are different in character: a pocket atom can contact
many ligand atoms, the contact degree varies a lot, and a contact's *strength* should depend on its
geometry. So here I do not sum raw source features — I let the projected contact geometry *gate* the
message multiplicatively, `m = h_src ⊙ e`, so a close, well-oriented contact (its geometric feature
large in the right components) passes more of the source atom's signal and a marginal one passes less.
Then aggregate by *mean*, not sum, precisely because the degree is so variable: summing would let an
atom with twenty weak contacts swamp one with three strong ones by sheer count, whereas the mean asks
"what is the typical gated message into this atom." A linear map of the mean aggregate plus a separate
linear map of the destination atom's own feature and a bias, `rst = fc_self(h_dst) + fc_neigh(mean) +
bias`, with `fc_neigh` applied *after* the mean (cheaper and, when in- and out-widths match,
equivalent), Xavier-uniform init on both maps. Run this in both directions, ligand→pocket and
pocket→ligand, separate weights, because a contact summarized from the ligand side and from the pocket
side are two different views. Each layer runs all four convolutions in parallel from the same inputs and
sums per destination type — `lig_out = CIG_lig(lig) + NIG_{p→l}(poc, lig)`, `poc_out = CIG_poc(poc) +
NIG_{l→p}(lig, poc)` — because covalent and non-covalent influences on an atom are additive. Three
layers, hidden width 256, same depth reasoning as before: the contact graph is a local 5 Å shell and
three rounds suffice without oversmoothing.

Now the readout that *is* the second inductive bias. After three layers I score each non-covalent
contact with a low-rank triple product of the projected source atom, the projected destination atom,
and the projected *contact geometry*: `i_lp = prj_edge(e) ⊙ prj_src(lig_h)[src] ⊙ prj_dst(poc_h)[dst]`,
collapsed by a final `Linear(H,1)` to one scalar per contact, summed over a complex's contacts via the
`inter_batch` index. The elementwise triple product is a low-rank bilinear scorer — it fires where the
source atom, the destination atom, and the contact geometry all agree, which is exactly "this kind of
atom meeting that kind of atom at this geometry is a favourable contact." Do it in both directions for
two affinity estimates per complex. But a raw, unweighted sum over every contact within the 5 Å cutoff
carries a systematic offset: complexes with more atoms simply have more contacts in the shell, many of
them incidental rather than favourable, so the sum drifts with size and with how many spurious pairs
fall inside the cutoff, independent of true binding strength. I hit the same wall the additive form
always hits — interpretability buys in a size-dependent nuisance. The fix is a learned, complex-specific
bias correction that knows not all contacts deserve equal weight: an attention over the contact edges.
Compute a logit per contact from its source, destination, and edge projections through a small head,
softmax it *over each complex's contacts* (so the weights sum to one and the correction is scale-stable
regardless of contact count — exactly the size-dependence I needed to kill), form an attention-weighted
triple-product aggregate, push it through a 2-layer `FC(H, 200)`, and subtract: `pred_lp =
atompairs_lp − bias_lp`, likewise `pred_pl`. Per direction, separate weights.

That leaves the two corrected directional estimates, and here is the move that distinguishes this rung's
*training* from everything before it. The two views — ligand→pocket and pocket→ligand — summarize the
*same* interface from opposite molecules, so they ought to agree, and a disagreement is a signal the
model is being inconsistent. So beyond fitting each to the label, I add a term that drives the two
toward each other. The loss is three MSE terms averaged: fit `pred_lp` to truth, fit `pred_pl` to truth,
and a consistency term `MSE(pred_lp, pred_pl)` forcing the two views to agree. This is the one rung
where the harness's plain-MSE default is not enough — a single `forward` output cannot express a
two-head, three-term objective. The harness anticipates exactly this: it calls `compute_loss(batch,
labels)` if the model exposes it, falling back to plain MSE otherwise. So I expose `compute_loss`,
compute both heads, and return `(MSE(lp,y) + MSE(pl,y) + MSE(lp,pl)) / 3`; the third term costs nothing
at inference and acts as multi-view distillation, pushing both directional heads to encode a consistent
interface, which should help generalization. At inference `forward` averages the two heads. Invariance
survives end to end exactly as before — every input (chemical node features, angle/area/distance edge
features) is frame-independent, so every message, aggregate, triple product, attention, and bias is
invariant, and coordinates never appear.

So the falsifiable expectations against EGNN's numbers. The full 17/11-dim edge geometry in the messages
should recover the distance resolution EGNN lost on 2016 while keeping its node-pair richness, so I
expect 2016 to be where this rung gains *most* over EGNN — RMSE should drop well below EGNN's 1.4213 and
Rp well above its 0.7646, plausibly back past SchNet's 0.7767 since I now have *both* resolution and
node expressiveness. I expect a clear 2016 improvement to be the headline. On 2013 I expect a modest
gain over EGNN (1.4114 / 0.8093). The open question is 2019: the consistency-regularized sum-of-contacts
form should generalize at least as well as EGNN's 0.6175 there, but it is *possible* the richer
covalent-edge features (bond type, ring membership) help less on the temporally distant set than on the
near-training core, so 2019 may move only a little and could even sit a touch below EGNN if the extra
covalent capacity overfits to training-era chemistry. If that happens — strong 2016/2013 gains but flat
or slightly worse 2019 — it would say the contribution is real but concentrated where the chemistry is
familiar, and it sets up the final rung's question: whether a *cleaner geometric* message (distance
through an RBF-style continuous filter, fused per atom, summed because affinity is extensive) can hold
the 2013 sharpness this rung wins while not paying the 2019 cost — i.e. whether the heavy dual-head,
bias-corrected, consistency-trained machinery is actually necessary, or whether a leaner geometric core
generalizes better on the hardest split. (The full scaffold module is in the answer.)
