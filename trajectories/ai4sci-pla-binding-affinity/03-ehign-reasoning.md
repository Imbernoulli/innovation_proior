The EGNN numbers landed exactly on the split I predicted, and the split is the whole instruction for
this step. Against SchNet, EGNN lifted 2019 the most: RMSE 1.5624 → 1.4422 (down 0.1202) and Rp
0.5699 → 0.6175, clearing the 0.60 bar; it improved 2013 too (RMSE 1.4765 → 1.4114, Rp 0.7921 →
0.8093). But on 2016 it did exactly what I feared: RMSE up 1.3702 → 1.4213, Rp down 0.7767 → 0.7646.
EGNN won four of the six metric slots and lost two, both on the near-training core set — the precise
trade I named: it fed the distance raw as one rank-one-at-init channel where SchNet expanded it into 60
RBF channels, so on familiar CASF-2016 motifs, where fine distance resolution pays off, its coarser
geometry cost it even as its node-side richness won everywhere else. One number is the strongest
evidence the node-pair move bought *generalization* rather than fit: SchNet's RMSE gap between the
near-training 2016 set and the temporally distant 2019 set was 0.192, a wide chasm, whereas EGNN's is
1.4422 − 1.4213 = 0.021 — almost gone. That property I want to keep, not trade away, as I add geometry
back.

So two diagnoses now stack from opposite sides: SchNet was geometry-resolved but node-poor and
orientation-blind; EGNN is node-rich but geometry-thin. Both still feed their messages essentially one
scalar — the contact distance — while the batch carries an 11-dim geometric description of every edge
(angle max/sum/mean, triangle-area max/sum/mean, neighbour-distance max/sum/mean, pairwise L1 and L2)
and a 17-dim description of every covalent edge that adds bond type, conjugation, and ring membership.
The two failures are complementary — SchNet's resolution helped 2016 and its node-poverty hurt 2019;
EGNN's node richness helped 2019 and its thinning hurt 2016 — so the right move is not to pick a side
but to hold *both*: recover the resolution while keeping the node-pair expressiveness. Stop discarding
the full edge geometry and let it into the message.

There are inequivalent ways to obey that. The most literal fuses the two prior models directly — keep
EGNN's additive endpoint MLPs but feed the geometry side the full 11-dim vector — but it leaves two
things I now think matter. It keeps a *single* message mechanism for physically distinct edge types,
where the batch's separate 17-dim and 11-dim feature sets are inviting me to treat a stiff low-degree
bond and a soft variable-degree contact differently. And it keeps the pool-and-regress readout
untouched, when I am about to argue the readout *form* is a second axis worth moving. A more ambitious
route passes messages over the *angles* between edges via a line graph — capturing orientation directly,
the thing SchNet was blind to — but the edit surface forbids it: the harness hands me one precomputed
feature vector per edge, not the edge-adjacency (which edges share an atom, at what angle) a line-graph
message needs, and I cannot reconstruct it without coordinates. The angle information reaches me only as
the summary statistics already in `edge_attr`. So the route is: distinct covalent and non-covalent
mechanisms, each consuming the full edge feature, with the rest of the gain earned from the readout form.

And there is a sharper diagnosis hiding in how all three fills read out affinity, which I nearly missed
because they share a readout. They all pool an interface representation and regress it. But reconsider
what binding affinity *is* as an output: to first order the binding free energy is a *sum* of pairwise
contributions, each non-covalent ligand-atom/pocket-atom contact contributing some favourable or
unfavourable amount. That is a claim about the *form* of the prediction, and none of my fills enforce it
— they let affinity be any learned function of the pooled embedding, including functions that could not
be a sum of local contact terms. If I make the output *literally* a sum over interface contacts of a
per-contact atom-atom affinity, I restrict the hypothesis class to functions that already look like the
physics, which shrinks the space the model can overfit into — the cleanest kind of inductive bias, and
one that should help exactly on the held-out split. I should not oversell it: the sum-of-pairwise
picture is only *first-order*; real binding carries non-additive cooperativity, desolvation, and
conformational entropy a strict sum cannot represent. The escape is that each per-contact score is a
learned function of the *post-message-passing* atom features, which after three rounds already encode
each atom's local neighbourhood, so a "per-contact" score is really per-contact-in-context and can
absorb some low-order cooperativity, and the bias-correction head absorbs more. A physically-motivated
scaffold with learned slack, not a claim that binding is exactly pairwise — and I get interpretability
for free: I can read off which contacts contributed. The dual interface readout the ladder already uses
is shaped this way, so what changes is not the readout's *form* but what feeds it and how the two
molecules' messages are built.

Take the messages relation by relation, because the two edge types are physically different and should
not share a mechanism. The covalent edges inside each molecule are stiff, low-degree, and chemically
meaningful, so I project each edge's 17 features to the hidden width and inject them *additively before
the nonlinearity*, `m_ij = ReLU(h_src + e_ij)`: a double bond in a ring then sends a genuinely different
message than a single acyclic bond, and a sharp angle differs from a straight one. Aggregate by *sum*,
not mean, for a chemical reason — covalent degree is small and real: a carbon with four bonds should
carry more incident structure into its update than one with two, and under a sum the aggregate is ~4
message-units versus ~2, preserving the valence distinction that a mean would wash away. Then a residual
`rst = h + agg` so the atom keeps its identity and the convolution learns only the *update*, and a
post-MLP of `Linear → Dropout → LeakyReLU → BatchNorm1d`. The BatchNorm earns its place: residual
message passing lets activation magnitudes drift upward across three stacked layers, and normalizing
after each convolution keeps the scale stable for later layers. Separate weights for the ligand's and the
pocket's covalent graphs, since drug-like and protein-pocket bond distributions differ.

The non-covalent contacts are different in character. A pocket atom can contact many ligand atoms, the
contact degree varies a lot, and a contact's *strength* should depend on its geometry. So here I do not
sum raw source features; I let the projected contact geometry *gate* the message multiplicatively,
`m = h_src ⊙ e`, so a strong contact passes more of the source signal and a marginal one less. Then
aggregate by *mean*, the mirror of the covalent choice, precisely because degree is so variable: suppose
pocket atom A sits in a crowded shell of twenty weak contacts each passing ~0.1, while B has three strong
contacts each passing ~0.9. Under a sum, A aggregates ~2.0 and B ~2.7 — nearly tied, A's twenty
incidental brushes almost matching B's three real interactions by count. Under a mean, A returns ~0.1 and
B ~0.9, a nine-fold gap reflecting typical contact strength rather than crowd size. Binding is set by
strong specific contacts, not by how many atoms fall inside the 5 Å shell, so the mean is right for the
non-covalent side exactly as the sum was right for the covalent. A linear map of the mean aggregate plus
a separate linear map of the destination's own feature and a bias, `rst = fc_self(h_dst) +
fc_neigh(mean) + bias`, run in both directions with separate weights. Each layer runs all four
convolutions in parallel and sums per destination type — `lig_out = CIG_lig(lig) + NIG_{p→l}(poc, lig)`,
`poc_out = CIG_poc(poc) + NIG_{l→p}(lig, poc)` — additive because covalent and non-covalent influences
are. Three layers, width 256, same depth reasoning as before.

The readout is the ladder's shared dual bidirectional scorer, now fed the full edge features: score each
contact by a low-rank triple product of projected source atom, projected destination atom, and projected
*contact geometry*, collapse to one scalar, sum over a complex's contacts, both directions. The triple
product is *multiplicative*, so it supplies the src⊗dst cross term the additive messages never form. And
as established when I first built this head, the raw sum over a variable contact count carries a
size-dependent offset — `kμ`, extensive in `k` — that I counter per direction with an
attention-normalized bias correction: a softmax over each complex's contacts gives a scale-stable,
intensive reference level to subtract, a de-biasing prior rather than an exact cancellation.

That leaves the two corrected directional estimates, and here is the move that distinguishes this model's
*training*. The two views — ligand→pocket and pocket→ligand — summarize the *same* interface from
opposite molecules, so they ought to agree, and disagreement is itself a signal of internal
inconsistency. So beyond fitting each head to the label I add a term driving the two toward each other:
the loss becomes three averaged MSE terms — fit `pred_lp` to truth, fit `pred_pl` to truth, and a
consistency term `MSE(pred_lp, pred_pl)`. The third needs no labels — free supervision from the symmetry
of the interface — and acts like a two-member ensemble with a consistency prior: two estimators of the
same quantity, tied so their average has lower variance and neither wanders into a regime the other
rejects. This is the one place the harness's plain-MSE default is not enough, and the harness anticipates
it: it calls `compute_loss(batch, labels)` when the model exposes it. So I expose `compute_loss`
returning `(MSE(lp, y) + MSE(pl, y) + MSE(lp, pl)) / 3`; at inference `forward` simply averages the two
heads, and the two per-complex `[B]` scalars make every term well-shaped. Invariance survives end to end
as before — every input is frame-independent and no coordinate is ever formed.

So the falsifiable expectations against EGNN's numbers. The full 17/11-dim geometry in the messages
should recover the resolution EGNN lost on 2016 while keeping its node richness, and the multiplicative
triple-product scorer adds the cross term the additive message lacked, so I expect 2016 to gain *most* —
RMSE well below EGNN's 1.4213 and Rp above SchNet's 0.7767. A clear 2016 win is the headline. On 2013 I
expect only a modest gain, tempered by the calibration caution: 2013's error was dynamic-range
compression, not comprehension, and richer messages may not sharpen the range, so 2013 RMSE may barely
move. The open question is 2019. The consistency-regularized sum-of-contacts form is a hypothesis-class
restriction toward the physics, which *should* generalize at least as well as EGNN's 0.6175 there; but
the richer covalent features — bond type, conjugation, ring membership — are exactly the training-era
chemistry that may not transfer to the temporally distant set, and could even overfit it, so 2019 may
move only a little and I would not be shocked to see its RMSE sit near or a touch worse than EGNN's while
Rp holds. If that is the pattern — strong 2016, flat 2013, flat-or-worse 2019 — it says the contribution
is real but concentrated where the chemistry is familiar, and it sets up the final question directly:
whether all this dual-head, bias-corrected, consistency-trained machinery is actually *necessary*, or
whether its advantage does not survive the split it was meant to help. (The full scaffold module is in
the answer.)
