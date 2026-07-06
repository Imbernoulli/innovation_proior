The scoring objective is the whole point, but it bolts onto an evaluation regime, and the floor I have
to start from is just ranking a library against a target at all — under the constraints that actually
bite here. The library is $10^8$–$10^9$ molecules and the hit rate is savage: far below one in a
thousand of them truly bind a given pocket. That single number reframes everything downstream of it. It
says I am hunting needles, and it says the arithmetic of the objective is lopsided in a way I can
exploit — confirmed positives are the scarce, expensive thing, while correct negatives lie around in
essentially unlimited free supply, because a molecule grabbed at random is a non-binder with probability
better than $0.999$. And the metric is not average-case: BEDROC at $\alpha=80.5$ and EF at the top
0.5%/1%/5% put almost all of their weight on the handful of ranks at the very top and essentially none on
behavior across the bulk of the curve. So the score I design has to (1) place true binders at the *top*
specifically, (2) run over a billion molecules in feasible wall-clock, (3) not lean on scarce affinity
labels or hand-built decoy sets, and (4) generalize to targets never seen in training. Those four are not
a wishlist; each one, below, kills a candidate family outright, and watching them do it is how I find the
shape of the thing I should build.

Take the prior families in turn and put numbers to where they break, because the failures are quantitative
and they point somewhere specific. Docking samples candidate poses in the pocket and scores each with an
empirical force field; it works — on the hard benchmarks it is essentially the only family that reliably
beats chance — but the pose search is the bottleneck, order ten seconds of compute per compound. Push that
through the regime's own premise: $10^{10}$ compounds at ten seconds each is $10^{11}$ seconds, and dividing
by the $\approx 3.16\times10^{7}$ seconds in a year gives roughly three thousand years of compute for a single
screen. Worse than the absolute number is its *scaling*: the cost grows linearly *with the library*, which
is exactly backwards for a regime whose entire thesis is "screen a bigger library to find more hits."
Docking cannot be the answer at this scale. Affinity regression skips the pose search but trades it for
three coupled problems: it needs reliable affinity labels, of which perhaps $10^{4}$ exist across all of
public medicinal chemistry — a rounding error against a billion-molecule library; it sees almost no true
negatives during training, so false positives explode at inference; and every pocket–molecule pair needs
its own forward pass, so screening $T$ targets against a library of size $M$ is $T\times M$ network
evaluations that amortize across neither axis. The decoy classifiers fail in a subtler and more instructive
way: they manufacture negatives from a rule — pad a pocket's actives with property-matched ZINC decoys —
and a model trained that way learns the *decoy-construction rule*, not binding. Train on one benchmark's
decoys, test on another built by a different rule, and it collapses. That last failure is the one I should
carry forward as a design principle rather than just a cautionary tale: the negatives I feed a contrastive
boundary *are* what the model learns, so any negatives that come from a fixed rule teach a fixed, artificial
boundary. Whatever I build has to draw its negatives from the real distribution.

So let me stare at what screening actually asks for, because I think the field has been answering a harder
question than the one posed. Docking predicts a *pose*. Regression predicts an *affinity value*. But for
screening I need neither: I need to know *which* molecules are likely to bind, ranked, in the top fraction I
will ever inspect. That is a matching problem — given the pocket as a query, find the molecules that go with
it — and matching at corpus scale has a name and a template I trust: dense retrieval. There, relevance is a
plain dot product of two independently computed embeddings, $s(q,p)=E_Q(q)^\top E_P(p)$, one encoder per
side, deliberately *not* a cross-encoder that jointly attends over the pair. Let me make sure the factorized
form actually buys what I think, because the whole feasibility argument rests on it. If
$s(p,m)=g_\phi(p)^\top f_\theta(m)$, then $f_\theta(m)$ has no dependence on $p$ whatsoever: I can encode the
entire library once, offline, and cache the vectors; a new target is then one pocket encoding plus a
dot-product sweep — an approximate-nearest-neighbor query — against the cache. The expensive neural
computation is paid once per molecule, *ever*, not once per (target, molecule) pair. A cross-encoder
$k(p,m)$ has no such property: $p$ and $m$ are coupled inside the network, so nothing about $m$ can be
precomputed without knowing $p$, and I am back to the $T\times M$ forward passes that just sank regression.
So the restriction to a factorized dot-product score is not a modeling concession I make reluctantly; it is
precisely the property that converts "score a billion molecules" from thousands of CPU-years into one matrix
multiply against a cached index. It is the answer to constraint (2), and it is load-bearing.

There is a second, quieter reason to want two towers, and I only see it when I ask which *conformation* of
the molecule I feed in. At training time, from a solved complex, I have the ligand in its bound pose; at
screening time I have no bound pose — that is the entire point of not docking. A single-tower scorer eats the
joint complex, including the protein–ligand *cross*-distances, and those cross-distances are exactly what you
only know after you have placed the ligand in the pocket. So a single tower implicitly needs a pose, which
means it needs docking, which is the cost I am fleeing. Does the two-tower form escape that, or merely hide
it? The molecule encoder here is SE(3)-invariant, built from the intra-molecular pairwise-distance map
$D(c_m,c_m)$. A rigid re-placement of the same molecule — some rotation and translation — is by definition an
isometry: it leaves every pairwise distance exactly fixed, so $D(c_m,c_m)$ is *identically* unchanged and the
molecule tower's input is bit-for-bit the same before and after. The score does not move under re-posing, not
approximately but exactly, because the tower never sees anything that a rigid motion can change. The single
tower's cross term is the opposite: the same rigid motion that fixes the molecule's internal distances moves
every molecule-to-pocket distance, because no rotation can simultaneously fix a molecule's internal geometry
and its distances to a *fixed* pocket unless it is already correctly posed. So the dual-tower restriction I
adopted for throughput also buys freedom from docking at train and test time — I can train the molecule tower
on a cheap, cheminformatics-generated conformer and, to the extent the placement error is rigid, the score is
insensitive to it. The honest caveat is that this only covers the rigid part of the perturbation: a genuinely
different internal geometry — a flipped ring, a different rotamer — *does* change the intra-distance map and
can move the score, so the robustness is real but conditional on the conformer getting the internal shape
roughly right. Two independent arguments now point at the same skeleton: two towers, one dot product,
precompute-and-index.

Now training. I have positive pairs — assays where a tested ligand genuinely binds its target. I have almost
no labeled true negatives. But the savage hit rate hands me negatives for free: any other molecule is
essentially certainly a non-binder, and the cheapest such molecules are the ones already sitting in the batch.
This is in-batch negatives. Form the full similarity matrix $S$ with $S_{ij}=s(p_i,m_j)$; the matched pairs
are positives on the diagonal, and every off-diagonal cell is a pocket paired with some *other* example's
molecule — a negative by default. One matrix multiply over a batch of $N$ yields $N$ positives and $N(N-1)$
negatives, and crucially these negatives are real drug-like molecules that bind *some* pocket, so they are
realistic hard negatives rather than artifacts of a construction rule — exactly the shortcut that broke the
rule-based classifiers, sidestepped by construction. This is the answer to constraint (3): no external decoy
mining, no affinity labels, and the negatives come from the true molecule distribution.

The natural loss over that matrix is the contrastive softmax: treat a pocket's row as logits over candidate
molecules and ask the model to pick out its true binder, a categorical cross-entropy. Why this object and not
a margin or a regression toward $1/0$? Two reasons, and I want to check both rather than repeat them as
folklore. First, the optimum of the InfoNCE loss drives the score toward a density ratio,
$\exp(s)\propto p(m\mid p)/p(m)$ — "how much more likely is this molecule given this pocket than at random" —
which is precisely a *relevance* signal for ranking and, notably, not an absolute affinity I would have to
have labeled. Let me sanity-check that the density ratio is even the right thing to want on a toy where I can
compute it. Let a pocket take two values $A,B$ with equal prior and a molecule take three values, with
$p(x\mid A)=(0.6,0.3,0.1)$ and $p(x\mid B)=(0.1,0.4,0.5)$; the marginal is
$p(x)=\tfrac12(0.6,0.3,0.1)+\tfrac12(0.1,0.4,0.5)=(0.35,0.35,0.30)$. The density ratios for pocket $A$ are
$(0.6/0.35,\,0.3/0.35,\,0.1/0.30)=(1.71,\,0.86,\,0.33)$, so the molecule the ratio ranks first for $A$ is
molecule 1 — which is also $A$'s single most-likely molecule. The ordering the density ratio induces is the
relevance ordering, with no affinity value anywhere in sight. Good: the object the loss optimizes toward is a
ranker by construction. Second, minimizing this loss maximizes a lower bound on the pocket–molecule mutual
information, $I(p;m)\ge\log N-\mathcal{L}_N$, whose ceiling $\log N$ rises with the number of negatives. I
should check the direction of that dependence with real numbers: $\log 8=2.08$, $\log 64=4.16$,
$\log 192=5.26$, $\log 1024=6.93$ — monotone increasing, so a larger batch genuinely raises the ceiling on
what the objective can express, not merely the convenience of the run. I will not over-read this — the bound
is on *achievable* MI, not a guarantee any run reaches it — but it says a large batch is doing real work. And
I symmetrize, reading the matrix by columns as well as rows, so the space is good for retrieval in both
directions and the molecule embeddings cannot collapse into one cluster that looks alike to every pocket.

The temperature is doing more than it looks, and it is worth a number rather than a shrug. If I L2-normalize
the embeddings — which I will, for reasons in a moment — the similarities are cosines in $[-1,1]$, a tiny
range, and my worry is that over so narrow a span the softmax is nearly flat and carries almost no gradient.
Let me put a number to "nearly flat." Take $N=192$, a positive at cosine $0.9$, and the negatives clustered
near cosine $0.6$; with no scaling the positive's softmax probability is
$1/(1+191\,e^{-0.3})=1/(1+191\cdot0.741)=0.0070$, essentially the $1/192=0.0052$ of a uniform guess, for a
loss of $4.96$ against the uniform baseline $\log 192=5.26$ — a margin of three-tenths of a nat, almost no
signal. Multiply the same cosines by $14$ and the positive's probability jumps to
$1/(1+191\,e^{-14\cdot0.3})=1/(1+191\cdot0.0150)=0.26$, dropping the loss to $1.35$: now the true pair
genuinely wins. So the scale is not cosmetic — without it the loss has almost nothing to push on. But I
should not overshoot either: at scale $100$ the exponent gap is $30$, the positive's probability rounds to
$1$ and the loss collapses to $\approx10^{-11}$, a saturated, gradient-starved softmax, which is why an
enormous scale is as useless as none. I store the scale on a log axis (it is multiplicative and must stay
positive) and initialize it so the multiplier is $\approx 13$, in the discriminative-but-unsaturated band the
check just mapped out. And I *detach* the scale inside the loss: otherwise the objective could lower itself
by cranking the scale up — sharpening every softmax — without improving a single embedding, and it would run
away toward the degenerate saturated regime. Detaching forces the contrastive gradient into the embedding
directions at a stable scale.

Two head choices carry the rest, with their reasons. First, I do not dot-product on the encoder's raw [CLS]
feature. The contrastive objective is aggressive and will warp whatever space it acts on, and if that space
*is* the backbone feature, I degrade the rich pretrained representation the encoders spent their pretraining
budget building. A projection head absorbs the contrastive distortion and protects the backbone — the
representation *before* the head generalizes better than the one after — so I interpose a nonlinear head,
$\text{Linear}(d,d)\to\text{ReLU}\to\text{Linear}(d,128)$, into a modest 128-d comparison space, one head per
modality since the three backbones have different statistics and dimensions and there is no reason to force
them through shared weights. Second, after the head I L2-normalize each embedding onto the unit sphere,
turning the dot product into bounded, scale-free cosine; without it the model could cheat the softmax by
inflating vector norms instead of improving directions, and the temperature scaling would be fighting an
uncontrolled magnitude. Normalize, then apply the detached scale: a clean, well-posed pipeline.

Now I have to fit this generic dense-retrieval picture to *this* scaffold, and it differs in three ways I
should be honest about because each forces an adaptation. First, the data is grouped **by assay**, not as
clean one-positive-per-pocket pairs. The harness hands me `batch_list[i]=(s,e)`, the contiguous column span
of pocket $i$'s tested ligands, and `act_list[i]`, their activities — so a pocket typically owns *several*
ligand columns. The literal one-positive diagonal of the DrugCLIP recipe would, for pocket $i$, shove every
*other* genuine active of the same target into its negative set and push it away, which is exactly wrong. The
honest generalization is a multi-positive *column* term: log-softmax the *transpose* of the similarity
matrix so each ligand column becomes a distribution over query pockets, and for every ligand in block $i$ the
target class is "pocket $i$." That lets one pocket own all of its tested ligand columns without any of them
competing against the others. The mirror direction is row-local by design: for ligand $k$ in pocket $i$, I
build a row mask that is $0$ at column $k$ and $-\infty$ elsewhere, add it to the masked row, log-softmax,
and take $-\log p_k$ — a selected-pair NLL, with the real competition carried by the column term rather than
by contrasting a pocket's own actives against each other. Second, assays vary wildly in size, and a straight
sum over per-pocket terms lets one big assay dominate the batch gradient. Let me size the effect. Each
per-pocket column term is a sum of $L_i$ NLLs, each $O(1)$, so its magnitude scales like $L_i$; a batch with
a 36-ligand assay and a 4-ligand assay would weight them $36:4=9:1$ under a raw sum. Dividing by $L_i$ flips
to $1:1$, which over-corrects — a big, information-rich assay genuinely *should* count for more than a tiny
one. Dividing by $\sqrt{L_i}$ leaves each assay's weight scaling like $L_i/\sqrt{L_i}=\sqrt{L_i}$, so the
same pair weights $6:2=3:1$: sub-linear, the middle ground between $9:1$ and $1:1$. That $\sqrt{L_i}$
discipline is what I apply to every per-pocket term. Third, the in-batch construction has a quiet bug: an
off-diagonal cell can be a *true* binder I am about to push down as a negative. The scaffold hands me the
metadata to catch exactly this. Suppose the batch has pockets with UniProt ids $[U_1,U_2,U_1]$ and some
molecule column $j$ that is a $U_1$ binder living in pocket 2's span; then for pocket 0 (also $U_1$) that
column is an off-diagonal cell that is genuinely a binder of the same target — a false negative. So before
any softmax I build a boolean mask setting `mask[i,j]=True` whenever `uniprot_poc[i]==uniprot_mol[j]` (same
target) or `lig_smiles[j]` is in pocket $i$'s known-binder set, and `masked_fill` those cells to the dtype
minimum so they vanish from every softmax. Tracing the $[U_1,U_2,U_1]$ example, the mask lights up precisely
the cross-target-shared cells and leaves the diagonal positives untouched. On a sub-0.1% distribution a false
negative is the one signal-corrupting mistake I cannot afford, and the mask is cheap insurance against it.

At evaluation the score is the matching quantity, read off exactly as training defined it. I embed the
target's pockets and every candidate molecule, form `pocket_reps @ mol_reps.T`, and reduce per molecule by
the *max* over the target's pocket conformers — a molecule that binds *any* available conformation should
rank high, and the max is precisely "best match among the pockets I have" — then sort descending and read out
the early-recognition metrics. The reduction and the similarity are the same dot product the loss optimized,
so train and test agree.

Two things this floor deliberately leaves on the table, and naming them is the point of running it as the
weakest rung rather than jumping past it. The harness hands me a third tower — the ESM-2 protein-sequence
embedding `prot_emb` — and the per-ligand pIC50 in `act_list`; this fill ignores both. The loss is
pocket$\leftrightarrow$molecule only, the protein tower is projected but never enters `compute_loss`, and
nothing in the objective distinguishes a strong binder from a merely-okay one: two actives of the same target
are pulled toward it with equal force, with no preference for the stronger. That is exactly what makes this a
floor — a real, scalable scorer, but the plainest one the contract admits.

So reason about what this floor must do, because that is why I run it before anything cleverer. The
contrastive objective aligns binders with their target in cosine space, so it should already concentrate
actives near the top far better than chance; this is not a null model. I expect solid AUROC across all three
benchmarks and respectable EF on DUD-E, where the decoys are property-matched ZINC molecules that a clean
binder/non-binder cosine boundary separates well — that is the regime the in-batch contrastive object is
tailored for. But two weaknesses are baked into the fill by construction, and I can name the benchmark each
one should surface on. With the protein-sequence view dropped, the only target signal is a single sampled
pocket conformation; where that pocket is noisy or ambiguous I have no complementary, structure-free view to
fall back on. And nothing in the loss is *top-heavy* in the metric's sense: the objective treats every active
as an equal positive, while the metric rewards getting the *strongest* binders to the very top — so on a
benchmark where realistic, graded actives crowd together and the top few ranks decide the score, this fill
should leave enrichment on the table. LIT-PCBA is exactly that benchmark — confirmed actives versus confirmed
inactives, no artificial-decoy crutch — so I expect vanilla CLIP to look weakest there: whatever AUROC it
manages, its BEDROC and EF should be soft, because it ranks binders above decoys without ranking *strong*
binders above *weak* ones. The falsifiable version of that expectation is directional and I will hold myself
to it: the ordering weakness should show up as the largest gap between LIT-PCBA and DUD-E on precisely the
top-heavy metrics (BEDROC, EF@0.5%), not on AUROC alone. Whatever the precise split turns out to be, the
diagnosis is already pointed at the next rung: I have an *ordering* problem sitting on top of a *separation*
problem, and the two levers to fix it — the unused per-ligand activities and the unused sequence tower — are
sitting right there in the scaffold, projected and waiting. At rung one my fill is the plainest contrastive
scorer the scaffold admits: three NonLinearHeads, L2-normalized Euclidean embeddings, a detached log-scale,
the assay-grouped symmetric in-batch softmax with false-negative masking and $\sqrt{L_i}$ normalization,
max-over-pockets scoring — and the protein tower and the activity numbers left on the table. (The full
scaffold module is in the answer.)
