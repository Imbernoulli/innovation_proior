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
labels or hand-built decoy sets, and (4) generalize to targets never seen in training. Each of the four,
below, kills a candidate family outright.

Take the prior families in turn. Docking samples candidate poses in the pocket and scores each with an
empirical force field; it works — on the hard benchmarks it is essentially the only family that reliably
beats chance — but the pose search is the bottleneck, order ten seconds of compute per compound. $10^{10}$
compounds at ten seconds each is $10^{11}$ seconds, and dividing by the $\approx 3.16\times10^{7}$ seconds
in a year gives roughly three thousand years of compute for one screen — and the cost grows linearly *with
the library*, exactly backwards for a regime whose whole thesis is "screen a bigger library to find more
hits." Affinity regression skips the pose search but trades it for three coupled problems: it needs
reliable affinity labels, of which perhaps $10^{4}$ exist across all of public medicinal chemistry — a
rounding error against a billion-molecule library; it sees almost no true negatives during training, so
false positives explode at inference; and every pocket–molecule pair needs its own forward pass, so
screening $T$ targets against a library of size $M$ is $T\times M$ network evaluations that amortize
across neither axis. The decoy classifiers fail in a subtler and more instructive way: they manufacture
negatives from a rule — pad a pocket's actives with property-matched ZINC decoys — and a model trained
that way learns the *decoy-construction rule*, not binding; train on one benchmark's decoys, test on
another built by a different rule, and it collapses. That failure is a design principle, not just a
cautionary tale: the negatives I feed a contrastive boundary *are* what the model learns, so any negatives
from a fixed rule teach a fixed, artificial boundary. Whatever I build has to draw its negatives from the
real distribution.

So let me stare at what screening actually asks for, because the field seems to be answering a harder
question than the one posed. Docking predicts a *pose*. Regression predicts an *affinity value*. But for
screening I need neither: I need to know *which* molecules are likely to bind, ranked, in the top fraction
I will ever inspect. That is a matching problem — given the pocket as a query, find the molecules that go
with it — and matching at corpus scale has a name and a template I trust: dense retrieval, where relevance
is a plain dot product of two independently computed embeddings, $s(q,p)=E_Q(q)^\top E_P(p)$, one encoder
per side, deliberately *not* a cross-encoder that jointly attends over the pair. The factorization is what
buys feasibility. If $s(p,m)=g_\phi(p)^\top f_\theta(m)$, then $f_\theta(m)$ has no dependence on $p$: I
encode the entire library once, offline, cache the vectors, and a new target is one pocket encoding plus a
dot-product sweep — an approximate-nearest-neighbor query — against the cache. The expensive neural
computation is paid once per molecule *ever*, not once per (target, molecule) pair. A cross-encoder
$k(p,m)$ couples $p$ and $m$ inside the network, so nothing about $m$ can be precomputed without knowing
$p$, and I am back to the $T\times M$ forward passes that just sank regression. So the restriction to a
factorized dot-product score is not a modeling concession; it is the property that converts "score a
billion molecules" from thousands of CPU-years into one matrix multiply against a cached index. That is
constraint (2).

There is a second, quieter reason to want two towers, and I only see it when I ask which *conformation* of
the molecule I feed in. At training time, from a solved complex, I have the ligand in its bound pose; at
screening time I have none — that is the entire point of not docking. A single-tower scorer eats the joint
complex, including the protein–ligand *cross*-distances, and those are exactly what you only know after
placing the ligand in the pocket: a single tower implicitly needs a pose, which means it needs docking,
the cost I am fleeing. Does the two-tower form escape that or merely hide it? The molecule encoder here is
SE(3)-invariant, built from the intra-molecular pairwise-distance map $D(c_m,c_m)$. A rigid re-placement
of the same molecule is an isometry: it leaves every pairwise distance fixed, so $D(c_m,c_m)$ is
identically unchanged and the molecule tower's input is bit-for-bit the same before and after. The score
does not move under re-posing, exactly, because the tower never sees anything a rigid motion can change.
The single tower's cross term is the opposite: the same rigid motion moves every molecule-to-pocket
distance, because no rotation can fix a molecule's internal geometry and its distances to a *fixed* pocket
at once unless it is already correctly posed. So the dual-tower restriction I adopted for throughput also
buys freedom from docking at train and test time — I can train the molecule tower on a cheap
cheminformatics conformer and, to the extent the placement error is rigid, the score is insensitive to it.
The honest caveat: this covers only the rigid part — a genuinely different internal geometry, a flipped
ring or a different rotamer, *does* change the intra-distance map and can move the score, so the robustness
is conditional on the conformer getting the internal shape roughly right.

Now training. I have positive pairs — assays where a tested ligand genuinely binds its target — and almost
no labeled true negatives. But the savage hit rate hands me negatives for free: any other molecule is
essentially certainly a non-binder, and the cheapest ones are already sitting in the batch. Form the full
similarity matrix $S$ with $S_{ij}=s(p_i,m_j)$; the matched pairs are positives on the diagonal, every
off-diagonal cell is a pocket paired with some *other* example's molecule, a negative by default. One
matrix multiply over a batch of $N$ yields $N$ positives and $N(N-1)$ negatives, and these negatives are
real drug-like molecules that bind *some* pocket — realistic hard negatives, not artifacts of a
construction rule, sidestepping the shortcut that broke the rule-based classifiers. No external decoy
mining, no affinity labels, negatives drawn from the true molecule distribution: that is constraint (3).

The natural loss over that matrix is the contrastive softmax: treat a pocket's row as logits over
candidate molecules and ask the model to pick out its true binder, a categorical cross-entropy. I want
this object, not a margin or a regression toward $1/0$, for two reasons. Its optimum drives the score
toward a density ratio, $\exp(s)\propto p(m\mid p)/p(m)$ — how much more likely this molecule is given
this pocket than at random, which is precisely a *relevance* signal for ranking and not an absolute
affinity I would have to have labeled. And minimizing it maximizes a lower bound on the pocket–molecule
mutual information, $I(p;m)\ge\log N-\mathcal{L}_N$, whose ceiling $\log N$ rises with the number of
negatives — so a larger batch is doing real work, not just a convenience, though the bound is on
*achievable* MI and no run is guaranteed to reach it. And I symmetrize, reading the matrix by columns as
well as rows, so the space is good for retrieval in both directions and the molecule embeddings cannot
collapse into one cluster that looks alike to every pocket.

The temperature is doing more than it looks. If I L2-normalize the embeddings — which I will, for reasons
in a moment — the similarities are cosines in $[-1,1]$, a tiny range over which the softmax is nearly flat
and carries almost no gradient. Put a number to it: $N=192$, a positive at cosine $0.9$, negatives near
$0.6$; unscaled, the positive's softmax probability is $1/(1+191\,e^{-0.3})=0.0070$, barely above the
$1/192=0.0052$ of a uniform guess — a loss of $4.96$ against the uniform baseline $\log 192=5.26$, three
tenths of a nat of signal. Multiply the same cosines by $14$ and the probability jumps to
$1/(1+191\,e^{-4.2})=0.26$, dropping the loss to $1.35$: the true pair genuinely wins. But overshoot to
scale $100$ and the exponent gap is $30$, the probability rounds to $1$, and the loss collapses to
$\approx10^{-11}$ — a saturated, gradient-starved softmax. So I store the scale on a log axis (it is
multiplicative and must stay positive) and initialize the multiplier at $\approx 13$, in the
discriminative-but-unsaturated band. And I *detach* it inside the loss: otherwise the objective lowers
itself by cranking the scale up — sharpening every softmax — without improving a single embedding, running
away toward the saturated regime. Detaching forces the contrastive gradient into the embedding directions
at a stable scale.

Two head choices carry the rest. First, I do not dot-product on the encoder's raw [CLS] feature: the
contrastive objective is aggressive and warps whatever space it acts on, and if that space *is* the
backbone feature I degrade the rich pretrained representation the encoders spent their pretraining budget
building — the representation *before* a projection head generalizes better than the one after. So I
interpose a nonlinear head, $\text{Linear}(d,d)\to\text{ReLU}\to\text{Linear}(d,128)$, into a modest 128-d
comparison space, one head per modality since the three backbones have different statistics and dimensions.
Second, after the head I L2-normalize each embedding onto the unit sphere, turning the dot product into
bounded, scale-free cosine; without it the model could cheat the softmax by inflating vector norms instead
of improving directions, and the temperature scaling would be fighting an uncontrolled magnitude.
Normalize, then apply the detached scale.

Now fit this generic picture to *this* substrate, which differs in three ways, each forcing an adaptation.
First, the data is grouped **by assay**, not as clean one-positive-per-pocket pairs. The harness hands me
`batch_list[i]=(s,e)`, the contiguous column span of pocket $i$'s tested ligands, and `act_list[i]` their
activities — so a pocket typically owns *several* ligand columns. The literal one-positive diagonal would,
for pocket $i$, shove every *other* genuine active of the same target into its negative set and push it
away, which is exactly wrong. The honest generalization is a multi-positive *column* term: log-softmax the
*transpose* of the similarity matrix so each ligand column becomes a distribution over query pockets, and
for every ligand in block $i$ the target class is "pocket $i$" — one pocket owning all its ligand columns
without any of them competing against the others. The mirror direction is row-local: for ligand $k$ in
pocket $i$ I build a row mask that is $0$ at column $k$ and $-\infty$ elsewhere, add it to the masked row,
log-softmax, and take $-\log p_k$ — a selected-pair NLL, with the real competition carried by the column
term rather than by contrasting a pocket's own actives against each other. Second, assays vary wildly in
size and a straight sum lets one big assay dominate the batch gradient. A per-pocket column term is a sum
of $L_i$ NLLs each $O(1)$, so its magnitude scales like $L_i$: a batch with a 36-ligand assay and a
4-ligand assay would weight them $9:1$. Dividing by $L_i$ flips to $1:1$, which over-corrects — a big,
information-rich assay genuinely *should* count for more. Dividing by $\sqrt{L_i}$ leaves each assay's
weight scaling like $\sqrt{L_i}$, so the same pair weights $6:2=3:1$: the sub-linear middle ground I apply
to every per-pocket term. Third, an off-diagonal in-batch cell can be a *true* binder I am about to push
down as a negative. The scaffold hands me the metadata to catch exactly this: before any softmax I build a
boolean mask setting `mask[i,j]=True` whenever `uniprot_poc[i]==uniprot_mol[j]` (same target) or
`lig_smiles[j]` is in pocket $i$'s known-binder set, and `masked_fill` those cells to the dtype minimum so
they vanish from every softmax. On a sub-0.1% distribution a false negative is the one signal-corrupting
mistake I cannot afford, and the mask is cheap insurance against it.

At evaluation the score is the matching quantity, read off exactly as training defined it: embed the
target's pockets and every candidate molecule, form `pocket_reps @ mol_reps.T`, and reduce per molecule by
the *max* over the target's pocket conformers — a molecule that binds *any* available conformation should
rank high — then sort descending and read out the early-recognition metrics. Same dot product the loss
optimized, so train and test agree.

Two things this floor deliberately leaves on the table, and naming them is the point of running it as the
weakest step first. The harness hands me a third tower — the ESM-2 protein-sequence embedding `prot_emb` —
and the per-ligand pIC50 in `act_list`; this fill ignores both. The loss is pocket↔molecule only, the
protein tower is projected but never enters `compute_loss`, and nothing in the objective distinguishes a
strong binder from a merely-okay one: two actives of the same target are pulled toward it with equal force,
with no preference for the stronger. That is what makes this a floor — a real, scalable scorer, but the
plainest the contract admits.

So what must this floor do? The contrastive objective aligns binders with their target in cosine space,
so it should already concentrate actives near the top far better than chance — I expect solid AUROC across
all three benchmarks and respectable EF on DUD-E, where the property-matched ZINC decoys separate cleanly
under a binder/non-binder cosine boundary. But two weaknesses are baked in by construction. With the
protein-sequence view dropped, the only target signal is a single sampled pocket conformation, with no
structure-free view to fall back on where that pocket is noisy or ambiguous. And nothing in the loss is
top-heavy in the metric's sense: it treats every active as an equal positive, while the metric rewards
getting the *strongest* binders to the very top. LIT-PCBA — confirmed actives versus confirmed inactives,
no artificial-decoy crutch — is exactly where that should bite: whatever AUROC it manages, its BEDROC and
EF should be soft, because it ranks binders above decoys without ranking strong binders above weak ones.
Directionally the ordering weakness should surface as the largest gap between LIT-PCBA and DUD-E on the
top-heavy metrics (BEDROC, EF@0.5%), not on AUROC alone. Either way the diagnosis already points forward:
an *ordering* problem sitting on top of a *separation* problem, and the two levers to fix it — the unused
per-ligand activities and the unused sequence tower — are sitting projected and waiting in the scaffold.
The full module is in the answer.
