The scoring objective is the whole point, but it bolts onto an evaluation regime, and the floor I have
to start from is just ranking a library against a target at all — under the constraints that actually
bite here. The library is $10^8$–$10^9$ molecules and the hit rate is savage, far below one in a
thousand, so almost any molecule I grab is, with overwhelming probability, a non-binder. That single
number reframes everything. It says I am hunting needles, and it says there is a free, essentially
unlimited supply of correct negatives lying around while confirmed positives are the scarce thing. And
the metric I am judged on — BEDROC at $\alpha=80.5$, EF at the top 0.5%/1%/5% — is violently top-heavy:
almost all the credit is for the handful of ranks at the very top, none for average behavior over the
whole curve. So I need a score that (1) puts true binders at the *top* specifically, (2) can run over a
billion molecules in feasible time, (3) does not lean on scarce affinity labels or hand-built decoy
sets, and (4) generalizes to targets never seen in training.

Look at what the prior families do and where each breaks. Docking samples poses and scores them
physically; it works, it is the only family that reliably beats chance on the hard benchmarks, but the
per-compound pose sampling is the bottleneck — order ten seconds a compound, so ten billion compounds is
thousands of years. The cost scales *with the library*, which is exactly backwards for a regime whose
premise is "use a bigger library." Affinity regression skips the sampling but needs reliable affinity
labels, of which there are only ~$10^4$, sees almost no true negatives so false positives explode, and
at inference every pocket-molecule pair needs a full forward pass — screening many targets is
(#targets)×(#library) network evaluations, which does not amortize at all. The decoy classifiers are
worse in a subtler way: they manufacture negatives from a rule, and it has been shown cleanly that such
a model learns the *decoy-construction rule* and not binding — train on one benchmark's decoys, test on
another built differently, and it collapses. The negatives I feed a contrastive boundary *are* what the
model learns, so artificial decoys teach an artificial boundary.

So let me stare at what screening actually asks for, because I think the field has been answering a
harder question than the one posed. Docking predicts a *pose*. Regression predicts an *affinity value*.
But for screening I need neither: I need to know *which* molecules are likely to bind, ranked, in the
top fraction I will ever inspect. That is a matching problem — given the pocket as a query, find the
molecules that go with it. It is retrieval. And the moment I say "retrieval," the template I trust snaps
into focus: dense passage retrieval defined relevance as a plain dot product of two independently
computed embeddings, $s(q,p)=E_Q(q)^\top E_P(p)$, one encoder per side, deliberately *not* a
cross-encoder that jointly attends over the pair. The dot-product form is decomposable: I can embed the
whole library once, cache the vectors, and every new target is one pocket encoding plus a
nearest-neighbor search against the cache. The expensive neural computation is paid once per molecule,
ever — not once per (target, molecule) pair. That restriction to a factorized score is not a modeling
concession; it is the entire reason the thing runs at scale, and it is the answer to the throughput
constraint that sank docking and regression.

Now training. I have positive pairs — assays where a tested ligand genuinely binds its target. I have
almost no labeled true negatives. But the savage hit rate hands me negatives for free: any other
molecule is essentially certainly a non-binder, and the cheapest other molecules around are the ones
already in my batch. This is in-batch negatives. Take the batch's pocket and molecule embeddings, form
the full similarity matrix $S$ with $S_{ij}=s(p_i,m_j)$; the matched pairs are the positives, every
other cell is a negative-by-default. One matrix multiply gives me a pile of realistic hard negatives —
real drug-like molecules that bind *some* pocket — with no external decoy mining and no rule to overfit,
which is exactly the shortcut that broke the rule-based classifiers. The natural loss over that matrix is
the contrastive softmax: treat a pocket's row as logits over candidate molecules and ask the model to
pick its true binder, a categorical cross-entropy. Two reasons this is the right object and not a
margin or a regression-to-1/0: its optimum drives the score toward a density ratio
$p(\text{molecule}\mid\text{pocket})/p(\text{molecule})$ — precisely the relevance signal for ranking,
not an absolute affinity I would have to have labeled — and minimizing it maximizes a lower bound on the
pocket-molecule mutual information that *tightens as the batch grows*, so a larger batch is provably
better, not just convenient. I symmetrize over rows and columns so the space is good for retrieval in
both directions and the molecule embeddings cannot collapse into a cluster that all looks alike to a
pocket.

Two head choices, with their reasons. First, I do not dot-product directly on the encoder's [CLS]
feature: the contrastive objective is aggressive and will warp whatever space it acts on, and if that
space *is* the backbone feature, I degrade the rich pretrained representation. A projection head absorbs
the contrastive distortion and protects the backbone — the representation *before* the head generalizes
better than the one after. So a nonlinear head, Linear$(d,d)\!\to\!$ReLU$\to$Linear$(d,128)$, into a
modest 128-d space, one head per modality since the three backbones have different statistics and
dimensions. Second, after the head I L2-normalize each embedding onto the unit sphere, turning the dot
product into cosine — bounded and scale-free. Without it the model could cheat the softmax by inflating
vector norms instead of improving directions. But cosine lives in $[-1,1]$, far too compressed for a
peaked softmax, so I scale the logits up by an inverse temperature, stored on a log axis and initialized
so the multiplier is ~13 (CLIP-ish). And I *detach* that scale inside the loss: otherwise the objective
trivially lowers itself by cranking the scale up — sharpening every softmax — without improving a single
embedding, and it would run away. Detaching makes the contrastive gradients sharpen the embedding
directions at a stable scale.

Now I have to fit this to *this task's* data and edit surface, and here the scaffold differs from the
generic dense-retrieval picture in three ways I should be honest about. First, the data is grouped **by
assay**, not as clean one-positive-per-pocket pairs. The harness hands me `batch_list[i]=(s,e)`, the
contiguous column span of pocket $i$'s tested ligands, and `act_list[i]`, their activities — so a pocket
typically owns *several* ligand columns. The clean one-positive diagonal of DrugCLIP would, for pocket
$i$, shove every *other* genuine active of the same target into its negative set and push it away. So I
cannot keep the literal diagonal. The honest generalization is a column term: log-softmax the *transpose*
of the similarity matrix so each ligand column is a distribution over query pockets, and for every
ligand in block $i$ the target class is "pocket $i$." That lets a pocket own all its tested ligand
columns. The mirror direction is row-local and deliberately so: for ligand $k$ in pocket $i$, I build a
row mask that is $0$ at column $k$ and $-\infty$ elsewhere, add it to the masked row, log-softmax, and
take $-\log p_k$ — a selected-pair NLL, with the real competition carried by the column term, not by
contrasting a pocket's own actives against each other. Second, assays vary wildly in size — some screen a
handful of ligands, some dozens — so a straight sum lets one big assay dominate the batch gradient.
Dividing by $L_i$ over-corrects (a big, information-rich assay genuinely *should* count for more); the
middle ground is $1/\sqrt{L_i}$, sub-linear, applied to each per-pocket term. Third, the in-batch
construction has a quiet bug: an off-diagonal cell can be a *true* binder I am about to push down as a
negative — same target, or the same known ligand. The scaffold gives me metadata for exactly this, so
before any softmax I build a boolean mask, `mask[i,j] = True` whenever `uniprot_poc[i] == uniprot_mol[j]`
or `lig_smiles[j]` is in pocket $i$'s known set, and `masked_fill` those cells to the dtype minimum so
they vanish from every softmax. On this sharp, low-hit-rate distribution false negatives are the one
signal-corrupting mistake I cannot afford.

One more thing this scaffold default does *not* do, which I want to name because it sets up the next
rung. The harness hands me a third tower — the ESM-2 protein-sequence embedding, `prot_emb` — and a
fourth piece of metadata, the per-ligand pIC50 in `act_list`. The vanilla CLIP fill ignores both: the
loss is pocket↔molecule only, the protein tower is defined and projected but never enters
`compute_loss`, and nothing in the objective distinguishes a strong binder from a merely-okay one — two
actives of the same target both get pulled toward it with no preference for the stronger. That is fine
for a floor; it is exactly what makes this the weakest rung. The scoring at evaluation is the matching
quantity: embed the target's pockets and every molecule, form `pocket_reps @ mol_reps.T`, and take the
max over the target's pocket conformers — a molecule that binds *any* conformation should rank high — then
sort. (The full scaffold module is in the answer.)

So at rung one my fill is the plainest contrastive scorer the scaffold admits: three NonLinearHeads,
L2-normalized Euclidean embeddings, a detached log-scale, the assay-grouped symmetric in-batch softmax
with false-negative masking and $\sqrt{L_i}$ normalization, max-over-pockets scoring — and the protein
tower and the activity numbers left on the table.

Now reason about what this floor must do, because that is the point of running it. The contrastive
objective aligns binders with their target in cosine space, so it should already concentrate actives
near the top far better than chance — this is a real, scalable scorer, not a null model, and I expect
solid AUROC on all three benchmarks and respectable EF on DUD-E, where the decoys are property-matched
ZINC molecules that a good binder/non-binder boundary separates cleanly. But two weaknesses are baked
into this fill by construction. One: with the protein-sequence view dropped, the only target signal is a
single sampled pocket conformation; where that pocket is noisy or ambiguous, I have no complementary,
structure-free view to fall back on. Two: nothing in the loss is *top-heavy*. The objective treats every
active as an equal positive, but the metric rewards getting the *strongest* binders to the very top — so
on the benchmark where realistic, graded actives crowd together and the top few ranks decide the score,
this fill should leave enrichment on the table. LIT-PCBA is exactly that benchmark — confirmed actives
versus confirmed inactives, no artificial-decoy crutch — so I expect vanilla CLIP to look weakest there:
decent AUROC but soft BEDROC and EF, because it ranks binders above decoys without ranking *strong*
binders above *weak* ones. Whatever the precise split across DUD-E, DEKOIS, and LIT-PCBA, the diagnosis
is already pointed at the next rung: I have an *ordering* problem on top of a *separation* problem, and
the unused activity numbers and the unused sequence tower are sitting right there to fix it.
