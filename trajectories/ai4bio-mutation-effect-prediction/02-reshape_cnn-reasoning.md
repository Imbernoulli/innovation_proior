The ridge floor told me exactly where a linear head breaks, and it told me in numbers. On BLAT_ECOLX
it landed 0.737 and on RASH_HUMAN 0.681 — respectable, both assays where the mutation-induced shift
maps roughly monotonically to fitness, so a single hyperplane through delta-space reads off most of
the rank order. But on ESTA_BACSU the Spearman came back *negative*, −0.107: the linear head did
worse than random, anticorrelated with the truth. That is not noise; it is the failure I flagged
going in. ESTA_BACSU is the stability assay, and stability is the textbook threshold phenotype — most
substitutions are tolerated and a minority push the protein over a folding cliff into the dead pile,
so fitness is a sharply curved, near-discontinuous function of the embedding shift. A linear map
cannot bend; forced to fit a cliff with a plane, it latches onto whatever global axis minimizes
squared error and ends up ranking mutations in roughly the wrong order. So the diagnosis is sharp and
it is *representational*, not a tuning problem: BLAT and RASH are approximately linear in the delta
and the linear head is fine there, but ESTA needs a head that can curve, that can carve delta-space
into regions and put the stability cliff where it belongs. The −0.107 is the signature of a
hyperplane trying to be a cliff. So the next head must be nonlinear, and it must keep the gains on the
two assays the linear head already handled while rescuing the one it inverted.

The plainest nonlinear head is just a wider MLP — `Linear → nonlinearity → Linear` over the delta —
and I will get there. But before I reach for the flat MLP let me ask whether the 1280-d (and the
2560-d, if I use both inputs) feature vector has any *structure* a head could exploit beyond pure
fully-connected mixing, because if it does, a structured head with weight-sharing could be a stronger
inductive bias than a dense layer that treats all coordinates as a-priori unrelated. That is the
hypothesis this rung tests: does imposing structured weight-sharing over the embedding features beat
a flat dense head, on the same inputs?

Here I have to be honest about what structure is and is not present, because it changes the whole
design and the README is explicit that a baseline named after a paper can be a very different object
in this harness. A convolutional head's natural home is per-residue token embeddings — a `[L, 1280]`
tensor where the length axis is the protein sequence and a 1D convolution slides over neighboring
residues with real spatial meaning. The harness does *not* expose that. It only stores the
**mean-pooled** ESM-2 vector, `[1280]` per mutant; the per-residue tokens were averaged away upstream
and I cannot get them back. So a true sequence-convolution is off the table — there is no sequence
axis to convolve over. Whatever "CNN" I build cannot be the paper-faithful Conv1D-over-residues
model; it has to operate on the flat pooled vector. I want to state that limitation up front so I do
not accidentally import the wrong story: there is no spatial or sequence locality along the 1280
embedding coordinates, so any convolution I run over them is *not* exploiting real adjacency. What it
*is* doing is enforcing weight-sharing across blocks of embedding dimensions — the same small filter
applied to many chunks of the feature vector — which is a different, weaker, but still real inductive
bias than a fully-connected layer that gives every coordinate its own independent weight. The
question is whether that weight-sharing bias helps or hurts versus a dense head, and the only way to
find out is to build it and measure it against the ridge floor.

So the construction. I take *both* inputs this time — concatenate `[embedding, delta_embedding]` into
a `[B, 2560]` vector. The ridge head used the delta alone, which was the right call for a linear
readout because the raw embedding's constant identity is pure noise to a linear map. But a nonlinear
head with normalization can use the raw embedding to *condition* on which region of fitness-landscape
this protein lives in — the absolute representation tells it "this is a stability assay near a folding
boundary" in a way the delta alone cannot — so feeding both is worth trying for a head that can
actually mix them nonlinearly. To give the convolution something to slide over, I reshape the 2560-d
vector into a fake `(channels=64, length=40)` "image": project `2560 → 64·40` with a linear layer,
reshape, and treat the 64 as channels and the 40 as a pseudo-spatial axis. I emphasize again — there
is nothing physical along that length-40 axis; the reshape just groups the embedding coordinates into
40 blocks of 64 and lets 1D convolutions share weights across those blocks. That is the entire
inductive bias on offer, and it is an honest one to test.

Now, the moment I stack convolutional layers I inherit a real problem that has nothing to do with
proteins and everything to do with deep stacks of nonlinear layers: they are hard to optimize, and
naively deepening them makes training *worse*, not from overfitting but from an optimization
conditioning failure. The clean way to see it: if a stack of layers would be best off doing nothing
to its input — passing it through as the identity — then a stack of conv-norm-nonlinearity layers
finds the identity surprisingly hard to learn, because weight decay and small-init optimization pull
the weights toward *zero*, and a block of zero weights computes the *zero* map (it annihilates its
input), not the identity. So precisely the mapping I would most like a redundant layer to fall into —
"refine the input a little, otherwise leave it alone" — is one of the harder things for it to express,
and it fights the regularizer. The fix is to change *what the layers are asked to learn*. Instead of
asking a block to produce its full output mapping `H(x)` from scratch, hand it `x` for free and ask
only for the correction `F(x) = H(x) − x`, then recover the output by adding the input back through a
parameter-free shortcut: `y = F(x) + x`. Both forms can represent the same functions — they differ
by the fixed term `x` — so capacity is unchanged; only the conditioning changes. Now "leave the input
alone" means `F = 0`, the easiest target there is and exactly where weight decay already pushes, and
"refine a little" means a small `F` referenced to `x`. That residual reparameterization is what lets
me stack several conv blocks without the deeper stack degrading, and it is one line per block:
`out += residual`.

A couple of wiring details the residual form forces, and I want them right. The nonlinearity must not
be the last thing on the residual branch: if I rectify `F` before adding `x`, then `F ≥ 0` and the
block can only ever push features up, never pull them down — a crippling one-sided restriction. So the
branch ends on conv-then-norm with no final activation, and the nonlinearity goes *after* the add. I
use GELU rather than ReLU here as the smooth activation, and BatchNorm after each conv to keep the
per-channel scale stable across the stack — normalization is what made deep plain stacks converge in
the first place, so it earns its place. Each block is therefore: conv (padded to preserve the
length-40 axis), BatchNorm, GELU, dropout, then add the input back. I use three blocks with kernel
sizes 3, 5, 7 so the shared filters cover progressively wider spans of the embedding-block axis —
larger receptive fields over the pseudo-spatial dimension, the only knob the fake-image layout gives
me. After the conv stack I global-average-pool over the length-40 axis (collapsing the pseudo-spatial
dimension to one 64-vector per mutant, a parameter-free, position-agnostic summary that is the right
move when the positions carry no real meaning), then a small two-layer head `Linear(64,128) → GELU →
dropout → Linear(128,1)` produces the scalar fitness prediction.

So the head is: project `[embedding, delta_embedding]` to a `(64, 40)` grid, run three residual Conv1d
blocks with BatchNorm/GELU/dropout over the embedding-channel axis, global-average-pool, and read out
with a two-layer MLP. The full scaffold module is in the answer. The residual shortcut is the load-
bearing piece for trainability; the convolutional weight-sharing is the bias under test; the both-
inputs concatenation and global pooling are the choices that let a nonlinear head condition on protein
identity and stay position-agnostic over a meaningless axis.

Now the falsifiable expectations against the ridge numbers, and they are the whole reason to run this.
The single thing I most need to see is ESTA_BACSU climbing out of the negative. The linear head's
−0.107 was a hyperplane failing to bend around a stability cliff; a nonlinear head — convolutional or
not — can in principle carve delta-space, so if convolutional weight-sharing is a useful bias here I
expect ESTA to jump from −0.107 to a solidly positive Spearman, somewhere in the low-to-mid 0.6s if
it works as well as the other assays. That is the make-or-break test: if ESTA does not recover, the
nonlinearity is not reaching the cliff and the head design is wrong. On BLAT and RASH, where ridge
was already strong (0.737 and 0.681), I expect the nonlinear head to *hold or modestly improve* — it
can fit the same approximately-linear signal and add a little curvature, so I would expect BLAT to
edge up toward the mid-0.8s and RASH into the high-0.7s. What I am genuinely unsure about is whether
the convolutional weight-sharing helps *relative to a plain dense MLP* on the same inputs. The fake-
image reshape imposes a bias that has no physical justification — there is no real adjacency along the
embedding axis — so it is entirely possible that a flat dense head, which lets every coordinate speak
independently, extracts the nonlinear signal at least as well and with less architectural overhead.
If that is what the next comparison shows — a plain MLP matching or beating this reshape-CNN across
all three assays — then the lesson is that the structured weight-sharing was a bias the data did not
want, and the right head is the dense nonlinear one. Either way, this rung's job is to confirm that a
nonlinear head rescues ESTA from the linear floor's −0.107 while holding the BLAT/RASH gains, and to
put a convolutional inductive bias on the board so the next step can decide whether structure or plain
density is the better nonlinear head.
