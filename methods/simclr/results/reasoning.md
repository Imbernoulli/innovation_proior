Let me start from the thing that actually hurts. Labels are expensive and almost nothing in the world is labeled, yet a supervised ResNet trained on a million human annotations learns features that transfer everywhere. I want those features without the labels. And I have a concrete number to chase, because the field has already agreed on a ruler: take whatever encoder I learn, freeze it stone-cold, train a single linear classifier on top, and read off its test accuracy. That number measures how *linearly separable* the representation is — whether one hyperplane per class can already carve the feature space, which only happens if the network has done the real work of pulling the semantics apart. So my whole job reduces to: drive the linear-probe accuracy of a frozen, label-free encoder up toward a supervised net of the same architecture. Not approximately measure it — that is the target.

The methods currently sitting at the top of the leaderboard all bother me, because every one of them buys its accuracy with a contraption. InstDisc drags a memory bank around — it stores a feature vector for every single instance in the dataset and refreshes them slot by slot as it trains. MoCo drags a queue of past minibatch features plus a second, momentum-updated encoder to keep that queue self-consistent. CPC chops the image into a deterministic grid of patches, runs a PixelCNN on top as a context aggregator, and only ever lets the encoder see small patches. DIM and AMDIM go further and *rebuild the network itself* — they strangle the receptive field with 1×1 convolutions so that "predict the global from the local" becomes possible. Each contraption brings its own hyperparameters, its own failure modes, its own engineering, and — worst of all — it welds the idea of contrastive learning to the crutch that implements it. None of them lets me take the strongest off-the-shelf ResNet and just *use* it. So I am going to keep asking two questions and refuse to answer either with a contraption: where does the predictive task come from, and where do the negatives come from. I want both answers to leave the architecture untouched and to need no external storage.

Before I derive anything I want to walk the ancestry, because I am going to stand on it, and I need to see exactly where each one cracks.

The oldest instinct is Becker & Hinton (1992): the two views of one input should agree, and you learn by maximizing the mutual information between the outputs of spatially adjacent patches. That is the seed of the whole line — *agreement under transformation*. But it predates deep nets, modern augmentation, and the softmax-over-many-negatives machinery; it is the right instinct at a scale that cannot test it.

Hadsell, Chopra & LeCun (2006) made "agree" trainable with the contrastive loss / DrLIM: pull a positive pair together, push a negative pair apart until it clears a margin, like a system of springs. This nails the basic shape — positive versus negative — but it is a *pairwise margin* loss. There is no temperature, no softmax over many negatives at once, no notion of harvesting negatives from a batch at scale.

Then comes instance discrimination, and the most naive version is Dosovitskiy et al. (2014): treat each image, together with all its augmentations, as its own class, and train a classifier to say which exemplar this patch came from. Let me follow that thread, because it immediately walks into a wall. Parametric classification means the softmax has one learnable weight vector w_j per instance. With a million images that is a million-output classifier head; the weight matrix grows linearly with the dataset, which is hopeless at ImageNet scale, and worse, those per-instance weights are tied to specific images — they do not generalize to anything unseen. The parametric route hits a scaling wall I cannot climb.

Wu et al. (2018), InstDisc, removes the wall with a clean move: if the per-class weight w_j is the problem, throw it out and let the *feature itself* play the role of its own class prototype. Normalize every feature to the unit sphere and write a non-parametric softmax,

  P(i | v) = exp(vᵢᵀv / τ) / Σ_j exp(vⱼᵀv / τ),   ‖v‖ = 1.

No learnable classifier head anymore; "which class" becomes "which stored feature is closest." This hands me two parts I want to keep verbatim: a temperature τ, and ℓ2-normalization. But the denominator sums over *all* instances, which is uncomputable, so InstDisc approximates it with noise-contrastive estimation and parks every instance's feature in a memory bank, overwriting v_i each time it is sampled and estimating the partition function by Monte Carlo. And there is the crack: the bank entry v_j was written by an *older* encoder than the one producing my current query v. Query and key live at different moments of the encoder's training. That staleness is a defect I must route around, and the bank is yet another contraption with its own knobs.

Now the real loss ancestor, CPC / InfoNCE (Oord et al. 2018). I am going to derive it from scratch rather than import it, because if I am going to lean my entire loss on it I will not build on a result I have not re-derived myself, and I do not yet know what shape of loss it forces.

The setup: an autoregressive context c predicts a future latent. You are handed a set X = {x_1, …, x_N} in which exactly one element is the true positive — drawn from the conditional p(x | c) — and the other N−1 are negatives drawn from the marginal p(x). The loss is

  L_N = − E_X [ log ( f(x_pos, c) / Σ_{x_j ∈ X} f(x_j, c) ) ],

with f a learnable score function. This is just the categorical cross-entropy of "point to the positive in the set." The first thing I need to know is what the *optimal* f looks like, because that will tell me the correct shape of my loss.

Treat it as Bayesian inference over which index is the positive. Given the set X and the context c, the hypothesis "index i is the positive" has likelihood proportional to: x_i came from the conditional, every other x_j came from the marginal, i.e. p(x_i | c) · Π_{j≠i} p(x_j). Normalize over all the hypotheses:

  P(pos = i | X, c) = [ p(x_i | c) Π_{j≠i} p(x_j) ] / Σ_k [ p(x_k | c) Π_{j≠k} p(x_j) ].

Now divide numerator and denominator by the common factor Π_{all} p(x). In the numerator, Π_{j≠i} p(x_j) = (Π_{all} p) / p(x_i), so dividing through leaves p(x_i | c) / p(x_i); the Π_{all} p cancels everywhere, term by term:

  P(pos = i | X, c) = [ p(x_i | c) / p(x_i) ] / Σ_k [ p(x_k | c) / p(x_k) ].

Lay that posterior next to the loss's softmax shape f(x_i, c) / Σ_k f(x_k, c). They coincide exactly when the score f is proportional to the **density ratio** p(x | c) / p(x) (any factor depending only on c washes out of the softmax, so it is fixed only up to that). So the f that minimizes L_N is the density ratio, and the loss shape that has that optimum is the log-sum-exp softmax — not a margin, not a hinge, not a pairwise sigmoid. The softmax form is not a stylistic choice; it is the form whose optimum is the quantity that distinguishes "drawn from the conditional given c" from "drawn at random." That is my first reason to take a softmax loss over a triplet loss, and I have not even gotten to the gradient yet.

Next, the mutual-information relationship, because the entire creed of "more negatives is better" rests on it and I refuse to chant it without proof. The tempting shortcut is to say that each negative ratio has expectation 1 and then replace the whole negative sum by N−1 inside the log. That is useful intuition, but it is not a proof; the logarithm is exactly where the randomness of the denominator matters. I need the bound in the classification form.

Keep the optimal score f ∝ p(x|c)/p(x), write r_j = p(x_j|c)/p(x_j), and choose index 1 as the positive without loss of generality. Then

  L_N^* = E log [ (Σ_{j=1}^N r_j) / r_1 ],

so

  log N − L_N^*
    = E log [ N r_1 / Σ_{j=1}^N r_j ]
    = E log r_1 − E log [ (1/N)Σ_{j=1}^N r_j ].

The first term is exactly the mutual information:

  E_{p(c)p(x_1|c)} log r_1
    = E log [ p(x_1|c) / p(x_1) ]
    = I(x;c).

Now the second term has to be nonnegative. Let

  A(x_{1:N},c) = (1/N)Σ_{j=1}^N r_j.

This A is not an arbitrary average; it is the density ratio between two distributions over the candidate set. The numerator distribution first picks a positive index uniformly, draws that one candidate from p(x|c), and draws the rest from p(x):

  m(x_{1:N}|c) = (1/N)Σ_i p(x_i|c) Π_{j≠i} p(x_j).

The denominator distribution draws every candidate from the marginal:

  q(x_{1:N}) = Π_j p(x_j).

Dividing m by q gives exactly A. Also, log A is symmetric in the N candidates, so its expectation when "candidate 1 is the positive" is the same as its expectation under the uniform mixture m. Therefore

  E log A
    = E_c KL( m(x_{1:N}|c) || q(x_{1:N}) )
    ≥ 0.

So

  log N − L_N^* = I(x;c) − E log A ≤ I(x;c).

And if my learned score is not optimal, its cross-entropy is only worse: L_N ≥ L_N^*. Therefore the usable bound is

  I(x;c) ≥ log N − L_N.

Two things in that derivation made me uneasy enough to want a number: the claim that the second term E log A is a genuine KL and so nonnegative, and the claim that raising N actually tightens the bound rather than just inflating log N to cancel an equally-growing L_N^*. Symmetry arguments are exactly where I fool myself, so let me build the smallest model where I can compute every quantity exactly and watch the inequality. Take a binary channel: c ∈ {0,1} uniform, x ∈ {0,1}, with p(x=1|c=1)=p(x=0|c=0)=0.9. Then the marginal p(x) is uniform [0.5, 0.5], and the true mutual information is

  I(x;c) = Σ_{c,x} p(c)p(x|c) log[p(x|c)/p(x)] = 0.368 nats

(I worked this out: each of the four terms is 0.5·0.9·log(1.8) or 0.5·0.1·log(0.2), and they sum to 0.368.) Now the optimal InfoNCE loss with N candidates — one positive drawn from p(x|c), the rest from p(x), scored by the exact density ratio r(x,c)=p(x|c)/p(x) — I estimate by sampling. The numbers come back:

  N=2:  L_N^* = 0.509,  log N − L_N^* = 0.184
  N=4:  L_N^* = 1.107,  log N − L_N^* = 0.280
  N=8:  L_N^* = 1.753,  log N − L_N^* = 0.326

So the bound holds at every N (0.184, 0.280, 0.326 all sit under I = 0.368), and — the part I actually doubted — it tightens monotonically toward the true MI as N grows, rather than stalling. The gap I = 0.368 minus the bound is the E log A term, and it shrinks from 0.184 to 0.042 as N goes 2→8, which is the KL going to zero, confirming numerically that it was nonnegative and that more candidates is what closes it. That is the slogan "more negatives is better" turned into three numbers I can see move. So my batches want to be large — not because the literature says so, but because this monotone curve says so. (I am aware of the caution from the Tschannen et al. (2019) direction — that the empirical success might be the loss *form* rather than mutual information as such. Fine: either way the two operational conclusions are identical — use the softmax form, and feed it many negatives — so I will not litigate the philosophy of mutual information; I will just take both.)

Now I can see what the bank and the queue really are: both are compensating for the same shortage — a single batch does not contain enough negatives. InstDisc routes around "the denominator sums over everything" by storing everything (price: staleness). MoCo routes around "one batch is too small" by accumulating a cross-step queue (price: queued keys come from an old encoder, which then needs a momentum encoder θ_k ← m·θ_k + (1−m)·θ_q to be kept *roughly* consistent — staleness made mild, not removed). Wait. What if I just make the batch genuinely huge? At batch size N = 8192, a single positive pair sees 2(N−1) = 16382 negatives, every one of them produced by *this* encoder at *this* step, all fresh, all back-proppable end to end. No bank. No queue. No momentum encoder. No NCE approximation of the partition function — the denominator simply *is* this one big batch. Where do the negatives come from? From the batch. The N-pair loss (Sohn 2016) and the whole in-batch-negatives tradition were already doing this; I can make that idea the whole recipe at ImageNet scale. The consistency-versus-freshness trade-off collapses one-sidedly: bank and queue buy quantity by paying in freshness; a large batch buys both and moves the cost to engineering — I have to *afford* a batch of thousands and *stabilize* it. I accept that trade. The second question has its answer.

Now the question I am more excited about: where does the predictive task come from. CPC manufactures "predict a future patch" with deterministic patching plus a PixelCNN context net; DIM/AMDIM manufacture "predict the global from the local" by strangling the receptive field. Both weld the task into the architecture, which is exactly why they cannot use a strong standard backbone. I keep staring at CPC's picture of global patches, local patches, adjacent patches, and the question that nags me is: what is the *essence* of those tasks? It is "look at different regions and scales of the same image, and force their representations to agree." So why am I editing the network's receptive field to create that? I can just *crop*. Take two random crops of one image: maybe one is large and one is small — that is global→local prediction, for free. Maybe the two are adjacent and barely overlap — that is adjacent-view prediction, for free. One operation, random cropping, subsumes both of the tasks that the prior work had to bake into architecture. The task does not have to come from the architecture; it can come from data augmentation. And the instant I say that, the predictive task is *decoupled* from the encoder — I can drop in any standard ResNet, unchanged, exactly the thing I wanted.

The framework now snaps together in outline. For each image x, draw two transformations t, t′ independently from an augmentation family 𝒯, get two views x̃_i = t(x) and x̃_j = t′(x) — that is one positive pair. Push both through the *same* shared encoder f to get representations h = f(x̃) (the post-pool vector). Then put InstDisc's normalization-and-temperature on top of CPC's softmax form: ℓ2-normalize and score by cosine similarity sim(u,v) = uᵀv/(‖u‖‖v‖) divided by τ; the positive for view i is its counterpart view j, and the negatives are every other view in the batch. For a positive pair (i,j),

  ℓ_{i,j} = − log [ exp(sim(z_i, z_j)/τ) / Σ_{k=1}^{2N} 1[k≠i] exp(sim(z_i, z_k)/τ) ].

(I am writing z rather than h here because I am about to discover that there should be a small network between h and the loss — but let me earn that, not assume it.) The indicator 1[k≠i] kicks out the k=i term, which is the view's similarity with itself — identically 1/τ after normalization, and it would otherwise dominate the denominator and teach nothing. And I compute this symmetrically, both (i,j) and (j,i), so the gradient flows from both view directions; computing only one direction throws away half the signal. This is a normalized, temperature-scaled cross-entropy — call it NT-Xent. It is not a new loss; Sohn (2016), Wu (2018), Oord (2018) all used this shape. What is new is the surrounding minimalism: augmentation defines the task, a large batch supplies the negatives, a standard ResNet is the encoder, and nothing else.

I will not take any of these pieces on faith, so let me interrogate each one and let it either justify itself or fall.

What are τ and the ℓ2-normalization actually *doing*? I will get the answer by differentiating NT-Xent with respect to an already-normalized anchor u, with positive v⁺ and negative set {v⁻}. Write the single-anchor negative-log-likelihood,

  ℓ = − uᵀv⁺/τ + log Σ_{v ∈ {v⁺}∪{v⁻}} exp(uᵀv/τ).

Let Z(u) = Σ_v exp(uᵀv/τ). Differentiate: the first term gives −v⁺/τ; the log-sum-exp gives (1/τ) Σ_v [exp(uᵀv/τ)/Z(u)] v. Collecting,

  ∂ℓ/∂u = (1/τ) [ Σ_v (exp(uᵀv/τ)/Z(u)) v − v⁺ ]
        = (1/τ) [ (p⁺ − 1) v⁺ + Σ_{v⁻} p⁻ v⁻ ]
        = − (1/τ)(1 − p⁺) v⁺ + (1/τ) Σ_{v⁻} p⁻ v⁻,

where p⁺ = exp(uᵀv⁺/τ)/Z(u) and p⁻ = exp(uᵀv⁻/τ)/Z(u) are exactly the softmax probabilities the loss assigns to each candidate. Now stare at the negative part. The gradient contains +(p⁻/τ)v⁻ for each negative, so the descent step contains −(p⁻/τ)v⁻: every negative direction is pushed away with weight equal to its *own* softmax probability p⁻ = exp(uᵀv⁻/τ)/Z(u). The more similar a negative is to the anchor (larger uᵀv⁻), the larger its p⁻, the harder it is shoved.

Before I call that "automatic hard-negative mining," I want to make sure I actually differentiated correctly and that the weighting really lands on the hard one — my algebra above could have a sign or a missing-Z mistake that this story would happily paper over. So a concrete instance, in 2D on the unit circle. Anchor u = (1,0). Positive v⁺ = (0.95, 0.30) normalized, cosine 0.949 to u. Two negatives: an easy one v⁻_easy = (−1, 0.2) normalized, cosine −0.981, and a hard one v⁻_hard = (0.8, 0.6) normalized, cosine 0.800. Temperature τ = 0.2. The softmax over the three candidates comes out p⁺ = 0.678, p_easy = 0.00002, p_hard = 0.322. Plugging into the analytic ∂ℓ/∂u above gives (−0.240, 0.457); a finite-difference gradient of −log-softmax at the same u gives (−0.240, 0.457) to four places — they match, so the derivation is right and not just plausible-looking. And the push: the hard negative carries weight 0.322 while the easy one carries 0.00002, a ratio of about 16000. The softmax has, with no instruction from me, dumped essentially all of the negative gradient onto the negative that is geometrically close to the anchor and ignored the one that is already far. That is hard-negative mining — *automatic*, hidden inside the softmax denominator, costing me zero lines of mining code.

Compare the margin triplet, max(0, uᵀv⁻ − uᵀv⁺ + m): when the margin is violated its gradient with respect to u is v⁻ − v⁺, so the descent direction is v⁺ − v⁻; when the margin is satisfied the gradient is 0. Every margin-violating negative is treated identically; there is no weighting by hardness anywhere in it. That is precisely why FaceNet *has* to bolt on semi-hard mining — hand-pick the negatives that sit inside the margin but farther than the positive — or training either drowns in trivial easy negatives or is yanked around by a handful of impossibly hard ones. NT-Logistic, the word2vec-style per-pair sigmoid loss, has the same defect: each pair gets its own σ(·) gradient with no *relative* weighting across negatives. So NT-Xent's softmax internalizes the one thing the alternatives need an external heuristic for. This is the *second* hard reason — now from the gradient — to take the softmax form over margin or logistic. Not taste; mechanism.

That makes τ legible. τ is the sharpness knob on this automatic weighting, and I can read it off the same three-vector example by sweeping τ and watching how the negative gradient splits between the easy and hard negative. Renormalizing the push among the two negatives only: at τ=1.0 the hard negative takes 0.856 of the negative-push; at τ=0.2 it takes 1.000; at τ=0.05 still 1.000. So shrinking τ does concentrate the gradient onto the hardest negative, and past a point it saturates entirely on it — which is exactly the failure mode I should worry about: pushed too far, one or two hardest negatives hijack the whole gradient, distort the geometry, and destabilize training. Make τ large instead and p⁻ flattens toward uniform — the hard-negative signal washes out and the loss degenerates into pushing equally on everything, which is weak. So τ wants an intermediate value, and the right value depends on how many and how hard the negatives are — tighter (smaller) on a big, hard ImageNet batch, looser on small CIFAR.

And ℓ2-normalization? Suppose I drop it, leaving uᵀv unbounded. In the softmax that means the network can manufacture arbitrarily large logit gaps simply by *growing the magnitude* of its vectors rather than by aligning their *directions* — it games the loss through scale. The likely outcome is perverse: the contrastive-task training accuracy goes *up* (magnitude is an easy lever) while the linear-evaluation accuracy goes *down*, because direction — the semantic content — was never properly learned. Normalizing everything to the unit sphere caps uᵀv ∈ [−1, 1], which (i) gives τ a clean, interpretable scaling instead of leaving it to absorb arbitrary magnitudes, and (ii) places the entire hardness-weighting on directional similarity, which is the thing I actually care about. Normalization and τ are a matched pair: without normalization τ is meaningless; with it, τ is a pure sharpness control. I can't settle this from the chair — the magnitude-gaming failure is an empirical claim about what the optimizer actually does — so I would sweep τ with and without normalization and watch whether contrastive accuracy and linear accuracy diverge. But the mechanism is concrete enough that I expect the normalized version to win, and I will go in with normalization on.

Now that little network between h and the loss. The most direct thing is to have *no* network and compute the loss straight on h. AMDIM hinted at putting a nonlinear transform before the loss. So I ask two questions: why have it at all, and — sharper — if I do add a head g, does downstream use h or z = g(h)? Intuition screams z, because z is the layer the loss actually optimizes. Let me follow the loss's semantics and watch that intuition break.

What is NT-Xent forcing z to do? It forces z_i and z_j — the two augmented views of one image — to agree, i.e. it forces z to be *invariant* to the augmentation. And what is the augmentation? Random cropping changes an object's position and scale; color distortion changes its color; (the blur I will add) removes high-frequency detail; a rotation would change orientation. So if I press the loss directly onto h and force *h* to be invariant, I am training h to *throw away* exactly color, orientation, position, and high-frequency content — and those are precisely the cues many downstream tasks need (telling a flower's species by its color, a car's pose by its orientation). That is a real conflict: the invariance the contrastive loss demands and the information content downstream wants are fighting over h. And forcing the representation to be z would be even worse — z is the most-compressed, most-invariant layer of all, the one squeezed hardest. So my intuition was backwards.

Back up and reconsider what g is *for*. Insert a nonlinear g between h and the loss. Now the "be invariant to augmentation" requirement can be satisfied *by g* — g learns to collapse the directions that vary under augmentation (color, orientation, high-frequency) into invariants, so that z = g(h) is invariant and the loss is happy — while h, sitting *upstream* of g, is under no such obligation and can keep all that information. g acts as a buffer that absorbs exactly the information the loss demands be discarded, taking the hit so h does not have to. The correct recipe falls out: train with the loss on z = g(h), then *throw g away* and use h downstream. Use the layer *before* the optimized one — counter-intuitive, and load-bearing.

Why nonlinear g rather than a linear projection? A linear g is a single matrix W; it can only carve out invariance along a linear subspace — collapse some directions to near-zero, low-rank style — and its capacity to absorb the augmentation-induced variation is limited. A one-hidden-layer nonlinear g(h) = W⁽²⁾ σ(W⁽¹⁾h) with a ReLU has far more freedom to build a nonlinear invariant map, and so it can take over the invariance burden more completely, leaving h richer. I expect a strict ordering: nonlinear g better than linear g better than no g — even one extra layer of freedom to soak up invariance beats making h bear it directly. The dimension of z can be small (say 128); z is a disposable interface to the loss, discarded after training, so there is no reason to make it wide. I can design a clean test of this whole story: take the trained model and train a small MLP on h, and separately on z, to *predict which augmentation was applied* (rotation angle, grayscale-or-not, noise, Sobel). If my reasoning is right, h recovers those attributes well and z recovers them poorly, because z was trained to be blind to them. I would bet h wins, and wins by a lot.

Augmentation itself has to earn its place, since "the task comes from augmentation" is the pivot of the entire framework. Is a single augmentation enough? I suspect not, but I have to see *why*, not just assert it.

Imagine cropping only. Two crops of the same image share almost the same color distribution — a photo of grass gives a green-dominated pixel histogram no matter where you crop it. So the network can solve the contrastive task by simply comparing the *color histograms* of the two views: different images have different histograms, so matching crops of the same image by color statistics alone is enough to identify the positive. It will take that shortcut: the contrastive training accuracy can climb toward perfect while what it learned is color statistics, not semantics. The linear probe will then crater, because a color histogram is a poor feature for "cat versus dog." So a contrastive task that can be solved is not necessarily a good one; whatever shortcut the task leaves open, the network will take. A perfectly-solvable task can teach nothing.

How do I shut the shortcut? Destroy the color cue — apply color distortion after the crop (random brightness, contrast, saturation, hue, plus a probability of dropping to grayscale entirely). Now the two views of one image have independently scrambled colors, the histograms no longer line up, and the network is forced to find what is genuinely invariant across the two views: shape, texture, semantics. So it is not that some single augmentation is magic; it is that the *composition* makes the task hard and shortcut-free. Crop + color distortion is the essential pair, and I expect it to stand out above all others: crop alone nearly solves the task (via color) yet learns little; color distortion alone has no spatial task; only stacked do they produce a task that is both hard and meaningful. A dash of Gaussian blur on top is worth adding as a high-frequency shortcut blocker. Color distortion in code is just: jitter brightness/contrast/saturation/hue, then drop to grayscale with small probability, with the intensity controlled by a single strength parameter s.

A counter-intuitive corollary falls out of this: contrastive learning wants *stronger* augmentation than supervised learning. Supervised learning has labels as a safety net, and overly aggressive color distortion can erase discriminative color cues and *cost* accuracy. Contrastive learning has no labels — augmentation is the *only* thing defining the task and blocking shortcuts — so it benefits from heavier augmentation. I want to run exactly that contrast: same augmentations, supervised versus contrastive, and see who gains from turning the strength up. And it suggests that even a policy like AutoAugment, painstakingly searched for supervised accuracy, may not be best for contrastive learning, because it was optimizing a different objective entirely.

Now two engineering landmines, because the large-batch path derails on both if I am not careful.

Mine one: BatchNorm. A standard ResNet is full of BN, and under data-parallel training BN's mean and variance are usually computed *per device* (local BN). In my setup the two views of a positive pair frequently land on the same device — and then that device's BN statistics encode "these particular samples are here together." The network can learn to *read* that leaked local statistic to identify the positive, cheating the contrastive task while improving the representation not at all. It is the same species of shortcut as the color histogram: a way to solve the task without learning anything. The fix: aggregate BN's mean and variance across *all* devices (global BN), so the statistics are global and carry no "who is co-located with whom" information. MoCo dodges this by shuffling samples across devices; CPC v2 swaps BN for layer norm. I pick global BN because it is the smallest change to a standard ResNet, which is the whole spirit of "do not touch the architecture."

Mine two: optimization stability at huge batch. Plain SGD/momentum with linear LR scaling (the Goyal et al. 2017 recipe) becomes unstable at batch sizes in the thousands — across layers the gradient-norm-to-weight-norm ratio varies wildly, so a single global learning rate either blows some layers up or starves others. LARS fixes exactly this by scaling each layer's update by that layer's ‖w‖/‖grad‖ ratio — layer-wise adaptive rate scaling, built for large-batch ImageNet. Pair it with a linear warmup (ramp the LR up from zero over the first ~10 epochs so a large LR does not detonate the random initialization) and cosine decay (anneal smoothly to zero). One more detail: exclude BN and bias parameters from weight decay — decaying normalization scales and biases has no regularization meaning and only disturbs them.

Let me run the whole causal chain once in my head and check that every link is pushed out by the one before it, with no choice arriving from nowhere. The ruler is linear separability of frozen features, so I take contrastive learning, the line already ahead. The InfoNCE Bayes derivation pins the loss shape to "softmax driving the score toward the density ratio," not a margin; the mutual-information bound I ≥ log N − L_N converts "more negatives is better" from a slogan into an inequality with ceiling log N. Wanting many negatives without staleness rules out bank and queue (which trade freshness for quantity) and rules *in* a large batch of fresh, back-proppable in-batch negatives. Affording a large batch forces global BN (to plug the BN co-location leak) and LARS + warmup + cosine (to stabilize the optimizer). Wanting the task without touching the architecture turns random cropping into a generator of global→local and adjacent-view tasks, so a standard ResNet drops straight in. Cropping leaks a color-histogram shortcut, so composing color distortion shuts it — and that yields the corollary that contrastive learning wants stronger augmentation than supervised. The NT-Xent descent direction grows its own automatic hard-negative weighting (each negative weighted by its softmax probability), so no mining is needed, and the same derivation explains τ (sharpness) and ℓ2-normalization (confine to the sphere so τ means something and magnitude cannot be gamed). The loss trains z to be invariant and would therefore strip color and orientation from the representation, so I insert a nonlinear g as a buffer, optimize z = g(h), and keep h before g. Every link is forced by a concrete pain point or a single line of math.

Now to drop this onto real code. The core is NT-Xent, and there is a clean implementation trick: never materialize negatives explicitly — phrase the whole thing as one cross-entropy classification where each anchor's "correct class" is its counterpart view, and mask out self-similarity (the k=i term) by subtracting a large constant from that logit. Across devices, gather the other view's features from all cores so each anchor sees the full set of in-batch negatives; the one-hot labels then point at the positive's index after the concatenation.

This indexing is exactly the kind of thing I get wrong on the first try, so let me hand-trace it on a single device with N=2 images before I trust it. Stack the four normalized rows as [a1, a2, b1, b2] — first two are view-a of images 1,2, last two are view-b. Split into hidden1=[a1,a2], hidden2=[b1,b2]. The aa-block logits are hidden1·hidden1ᵀ/τ; subtracting masks·LARGE_NUM (masks = I₂) sends the diagonal to ≈ −10⁹, so a row's similarity-to-itself (which after normalization is the constant 1/τ and would otherwise swamp the denominator) is removed — I checked the masked diagonal comes out −1.0×10⁹, dead. The ab-block logits hidden1·hidden2ᵀ/τ carry the positives on the diagonal: with images well-separated I get diagonal (aᵢ·bᵢ)/τ ≈ (9.86, 9.94), large and positive, as they should be. Concatenating [logits_ab, logits_aa] and labeling anchor aᵢ with column i, the softmax cross-entropy puts p(target)=0.999 on b1 for a1 and 0.997 on b2 for a2 — the label index lands on the counterpart view and nowhere else. So the one cross-entropy call really is NT-Xent: positive = counterpart, self masked out, everything else a negative. Good — the trick is not just compact, it indexes correctly. The surrounding TensorFlow v1 module provides `tf`, `xla.replica_id`, `tpu_cross_replica_concat`, `FLAGS`, `linear_layer`, `tpu_function`, and `LARSOptimizer`.

```python
LARGE_NUM = 1e9

def add_contrastive_loss(hidden, hidden_norm=True, temperature=1.0,
                         tpu_context=None, weights=1.0):
    # hidden: (2N, dim) — first N rows are view a, last N are view b
    # (the two augmentations of the same N images)
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)   # to the unit sphere: cosine sim, clean tau
    hidden1, hidden2 = tf.split(hidden, 2, 0)         # the two views
    batch_size = tf.shape(hidden1)[0]

    if tpu_context is not None:                        # large batch: pull negatives across cores
        hidden1_large = tpu_cross_replica_concat(hidden1, tpu_context)
        hidden2_large = tpu_cross_replica_concat(hidden2, tpu_context)
        enlarged = tf.shape(hidden1_large)[0]
        replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
        labels_idx = tf.range(batch_size) + replica_id * batch_size
        labels = tf.one_hot(labels_idx, enlarged * 2)  # positive's position after concat
        masks  = tf.one_hot(labels_idx, enlarged)
    else:
        hidden1_large, hidden2_large = hidden1, hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks  = tf.one_hot(tf.range(batch_size), batch_size)

    # four similarity blocks, all divided by the temperature
    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM          # mask self-similarity (k=i term)
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature  # a->b, positive here
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature  # b->a

    # per anchor: correct class = the counterpart view; all other entries are negatives
    loss_a = tf.losses.softmax_cross_entropy(labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
    loss_b = tf.losses.softmax_cross_entropy(labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
    return loss_a + loss_b, logits_ab, labels          # symmetric: both (i,j) and (j,i)
```

The projection head is the one-hidden-layer (here generalized to a few layers) MLP — hidden layers are linear → BN → ReLU, and the final layer is still batch-normalized but carries neither bias nor ReLU; pretraining reads off the last layer z to compute the loss, while downstream can select back to h (index 0, before the head):

```python
def projection_head(hiddens, is_training, name='head_contrastive'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        mid_dim = hiddens.shape[-1]
        out_dim = FLAGS.proj_out_dim
        hiddens_list = [hiddens]                        # out[0] is h, before g
        if FLAGS.proj_head_mode == 'none':
            pass
        elif FLAGS.proj_head_mode == 'linear':
            hiddens = linear_layer(
                hiddens, is_training, out_dim,
                use_bias=False, use_bn=True, name='l_0')
            hiddens_list.append(hiddens)
        elif FLAGS.proj_head_mode == 'nonlinear':
            for j in range(FLAGS.num_proj_layers):
                if j != FLAGS.num_proj_layers - 1:
                    dim, bias_relu = mid_dim, True
                else:
                    dim, bias_relu = FLAGS.proj_out_dim, False
                hiddens = linear_layer(
                    hiddens, is_training, dim,
                    use_bias=bias_relu, use_bn=True, name='nl_%d' % j)
                hiddens = tf.nn.relu(hiddens) if bias_relu else hiddens
                hiddens_list.append(hiddens)
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))
        if FLAGS.train_mode == 'pretrain':
            return hiddens_list[-1]                     # z = g(h) for the loss
        return hiddens_list[FLAGS.ft_proj_selector]     # h (out[0]) downstream
```

Global BN plugs the co-location leak by averaging BN's moments across all cores; the optimizer is LARS, with BN/bias/supervised-head parameters excluded from weight decay:

```python
class BatchNormalization(tf.layers.BatchNormalization):
    def _cross_replica_average(self, t):
        n = tpu_function.get_tpu_context().number_of_shards
        return tf.tpu.cross_replica_sum(t) / tf.cast(n, t.dtype)

    def _moments(self, inputs, axes, keep_dims, mask=None):
        shard_mean, shard_var = super(BatchNormalization, self)._moments(
            inputs, axes, keep_dims=keep_dims, mask=mask)
        n = tpu_function.get_tpu_context().number_of_shards
        if n and n > 1:                                # aggregate BN stats across cores
            group_mean = self._cross_replica_average(shard_mean)
            group_var = self._cross_replica_average(shard_var)
            group_var += self._cross_replica_average(tf.square(group_mean - shard_mean))
            return group_mean, group_var               # no "who is co-located" leak
        return shard_mean, shard_var

optimizer = LARSOptimizer(learning_rate, momentum=FLAGS.momentum,
                          weight_decay=FLAGS.weight_decay,
                          exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])
```
