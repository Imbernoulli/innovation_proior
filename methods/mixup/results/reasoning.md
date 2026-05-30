OK, let me think this through from scratch. I'm training a big classifier — more parameters than I have training images — and two things keep going wrong, and they feel like the same thing wearing two costumes.

First costume: memorization. There's that unsettling result from Zhang and co. — take CIFAR, replace every label with a random one, and the network still drives training error to zero. With *real* labels too, it'll happily fit everything even when I pile on regularization. So "low training loss" tells me almost nothing about whether the thing generalizes. Second costume: I take a correctly-classified image, nudge it by an amount I can't even see — Szegedy's adversarial examples, the perturbation you get by walking up the gradient of the loss with respect to the input — and the prediction flips. Goodfellow's FGSM does it in one gradient sign step.

Why are these the same thing? Let me write down what I'm actually minimizing. I want the expected risk R(f) = ∫ ℓ(f(x), y) dP(x, y) under the true distribution P. I don't have P. So what does everyone do — what do *I* do by default? I replace P with the empirical distribution: P_δ(x, y) = (1/n) Σ_i δ(x = x_i, y = y_i). A pile of Dirac spikes, one on each training pair. Plug that in and the integral collapses to R_δ(f) = (1/n) Σ_i ℓ(f(x_i), y_i). That's ERM. Vapnik's principle, and there's even a theorem — VC theory — that says minimizing R_δ is fine *as long as the capacity of my function class doesn't grow with n*. But my network's parameter count scales right along with the dataset. So the theorem's premise is exactly the thing I'm violating.

And now the two costumes drop off at once. P_δ is a sum of *Dirac masses*. It has support only on the n training points. So R_δ only ever looks at f at those n points. Everywhere else — between two training images, just outside one of them — the objective is completely silent. f can do *anything* off those points and pay nothing. Memorization is one way to exploit that silence (fit the spikes, ignore the gaps). Adversarial fragility is another reading of the same silence (the function is wild in the gap right next to a data point, so a tiny step lands somewhere the loss never constrained). It's not two problems. It's one: I'm minimizing against the wrong distribution. The empirical distribution is a degenerate stand-in for P, and its degeneracy is that it's all spikes and no neighborhood.

So the real question isn't "how do I regularize the network" — it's "what should I put in place of P_δ?" I want something that still uses my n points but *spreads* them out, that says something about the region around each one, not just the point.

What do people already do that spreads the points out? Data augmentation. For images I flip horizontally, crop with a little padding, rotate a hair, rescale. Each real image becomes a little cloud of variants, all carrying the same label. And empirically it works — it's the single most reliable generalization trick there is. Let me make sure I understand *why* it works in the language I just set up, because that's the clue. Augmentation is replacing the Dirac on (x_i, y_i) with a *distribution* — a cloud of (transformed x_i, same y_i). It's not P_δ anymore. There's a name for this, Chapelle's Vicinal Risk Minimization: pick a *vicinity distribution* ν(x̃, ỹ | x_i, y_i) — a measure of how likely a virtual pair (x̃, ỹ) is to live in the vicinity of (x_i, y_i) — and use P_ν(x̃, ỹ) = (1/n) Σ_i ν(x̃, ỹ | x_i, y_i). Sample a virtual dataset from that and minimize R_ν(f) = (1/m) Σ_j ℓ(f(x̃_j), ỹ_j). ERM is the special case ν = δ, the spike. Augmentation is some hand-built ν. Chapelle's own choice was the simplest non-trivial one: ν = N(x̃ − x_i, σ²) · δ(ỹ = y_i) — Gaussian noise on the input, label untouched.

So VRM is the right frame. The whole game reduces to: *choose the vicinity ν.* Augmentation chooses it with domain knowledge. And staring at that — two things bother me about every vicinity on the table.

One: they're all dataset-specific. Flips and crops are right for natural images and meaningless for a speech spectrogram or a table of numbers. I'd have to hand-engineer ν per modality. I want something data-agnostic — a vicinity I can write down without knowing anything about what the inputs *are*.

Two — and this is the deeper one — every vicinity I've listed keeps the label fixed. The Gaussian ball: δ(ỹ = y_i). Augmentation: every variant inherits y_i. SMOTE, when it interpolates, stays within one class. DeVries and Taylor interpolate and extrapolate, but among same-class neighbors. So the entire vicinity of each point lives *inside its own class*. Which means the region I most want to control — the space *between* a cat and a dog, where the decision boundary sits and where the function is least constrained — never gets any virtual data at all. Single-class vicinities can't, by construction, say anything about the cross-class gap. And the cross-class gap is exactly where adversarial examples live and where the boundary oscillates.

So now I have a sharp target: a vicinity distribution that (a) needs no domain knowledge — I can write it for pixels, audio, or tabular rows identically — and (b) deliberately reaches *across* classes, putting probability mass in the between-class region.

How do I make a virtual point that's data-agnostic and lands between two examples? The most primitive operation I have on feature vectors, regardless of what they are, is to average them. Take two training inputs x_i and x_j and form a convex combination x̃ = λ x_i + (1 − λ) x_j, λ ∈ [0, 1]. No domain knowledge in there at all — it's arithmetic on vectors. As λ sweeps 0→1, x̃ traces the straight segment from x_j to x_i. That's literally a point in the between-region. Good. That's half of it.

But what label do I give x̃? If x_i is a cat and x_j is a dog and λ = 0.5, the standard moves are: call it a cat (it inherits the dominant source's label), or call it a cat with some smoothing. But that throws away the structure I just built. I made a point that is *literally half cat, half dog* in input space — why would I tell the network it's all cat? The honest target is the *same* convex combination of the labels: ỹ = λ y_i + (1 − λ) y_j, with the y's as one-hot vectors. So at λ = 0.5, ỹ = (0.5 cat, 0.5 dog). The label interpolates exactly as the input interpolates.

Let me say what prior I've just encoded, because it's the whole idea: *linear interpolations of inputs should map to linear interpolations of targets.* I'm asserting that f, at least between training points, ought to behave linearly. And — this is the part the single-class vicinities structurally couldn't do — the supervision now *depends jointly on the input and its label*. Label smoothing softens the target too, but it softens it independently of the input, a fixed ε spread over wrong classes. Here the softness of the label is *tied* to how the input was blended. That coupling is the new thing.

So my vicinity distribution is: for each (x_i, y_i), mix it with a randomly chosen (x_j, y_j) at a random λ,

  μ(x̃, ỹ | x_i, y_i) = (1/n) Σ_j E_λ[ δ(x̃ = λ x_i + (1 − λ) x_j, ỹ = λ y_i + (1 − λ) y_j) ],

and sampling from it is just: draw a pair, draw λ, output (λ x_i + (1−λ) x_j, λ y_i + (1−λ) y_j). It's a *generic* vicinity — it slots straight into Chapelle's VRM in place of his Gaussian.

Now, the distribution of λ. I want a single knob that controls how aggressively I mix. λ ∈ [0, 1], so the natural family is Beta(α, α) — symmetric, since which example I call x_i versus x_j is arbitrary. Let me check the limits, because a good knob has to recover the old behavior at one end. As α → 0, Beta(α, α) piles all its mass at 0 and 1 — so λ is essentially always 0 or 1, x̃ is essentially always one of the raw examples, ỹ its raw label. That's ERM back exactly. Good — so this is a strict generalization of standard training, ERM sitting at α → 0. As α grows, λ concentrates near 1/2 — heavy mixing, virtual points pushed far into the between-region. At α = 1, Beta(1,1) is uniform on [0,1] — every blend ratio equally likely. So α tunes interpolation strength, and small α (a little mixing) versus large α (mostly half-and-half blends) is one dial. That's clean.

Let me pause on whether the label-mixing actually matters or whether I'm overcomplicating — could I just mix inputs and keep the dominant label? If I do that I'm back to a single-class-ish target and I've thrown away the joint coupling. The cross-class point would be supervised as if it belonged to only one endpoint, which invites a sharp boundary right through the segment. The only target that matches the construction of the input is the convex target.

Now — *why* would training on this help? Let me reason about what minimizing R_μ does to f. I'm feeding f points all along the segment between x_i and x_j and demanding its output be the linear blend of the endpoint labels. If f obeys, then between any two training points f's prediction moves linearly. Linear is about the simplest behavior a function can have between two anchors — by Occam, a good default for a region I have no other information about. And linearity is the *opposite* of oscillation: a function pinned to interpolate linearly between points can't wiggle wildly in the gaps, can't be wildly overconfident in the middle of nowhere. So the off-data silence that caused both my problems gets filled with the mildest possible behavior.

Trace it back to the two costumes. Adversarial fragility: the attack ascends the input-gradient of the loss; its bite depends on how sharply the loss can rise along plausible directions away from an example, and the most natural such directions point toward other examples. If I force f to vary linearly along those chords, the loss no longer gets to hide arbitrary spikes between the endpoints; the slope along the chord is tied to the endpoint labels and the endpoint distance. That is exactly the kind of constraint a gradient-ascent attack should hate, and I get it without an explicit Jacobian penalty or adversarial-example generation in the loop. Memorization: to memorize a random label I need f to spike at one point and ignore its neighbors. But this rule keeps showing f interpolations of real examples, which are cheap to fit with smooth structure, while interpolations that drag along a random label are awkward because every point now has to agree with mixtures involving other points. Larger α puts more mass away from the endpoints, so the spiky solution has less room to hide near the original samples.

I want to make the "controls complexity" claim precise, not just hand-wave "linearity is simple." Let me try to bound the complexity of the mixed model. Measure complexity by a Lipschitz constant over the data: for a function g, Lip̂(g) = sup_{x, x' ∈ D} ‖g(x') − g(x)‖ / ‖x' − x‖ — how fast g can change between training points. Consider the *expected* mixup model, f̃(x) = E_{x', λ}[ f̂(λ x + (1 − λ) x') ], where f̂ is my trained network. Suppose f̂ has driven training error to zero — then on the interpolated inputs it actually produces the interpolated targets, i.e. f̂(λ a + (1 − λ) b) = λ f(a) + (1 − λ) f(b) for training points a, b, where f is the target function. Now compute:

  Lip̂(f̃) = sup_{x, x'} ‖f̃(x') − f̃(x)‖ / ‖x' − x‖
        = sup_{x, x'} ‖ E_{x'', λ}[ f̂(λ x' + (1−λ) x'') − f̂(λ x + (1−λ) x'') ] ‖ / ‖x' − x‖.

Use zero-error linearity on each term inside:

        = sup_{x, x'} ‖ E_{x'', λ}[ λ f(x') + (1−λ) f(x'') − λ f(x) − (1−λ) f(x'') ] ‖ / ‖x' − x‖.

The (1−λ) f(x'') terms cancel — same x'' in both, so the mixing partner washes out:

        = sup_{x, x'} ‖ E_λ[ λ ( f(x') − f(x) ) ] ‖ / ‖x' − x‖
        = sup_{x, x'} E[λ] · ‖ f(x') − f(x) ‖ / ‖x' − x‖
        ≤ E[λ] · sup_{x, x'} ‖ f(x') − f(x) ‖ / ‖x' − x‖
        = E[λ] · Lip̂(f).

So the Lipschitz constant of the mixed model is at most E[λ] times that of the target. For the symmetric Beta(α, α) family, E[λ] = 1/2 for every positive α, so this particular bound is a coarse half-Lipschitz statement rather than an α-sensitive one. α still changes the sampled geometry — near endpoints when α is tiny, near the segment middle when α is large — but this proof only sees the mean. And notice what made the third line work: I needed the value of the target *at the interpolated location*, λ f(a) + (1−λ) f(b) — which is exactly the supervision this rule provides and ordinary augmentation does not. Gaussian-noise augmentation never tells me the target at a blended point; shrinking inputs toward a constant does not either. The label-mixing is not decoration; it is the thing that makes the bound go through.

I could try to apply the same interpolation inside the network instead of at the input. But the problem I started with lives in input space: adversarial steps, decision-boundary gaps, and the silence between training images are all statements about f as a function of x. A hidden representation is something the network can reshape to make the constraint easier to satisfy. The earlier I impose the linearity, the less the network can route around it, so the clean version mixes raw inputs.

The pairs should not be restricted to examples with the same label. Same-class interpolation is the SMOTE instinct, and it is useful for oversampling, but it leaves the decision-boundary gap exactly where it was. The point here is to put supervision on the chord between different classes too, so the partner has to be a random training example, not a same-class neighbor.

Nearest-neighbor pairing is tempting because nearby blends look more realistic. But realism is not the only goal; the unconstrained space is large, and random pairs span much more of it. A k-neighbor rule would collapse the vicinity back toward local same-manifold clouds, close to ordinary augmentation. Random pairing is also the cheapest possible sampling rule, so I will take the larger coverage and the simpler code.

I could blend three or more examples with a Dirichlet vector, but that changes the prior from pairwise linear behavior along chords to averaging toward a many-point center. It costs more, and the extra partners do not add a new constraint I can state as cleanly as "between two anchors, interpolate linearly." Two examples are enough.

Now the implementation, and there's a small trick that makes it nearly free. Naively I'd run two data loaders to get two independent streams of examples to pair up. But within a single minibatch the examples are already i.i.d. draws — so I can just pair the batch with a shuffled copy of itself. One loader, one random permutation of the batch indices, and I get random pairs at no extra I/O. Draw a single λ ~ Beta(α, α) for the batch, form mixed_x = λ·x + (1−λ)·x[perm].

And the loss has a shortcut. My target is the soft label ỹ = λ y_a + (1−λ) y_b. I could build that soft vector and feed it to a soft-target cross-entropy. But cross-entropy is *linear in the target*: CE(p, λ y_a + (1−λ) y_b) = −Σ_k (λ y_a + (1−λ) y_b)_k log p_k = λ·(−Σ_k y_{a,k} log p_k) + (1−λ)·(−Σ_k y_{b,k} log p_k) = λ·CE(p, y_a) + (1−λ)·CE(p, y_b). So I never have to materialize the soft label — I just compute the ordinary hard-label cross-entropy against y_a and against y_b and take the convex combination with the same λ. Same number, less plumbing, works with the stock CrossEntropyLoss that wants integer targets. Everything else — the network, SGD with momentum, weight decay, the step-decay schedule, the underlying flip/crop augmentation — stays exactly as it was; the new code only has to construct the virtual pair and evaluate the paired target.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = datasets.CIFAR10(root='~/data', train=True, download=False,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=8)

net = build_network()                       # e.g. a PreAct ResNet-18
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
vicinity_strength = 1.0                     # alpha in the equations


def make_training_pairs(x, y, strength=vicinity_strength, use_cuda=True):
    # one Beta draw per minibatch; strength <= 0 leaves the empirical batch unchanged
    lam = np.random.beta(strength, strength) if strength > 0 else 1.0

    # pair the batch with a shuffled copy of itself -> random cross-class pairs,
    # one data loader, no extra I/O
    if use_cuda:
        index = torch.randperm(x.size(0)).cuda()
    else:
        index = torch.randperm(x.size(0))

    mixed_x = lam * x + (1 - lam) * x[index, :]                # convex combo of inputs
    target_spec = (y, y[index], lam)                           # endpoint labels + weight
    return mixed_x, target_spec


def compute_loss(criterion, outputs, target_spec):
    y_a, y_b, lam = target_spec
    # CE is linear in the target, so loss on the mixed soft label
    # lam*y_a + (1-lam)*y_b  ==  lam*CE(.,y_a) + (1-lam)*CE(.,y_b);
    # no need to build the soft-label vector
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)


def train(epoch):
    net.train()
    for inputs, targets in trainloader:                       # stock loader, hard labels
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, target_spec = make_training_pairs(inputs, targets,
                                                  vicinity_strength, use_cuda)
        outputs = net(inputs)
        loss = compute_loss(criterion, outputs, target_spec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Minimizing against the empirical distribution P_δ — a sum of Diracs — only constrains f at the n training points, so f is free between and beside them, which surfaces as memorization and adversarial fragility. The fix is to minimize against a better approximation of P, a vicinity of each point instead of a spike. The vicinities people use are domain-specific and single-class, so they leave the cross-class between-region unconstrained. The generic, data-agnostic vicinity that reaches across classes is the convex combination of two random examples and their labels: x̃ = λ x_i + (1−λ) x_j, ỹ = λ y_i + (1−λ) y_j, λ ~ Beta(α, α). This encodes the prior that f should interpolate linearly between points; linearity limits off-data oscillation, makes random-label memorization expensive, and the Lipschitz calculation shows why the label mixing is necessary for the smoothness argument. α dials where the samples lie on the chord, recovering ordinary ERM in the α → 0 limit, and the implementation is just a shuffled minibatch, one Beta draw, and the linearity of cross-entropy in the target.
