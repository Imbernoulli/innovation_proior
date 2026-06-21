Supervised learning wants a predictor with low expected loss under an unknown joint distribution over inputs and targets, but in practice we replace that distribution with the uniform empirical measure over the observed examples, which turns the objective into a finite list of point constraints. That replacement is convenient, yet it is almost completely silent about the rest of input space: a model can drive the observed loss to zero while behaving arbitrarily near, around, and between the sample atoms, because empirical risk simply never asks what the predictor does off the data. With modern networks this is not a corner case but the typical situation. Their parameter counts rival or dwarf the number of training examples, so capacity alone does not explain generalization, and randomization tests make the point sharp: standard architectures fit random labels to nearly zero training error, so empirical fit cannot by itself separate a useful rule from memorization. The same gap shows up around the data as fragility, where a network that classifies ordinary inputs well still flips its prediction under a small optimized perturbation, contradicting the unstated hope that fitting isolated points yields smooth behavior in their neighborhoods.

Vicinal risk minimization names what is missing. Rather than placing all probability mass on each training example, it replaces each atom with a vicinity distribution and trains on virtual samples drawn from those vicinities. Classic data augmentation is exactly this — translations, rotations, tangent moves, Gaussian noise — but each of those moves keeps the original label fixed, so it only works when we already know the transformation preserves class identity, and it never says what should happen as one example turns into another. The two natural ways to attack that question each solve only half of it. Same-class or feature-space interpolation thickens a class region but deliberately stays inside one class or one local neighborhood, dodging the case where a synthetic point moves toward a different class. Soft-target methods such as label smoothing reduce overconfidence by spreading target mass off the hard label, but that spread is fixed independently of which direction the input was displaced. The open problem is to enlarge empirical support while making the target vary with the actual synthetic input, instead of treating input augmentation and target smoothing as two unrelated tricks.

I propose mixup. The defining move is to draw two examples, x_i with target y_i and x_j with target y_j, draw a single coefficient lambda from a Beta distribution with parameters alpha and alpha where alpha is positive, and form a virtual example using the same coefficient in input space and target space. Concretely, the synthetic input is lambda times x_i plus one minus lambda times x_j, and the synthetic target is lambda times y_i plus one minus lambda times y_j, and we minimize the loss of the model on this pair. The whole method is this coupling: one lambda controls both the position of the synthetic input and the value of its target, encoding the prior that linear movement in input space should induce linear movement in target space. The role of alpha is to set where the virtual points concentrate. As alpha approaches zero the Beta distribution piles up at the endpoints and the virtual examples approach ordinary empirical examples; larger positive alpha pushes mass toward the interior and asks more questions between examples. In code the degenerate alpha less than or equal to zero case is handled by setting lambda to one, which recovers ordinary minibatch training exactly.

It matters that the obvious shortcuts are worse for concrete algebraic reasons, not just by taste. Averaging the inputs but keeping one endpoint's hard label creates a supervision cliff: a point just past the halfway mark gets one class and a point just before it gets the other, and a 55-45 mixture is made indistinguishable from a 95-5 mixture whenever both are snapped to the nearer endpoint, which cannot express the graded behavior we want. Plain label smoothing is blind to direction — it assigns the same small mass no matter which partner we moved toward, so the target never learns either the partner example or the amount of movement. Adding Gaussian noise has no honest target value at the perturbed point at all. mixup is built precisely so the target knows both the partner and the coefficient, and the modesty of the claim is deliberate: I am not asserting that an averaged image is a natural photograph. The classifier is already a function on the ambient input vector space, so it already produces some output on the averaged vector; the averaged input simply gives a place to ask what the output should be, and the averaged target gives the answer that makes the constraint meaningful.

The regularization argument shows why the paired target, and nothing else, does the work. Define an averaged predictor as the expectation over partner examples x double prime and lambda of the learned model evaluated at lambda times x plus one minus lambda times x double prime, and measure complexity by an empirical Lipschitz constant taken only over real training inputs, that is the supremum over training pairs of the norm of the difference in predictions divided by the norm of the difference in inputs. If the learned model fits the virtual examples, then at a paired virtual location its output equals the mixed target, so the model at lambda times x prime plus one minus lambda times x double prime equals lambda times the model at x prime plus one minus lambda times the model at x double prime, and likewise for x. Subtracting the two expressions, the shared one minus lambda times the model at x double prime term cancels, leaving only lambda times the difference between the model at x prime and the model at x. Because lambda is nonnegative, the norm of the expectation is bounded by the expectation of lambda times the corresponding training-data Lipschitz numerator, giving an empirical Lipschitz bound for the averaged predictor that is at most the expectation of lambda times the empirical Lipschitz constant of the original model. For the symmetric Beta distribution this expectation is one half for any positive alpha, so the load-bearing content of the bound is the cancellation enabled by paired target mixing, not an alpha-dependent constant; the varying regularization strength of alpha comes from where the sampled virtual points sit, not from a changing mean. The same cancellation explains the failures algebraically: keep one hard label and the output is no longer the convex combination, so the shared term does not cancel; smooth labels with a fixed amount and the softness is detached from the actual partner and coefficient.

The implementation should preserve exactly this coupling with no extra machinery. In a minibatch I sample one scalar lambda, shuffle the batch by a random permutation to choose partners, form the mixed input as lambda times the original batch plus one minus lambda times the shuffled batch, and compute lambda times the cross-entropy between the prediction and the original targets plus one minus lambda times the cross-entropy between the prediction and the shuffled targets. Cross-entropy is linear in the target distribution, so this weighted pair of hard-label losses is identical to cross-entropy against the dense target lambda times the original targets plus one minus lambda times the shuffled targets, which is why no custom loss is needed. The remaining design choices follow the same logic: random partners cover more of the span of high-dimensional training data than a nearest-neighbor rule and avoid committing to a metric; raw input interpolation constrains the classifier where it is actually evaluated even when the virtual input is not perceptually natural; and two-example mixtures keep the routine cheap, since adding more examples with Dirichlet weights raises cost without changing the core prior. Same-class pairs need no special case — if the two targets agree the target stays that class; if the two indices are identical the virtual example is the original for any lambda; and lambda near zero or one lands near an endpoint with that endpoint's label. The one honest limitation lives inside the same prior: if an interpolated point falls on the true data manifold with a soft target that conflicts with the real label there, the linearity constraint can cause underfitting. That does not invalidate the construction; it names the assumption, which is local linear behavior between sampled examples unless the data geometry makes that bias wrong.

```python
import numpy as np
import torch
import torch.nn as nn


def mixup_data(x, y, alpha=1.0):
    """Return mixed inputs, paired targets, and the scalar interpolation weight."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Cross-entropy against the two-point soft target."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# Small runnable demonstration and sanity check.
torch.manual_seed(0)
np.random.seed(0)

x = torch.randn(8, 4)
y = torch.randint(0, 3, (8,))
model = nn.Linear(4, 3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

for step in range(20):
    model.train()
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
    pred = model(mixed_x)
    loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final loss:", loss.item())
print("Lambda used in last step:", lam)
```
