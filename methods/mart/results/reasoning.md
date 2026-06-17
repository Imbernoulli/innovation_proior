Let me start from what actually hurts. I train an image classifier to be robust by the saddle-point recipe: for each example I find the worst perturbation inside the eps-ball with PGD and then I minimize cross-entropy on that perturbed input. It works — it's the one thing that produces genuinely robust nets instead of nets that only look robust until someone attacks them properly. And yet on CIFAR-10, a dataset I can drive to near-perfect clean accuracy in my sleep, the robust accuracy under a strong PGD attack sits far below the clean accuracy, and the gap refuses to close. So the question is not "does adversarial training work" but "what is the standard recipe leaving on the table." I want to stare at the recipe itself rather than reach for more data or a bigger net, because there's something in the formulation that has always bothered me.

Here it is. An adversarial example, by its own definition, is a perturbation of a *correctly classified* example that flips it to wrong. That's the whole concept: you were right, the adversary nudges you, now you're wrong. But during training, at any epoch, the current model already gets some natural training examples wrong with no perturbation at all. For those, the "adversarial example" I generate by PGD is perturbing an input that was already misclassified — the object I'm crafting isn't even well-defined under the textbook notion. And the standard loss doesn't care: it perturbs every example and slaps the same cross-entropy on every perturbed input, whether the clean version was a confident correct answer or an outright miss. I've been treating "the example the model nails" and "the example the model already blows" as identical citizens of the training set. That can't be right, and nobody seems to have asked whether the distinction matters.

So before I design anything, let me actually probe it — split the training data by what the *current* model does on the *natural* image, and poke at the two halves separately. Take a model trained the standard way to maybe 87% clean training accuracy, and partition the natural training set into S-, the examples it currently misclassifies, and S+, an equal-size chunk it currently classifies correctly. Now I can ask three concrete things, and the answers tell me where the pressure is.

First: how much does each subset contribute to final robustness? If I simply *don't perturb* S- during training — train on the clean version of those examples while still perturbing everything else — the final robustness drops drastically. Do the same to S+ and robustness barely moves. So the misclassified examples are carrying most of the robustness; the correctly classified ones are comparatively expendable. That already tells me where to spend effort. Second: within the *inner maximization*, does attack strength on S- matter? Replace PGD with a weak one-step FGSM on S- only — and final robustness is essentially unchanged. Do the weak attack on S+ and robustness degrades. So on the misclassified subset, *how hard I attack* is nearly irrelevant; on the correctly classified subset it matters. Third: within the *outer minimization*, does the loss on S- matter? Add a consistency regularizer — a term that asks the perturbed output to match the clean output — to the loss on S- only, and final robustness improves substantially; add it to S+ only and it helps far less.

Let me extract the lesson, because it's pointed. The leverage is on misclassified examples; and on those, the maximization is a near-no-op while the minimization is where the action is. That kills the instinct to fix this by inventing a cleverer attack. Whatever I build has to live in the *outer loss*, and it has to *single out misclassified examples* and treat them differently from correctly classified ones. The standard recipe's sin is precisely that it doesn't differentiate. So let me try to write down a risk that does.

I'll work against the 0-1 loss first and worry about making it trainable afterward — that keeps me honest about what I actually want before surrogate losses blur it. The standard adversarial risk is the average, over examples, of "does the worst perturbation flip me off the label,"

  R(h) = (1/n) sum_i  max_{x' in B_eps(x_i)}  1( h(x') != y_i ).

Now I want to split this by whether the natural example is already right or wrong under the current model. Define, by the model's prediction on the clean image,

  S+ = { i : h(x_i) = y_i },   S- = { i : h(x_i) != y_i },

and let xhat'_i be the adversarial example, the argmax of the 0-1 attack inside the ball. The clean split is to define a per-example risk separately on the two halves and add them up. The correctly classified half I have no reason to change — those are the citizens the standard recipe already handles fine — so for S+ I just keep the standard adversarial risk,

  R+(h, x_i) := max_{x'} 1( h(x') != y_i ) = 1( h(xhat'_i) != y_i ).

The misclassified half is where I need to do something different. What? My first instinct is to also just minimize 1( h(xhat'_i) != y_i ) there — demand the perturbed misclassified example be classified correctly. But that's asking for the impossible-feeling thing: the *clean* version of this example is already wrong, so demanding its *perturbed* version be right is a harder target than I've even met on the easy case. Pushing hard on that term on examples the model can't even get right unperturbed feels like it'll just thrash. The diagnostic agrees from the other direction — it was the *consistency* regularizer, not a harder classification demand, that helped on S-.

So let me ask for something weaker and more honest on S-: not "be correct after perturbation," but "be *stable* under perturbation" — the perturbed output shouldn't differ from the clean output. Write that as an extra indicator, 1( h(x_i) != h(xhat'_i) ), the event that the perturbation actually changed the prediction. The idea of penalizing the perturbed prediction for *disagreeing with the clean prediction* rather than for *disagreeing with the label* is exactly the stability-training instinct — make the function locally flat around the input — and here it's the gentler ask that suits an example the model is already failing. So for misclassified examples I take the standard risk *plus* this stability term:

  R-(h, x_i) := 1( h(xhat'_i) != y_i ) + 1( h(x_i) != h(xhat'_i) ).

Now I have two different per-example risks and I want one objective. Before I just bolt them together with a case split, let me check whether the split is even necessary — what does R-'s extra term *do* on a correctly classified example? On S+, h(x_i) = y_i by definition. Substitute that into the stability indicator: 1( h(x_i) != h(xhat'_i) ) becomes 1( y_i != h(xhat'_i) ), which is *identical* to the first term 1( h(xhat'_i) != y_i ). The regularizer and the adversarial risk are the same event on correctly classified examples. So if I were to apply R- everywhere, on S+ it would just double-count the standard risk — harmless in spirit but redundant — and on S- it adds the genuinely new stability term. That means I don't need two separate formulas with a hard branch; I can write a single risk where the stability term is *switched on only for misclassified examples* by an indicator gate:

  R_misc(h) = (1/n) sum_i { 1( h(xhat'_i) != y_i )  +  1( h(x_i) != h(xhat'_i) ) · 1( h(x_i) != y_i ) }.

The first term is the ordinary adversarial risk, untouched, applied to everyone. The second is a *misclassification-aware regularizer*: a stability penalty gated by 1( h(x_i) != y_i ), so it only fires on examples whose clean version is already wrong. That gate is the whole idea — it's the formal version of "single out the misclassified examples and regularize *them*." And it falls out of the algebra: I didn't impose the gate by hand, the gate is what makes the two-subset definition collapse into one expression.

This is a clean target, but it's a sum of indicator functions — discontinuous, non-differentiable, intractable to minimize directly. I need surrogate losses, one for each indicator, and I want each surrogate to be *physically meaningful* for what its indicator was doing, not just smooth. There are three indicators to replace: (1) 1( h(xhat') != y ), "the perturbed input is misclassified"; (2) 1( h(x) != h(xhat') ), "the perturbation changed the prediction"; and (3) 1( h(x) != y ), the gate, "the clean input is misclassified." Let me take them one at a time.

Start with (1), the classification term on the perturbed input. The reflex is plain cross-entropy on xhat', -log p_y(xhat') — that's exactly what standard adversarial training uses. Let me think about whether I can do better, because this term applies to *every* example and it's the backbone of robustness. The thing I keep coming back to is the capacity argument: a robust decision boundary is genuinely more complicated than a clean one, classifying adversarial inputs needs a *stronger* classifier than classifying clean inputs, and the standard fix for that is just "use a bigger net." But I can also ask the *loss* to push harder for separation. Cross-entropy maximizes the true-class probability, but it doesn't directly care how large the *best wrong-class* probability is once the true class is on top. What I want, for robustness, is a wide margin between the true class and its nearest competitor — that's the margin-loss instinct, penalize the gap to the best other class so the boundary is shoved well clear of the example. So I'll *boost* cross-entropy with a margin term that explicitly drives down the largest wrong-class probability:

  BCE( p(xhat'), y ) = -log p_y(xhat')  -  log( 1 - max_{k != y} p_k(xhat') ).

The first term is ordinary CE, fitting the true class. The second term is the boost: it's small when the best competing class probability max_{k!=y} p_k is small, and it blows up as that competitor approaches certainty, so minimizing it shrinks the runner-up and widens the margin. It's a simple boost — not the only possible one, any margin-style enhancement of CE would serve — but it directly answers the "robust classification needs a stronger classifier" pressure with the loss rather than only with model size.

Now (2), the stability term 1( h(x) != h(xhat') ) — the perturbed prediction differing from the clean prediction. This one wants a surrogate that measures how *different* two output distributions are, the clean p(x) and the perturbed p(xhat'). The natural choice is the KL divergence,

  KL( p(x) || p(xhat') ) = sum_k p_k(x) · log( p_k(x) / p_k(xhat') ),

which is zero exactly when the two distributions match and grows as they diverge — a smooth surrogate for "the prediction changed." Minimizing it makes the network's output locally flat around the input, which is precisely the smoothness/stability I was after for the hard examples.

And (3), the gate 1( h(x) != y ), "the clean input is misclassified." This is the delicate one, because it's the *switch* that turns the regularizer on for misclassified examples, and a hard 0/1 switch is the worst possible thing to put inside a loss I want to differentiate and learn jointly. If I commit to a hard decision — threshold the prediction, decide "this example is in S-," and only then apply the regularizer — I freeze that decision, can't backprop through it, and can't let it co-evolve with the model as the example's status flips over training. A prior method that uses misclassified examples does exactly this hard thresholding and then doesn't even optimize the condition. I want a *soft* gate instead. What continuous quantity is large when the clean example is misclassified and small when it's confidently correct? The model already hands me one: 1 - p_y(x), the probability mass the network puts on *everything except* the true class, i.e. its error probability on the clean input. When the example is confidently correct, p_y(x) is near 1 and 1 - p_y(x) is near 0, so the regularizer is switched almost off. When the example is misclassified or merely uncertain, p_y(x) is small and 1 - p_y(x) is near 1, so the regularizer is switched on. It's a smooth, differentiable, jointly-learnable stand-in for the hard gate 1( h(x) != y ), and it does something nicer than the hard version: it gives a *graded* weight, leaning hardest on the most-wrong examples and tapering off as an example becomes confident — exactly the "spend effort on the hard ones" pressure the diagnostic pointed at, made continuous.

Now assemble the surrogate objective. The first indicator's surrogate, BCE, applies to all examples. The second indicator's surrogate, KL, is multiplied by the third indicator's soft surrogate, (1 - p_y(x)), so the regularizer is up-weighted on misclassified examples and down-weighted on confident ones. I add a single global scalar lambda to balance the classification term against the regularizer — one knob for the natural-vs-robust trade-off, fixed across examples since the *per-example* weighting is already handled by (1 - p_y). The per-example loss is

  ell(x, y, theta) = BCE( p(xhat'), y )  +  lambda · KL( p(x) || p(xhat') ) · ( 1 - p_y(x) ),

and the training objective averages this over the minibatch. The adversarial example xhat' I still generate with strong CE-PGD — the same attack as standard training — and I do *not* try to be clever in the inner maximization, because the diagnostic was unambiguous that on the misclassified examples the attack strength barely matters; the differentiation belongs entirely in this outer loss. So I reuse the cheap, strong, standard attack and put all the new structure in the minimization, which is exactly where the measurements said the leverage is.

Let me pause and check this against the loss I'd otherwise have reached for, the KL-regularized robust loss that decomposes robust error into natural plus boundary error: CE(p(x), y) + (1/lambda)·max KL(p(x) || p(x')). Two differences jump out and both are deliberate. That method fits the *clean* label with CE(p(x), y) and pushes the boundary with a KL term whose weight is *the same for every example*. Mine fits the *perturbed* input with a boosted CE and — the crucial part — weights the KL regularizer *per example* by (1 - p_y(x)), so an already-misclassified example gets a much heavier robustness push than a confidently-correct one. The uniform-weight version pushes equally hard on the example it already nails and the example it already blows; my measurements say that's the wrong allocation, and the (1 - p_y) factor reallocates the regularization toward the examples that the diagnostic showed actually move robustness. So the per-example reweighting isn't a flourish; it's the direct implementation of the one empirical finding that motivated the whole exercise.

Now I have to turn this into code, and there are a couple of places where the clean math meets a softmax and I have to be careful. Let me write the inner attack first, because it's the standard part. Put the model in eval mode, start from the clean image plus a tiny random nudge so the attack isn't deterministic, and run PGD on cross-entropy: each step, take a signed-gradient step of size alpha, project the perturbation back into the L_inf eps-ball, and clamp the pixels to a valid [0,1] image. After the loop I have xhat' = adv.

Then the loss. Put the model back in train mode, forward both the clean images and the adversarial images, and softmax the adversarial logits to get adv_probs = p(xhat'). The first subtlety is the margin term, -log(1 - max_{k!=y} p_k(xhat')). I need, per example, the largest probability among classes *other than* y. The trick is to argsort the probabilities and take the top two indices; the top-1 is the model's predicted class. If the top-1 equals y, the runner-up I want is the top-2; if the top-1 is *not* y, then the top-1 itself is already the best wrong class, so that's what I want. So define new_y = (top-1 == y ? top-2 : top-1), and then in *both* cases p_{new_y} = max_{k!=y} p_k exactly — when the model is right, new_y is the second-best class; when it's wrong, new_y is the (wrong) top class. That's precisely the largest competing-class probability the margin term needs. The boosted CE is then the ordinary cross-entropy on the adversarial logits plus the negative-log of (1 - p_{new_y}); I write the second piece as an nll_loss against target new_y on log(1 - adv_probs), with a hair of slack (1.0001 instead of 1.0, plus a tiny epsilon) so that when a probability rounds to 1 the log doesn't go to -inf.

The second subtlety is the KL term. I want KL(p(x) || p(xhat')) = sum_k p_k(x)·(log p_k(x) - log p_k(xhat')), per example, summed over classes — and the framework's KL-divergence primitive expects the *log* of the second distribution as input and the first distribution as target, returning the elementwise summand, so I feed it log(adv_probs) and nat_probs and sum over the class dimension to get the per-sample divergence. The per-sample (1 - p_y(x)) weight is just one minus the natural-probability gathered at the true label. Multiply the per-sample KL by (1 - p_y(x)), average over the batch, and that's loss_robust. The natural logits I leave attached to the graph — I want the regularizer's gradient to flow through the clean forward pass too, since p(x) appears in both the KL target and the (1 - p_y) weight and both are part of what I'm optimizing. Total loss is the boosted-CE term plus lambda times the weighted-KL term; lambda I set to 6, a robustness weight in the same range the KL-decomposition method uses, balancing the natural and robust pieces. Backprop, step the optimizer, done.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialTrainer:
    """Misclassification Aware adveRsarial Training (MART)."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes
        self.beta = 6.0          # lambda: weight on the misclassification-aware KL regularizer
        self.kl = nn.KLDivLoss(reduction='none')

    def train_step(self, images, labels, optimizer):
        # --- inner maximization: strong L_inf CE-PGD with a small random nudge ---
        self.model.eval()
        adv = images.detach() + 0.001 * torch.randn_like(images)
        adv = torch.clamp(adv, 0.0, 1.0)
        for _ in range(self.attack_steps):
            adv.requires_grad_(True)
            loss_ce = F.cross_entropy(self.model(adv), labels)
            grad = torch.autograd.grad(loss_ce, adv)[0]
            adv = adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv - images, min=-self.eps, max=self.eps)
            adv = torch.clamp(images + delta, 0.0, 1.0).detach()

        # --- outer minimization: the misclassification-aware loss ---
        self.model.train()
        optimizer.zero_grad()
        batch_size = images.size(0)

        logits = self.model(images)            # clean forward (kept in the graph)
        logits_adv = self.model(adv)           # adversarial forward
        adv_probs = F.softmax(logits_adv, dim=1)

        # boosted CE: CE on adv + margin term -log(1 - max_{k!=y} p_k(xhat'))
        # new_y = best WRONG class: top-2 picks top1 (if top1 != y) else the runner-up,
        # so p_{new_y} = max_{k!=y} p_k in both cases.
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(logits_adv, labels) \
            + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

        # misclassification-aware KL regularizer, weighted per example by (1 - p_y(x))
        nat_probs = F.softmax(logits, dim=1)
        true_probs = nat_probs.gather(1, labels.unsqueeze(1)).squeeze(1)   # p_y(x)
        kl_per_sample = torch.sum(
            self.kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1)       # KL(p(x)||p(xhat'))
        loss_robust = (1.0 / batch_size) * torch.sum(
            kl_per_sample * (1.0000001 - true_probs))                      # x (1 - p_y(x))

        loss = loss_adv + self.beta * loss_robust                          # BCE + lambda * weighted-KL
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}
```

Let me trace the causal chain one more time so I trust it. The standard recipe applies one cross-entropy to every perturbed input identically, and the robust-vs-clean gap won't close. Staring at the definition of an adversarial example — it's only defined on a *correctly* classified input — made me split the training set by the current model's clean prediction and measure the two halves. The measurements said three things: misclassified examples dominate final robustness, the inner attack strength on them barely matters, and the outer loss on them matters a lot. So I wrote a risk that differentiates the halves: the standard adversarial risk for everyone, plus a *stability* regularizer for the misclassified ones, and on the correctly classified half the stability term collapses into the standard risk, so the whole thing becomes one objective with a misclassification gate. Making it trainable forced three surrogate choices: boosted cross-entropy for the classification term, because robust classification needs a stronger, wider-margin classifier; KL divergence between clean and perturbed outputs for the stability term; and — the linchpin — a *soft* gate (1 - p_y(x)) for the misclassification indicator, so the regularizer is reweighted per example toward the hard, already-wrong ones instead of switched by a frozen hard threshold. The inner attack stays the cheap, strong, standard PGD, because the measurements said all the leverage is in the minimization. The result is BCE on the adversarial input plus lambda times a (1 - p_y)-weighted KL between clean and adversarial outputs, and it drops straight into the standard adversarial-training step.
