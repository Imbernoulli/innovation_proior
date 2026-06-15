Let me start from the thing that actually breaks. I train a network on task A, it gets good, I move on and train it on task B with ordinary SGD — and when I go back and test A, it's gone. Not degraded gracefully, collapsed. And I know exactly why mechanically: the loss I'm minimizing on B is a function of the same shared weights theta, and gradient descent on that loss moves whatever weights reduce the B-loss fastest, with no term anywhere that says "don't touch the weights A was relying on." So the B-gradient happily drags the A-critical weights to wherever B wants them. This is the stability-plasticity tension stated concretely: the network is plastic enough to learn B precisely because every weight is free to move, and that same freedom is what loses A. If I want to keep A I have to take some of that freedom away — but not all of it, or I can't learn B. The whole game is going to be: take freedom away *selectively*, from the right weights, by the right amount.

What are my options, and why does each fall short? The clean fix is to just train on A and B together, interleaved — joint optimization over both losses removes forgetting entirely, because the weights get jointly fit to both. But in my setting the tasks arrive in sequence and I don't get to keep A's data around; to interleave I'd have to store A's examples and replay them while training B, and as the sequence grows the amount of stored-and-replayed data grows with the number of tasks. That's the cost the setting forbids — it's remembering by hoarding the data, not by remembering anything about the network. So replay is out. What about the opposite extreme that *does* respect the constraint: when I finish A, snapshot theta*_A, and while training B add a penalty pulling every weight back toward its A-value, sum_i (lambda/2)(theta_i - theta*_{A,i})^2 — plain L2 anchoring to the old solution. No old data, constant memory. But this has a single global stiffness lambda. If I make lambda big enough to hold A in place, I've frozen the entire network and there's no slack left to fit B — every weight is glued, including the ones B desperately needs to move. If I make lambda small enough that B can learn, the anchor is too weak to hold A. There's no global constant that does both, and I can see *why*: the weights that matter to A and the weights B needs to move are different sets, and a uniform penalty can't tell them apart. It pays the same protection-cost on weights A doesn't even care about, wasting exactly the capacity B wanted. Dropout and the other standard regularizers do a bit better in practice, but the diagnosis there is unflattering — they resist forgetting mostly by pushing toward *larger* nets, i.e. by buying spare capacity, and they stop working after a couple of sequential tasks. None of these is protecting *specific* knowledge.

So the uniform-anchor failure is actually giving me a sharper requirement: I want the same quadratic-spring idea — anchor to theta*_A, penalize departure — but with a *per-parameter* stiffness, big on the weights that mattered to A and small on the ones that didn't. The biology says the same thing, if I let myself take the analogy seriously: the brain doesn't wall off whole regions, it renders *specific important synapses* less plastic and leaves the rest free. So the structural question reduces to one number per weight: how important was weight i to task A? Get that, multiply it into the spring constant, done. Everything now hangs on defining "importance" in a way I can actually compute, cheaply, from a finished task with its data already gone.

Let me not guess at importance heuristically — let me derive it, because I want it principled, not bolted on. The honest way to think about "what does the network know about A" is probabilistic. Training to optimize parameters is, read properly, finding the most probable theta given the data. By Bayes, log p(theta | D) = log p(D | theta) + log p(theta) - log p(D), and the log-likelihood log p(D | theta) is just the negative of my loss — the cross-entropy loss *is* a negative log-likelihood. So MAP estimation and loss minimization are the same act viewed two ways. Now split the data into the part that defined A and the part that defines B, and assume they're independent given theta. I can apply Bayes a second time, peeling off B:

  log p(theta | D_A, D_B) = log p(D_B | theta) + log p(theta | D_A) - log p(D_B | D_A).

(I'll be careful: that last constant is the marginal likelihood of B given A; one's tempted to call it log p(D_B) by independence, but the Bayesian treatment couples the tasks — they're exchangeable, not independent — so it's a conditional. Either way it doesn't depend on theta and I never compute it, so it falls away the moment I take a gradient. Set it aside.)

Stare at the right-hand side, because it's telling me something. The first term, log p(D_B | theta), is just the task-B loss — tractable, it's what I'd minimize anyway. The middle term, log p(theta | D_A), is the *posterior over weights given task A* — and notice that ALL of the information about task A has been absorbed into this one object. The full posterior on the left depends on the entire history, but the whole of A's contribution shows up as p(theta | D_A). So if I had that posterior, I'd be done: maximize log p(D_B | theta) + log p(theta | D_A) and I'm simultaneously fitting B and respecting everything A taught me. The posterior *is* the compact summary of A I was looking for. The trouble is it's a distribution over millions of weights, completely intractable. I need an approximation of it that (i) is cheap to store, (ii) is cheap to add to the per-step loss, and (iii) actually encodes per-weight importance. A quadratic — a Gaussian — would give me all three: a Gaussian's negative-log-density is a quadratic in theta, which is exactly the spring penalty I already want, and its precision can become per-weight stiffness once I make the approximation diagonal.

How do I get a Gaussian approximation to p(theta | D_A) when I can't even write it down? This is the Laplace approximation, and it's natural here. I trained A to a (local) optimum theta*_A. At that point the gradient of -log p(theta | D_A) vanishes — that's what "optimum" means. So expand -log p(theta | D_A) in a second-order Taylor series around theta*_A: the constant term I don't care about, the *first*-order term is zero because the gradient vanishes there, and I'm left with

  -log p(theta | D_A) ~ const + (1/2)(theta - theta*_A)^T H (theta - theta*_A),

where H is the Hessian of -log p(theta | D_A) at theta*_A. That's the quadratic part of a Gaussian with mean theta*_A and precision H. And because theta*_A is a minimum of -log p, H is positive semi-definite there — good, a precision matrix had better be PSD or the "Gaussian" isn't one. So the posterior-summary of A is: a quadratic bowl centered at theta*_A whose curvature is H. The diagonal of H tells me, per weight, how sharply the old negative log posterior rises as I move that weight. A weight with large curvature is one the A-solution is sensitive to: move it and A degrades fast, so it should be stiff. A weight with near-zero curvature is one A doesn't care about: leave it loose so B can use it. The importance number I wanted is the local curvature around the old solution.

So I need H. Its data-likelihood part is a Hessian, its prior part is whatever curvature the parameter prior contributes, and the full thing is a millions-by-millions matrix; I can't form it, store it, or invert it. Wall. I need a first-order object that matches the likelihood curvature under the model distribution, then I can drop a broad prior term or absorb ordinary weight decay into the scalar that trades old and new tasks. This is where the Fisher information matrix earns its place. The Fisher is defined as F = E_{y ~ p_theta(y|x)}[ (grad_theta log p_theta(y|x))(grad_theta log p_theta(y|x))^T ] — the expected outer product of the score (the gradient of the log-likelihood), with the output y drawn from the model's *own* predictive distribution. The key fact is that this expected outer product equals the expected Hessian of the negative log-likelihood. Let me actually check that, because the whole argument leans on it and I don't want to take it on faith. Take one coordinate; write p for p_theta(y) and suppress arguments. The second derivative of log p is d/dtheta (1/p · dp/dtheta) = (1/p) d^2p/dtheta^2 - (1/p^2)(dp/dtheta)^2 = (1/p) d^2p/dtheta^2 - (d log p/dtheta)^2. Now take the expectation over y ~ p, i.e. multiply by p and sum/integrate over y:

  E[ d^2 log p / dtheta^2 ] = sum_y p · [ (1/p) d^2p/dtheta^2 - (d log p/dtheta)^2 ]
                            = sum_y d^2p/dtheta^2  -  E[ (d log p/dtheta)^2 ].

The first piece is d^2/dtheta^2 ( sum_y p ) = d^2/dtheta^2 (1) = 0, because the probabilities sum to one regardless of theta. So E[ d^2 log p/dtheta^2 ] = -E[ (d log p/dtheta)^2 ], i.e. E[ -d^2 log p/dtheta^2 ] = E[ (d log p/dtheta)^2 ]. The expected Hessian of the negative log-likelihood is the expected squared score — the Fisher. That single line is what rescues me from the likelihood Hessian: under the model distribution the curvature is an *average of squared first-order gradients*. No second derivatives at all. And as a bonus the outer-product form is automatically positive semi-definite — a sum of squares can't be negative — whereas the raw Hessian of the empirical loss can be indefinite and would give me negative "stiffnesses", which are meaningless as importances. So Fisher is the usable curvature proxy here: exact as an expected likelihood identity, computable from first order, guaranteed PSD, and close to the empirical old-task curvature when the fitted model is a good local description of the task.

So replace the data-curvature part of H by the Fisher F at theta*_A, fold the sample-size scaling and any broad prior stiffness into a hyperparameter, and keep moving. I still have a full matrix, F. But I'm in a million-dimensional weight space, and storing and using even one full Fisher per task is hopeless. The simplification that makes it real is to keep only the *diagonal* of F — treat the off-diagonals as zero. That's a genuine approximation: I'm asserting the posterior is a *factorized* Gaussian, one independent quadratic per weight, no cross-coupling. It's lossy — weights surely covary in their effect on A — but it's the price of linearity in the number of parameters, and the diagonal already carries the thing I most need: a per-weight curvature F_i, the per-weight importance. With the diagonal-Gaussian summary, the old-task term becomes sum_i (1/2) F_i (theta_i - theta*_{A,i})^2 up to a constant and a scalar multiplier.

Now assemble it. Maximizing log p(theta | D_A, D_B) = log p(D_B | theta) + log p(theta | D_A) is the same as minimizing the negative, and plugging the Laplace-Fisher approximation for the A-posterior gives the loss I'll actually minimize while training B:

  L(theta) = L_B(theta) + sum_i (lambda/2) F_i (theta_i - theta*_{A,i})^2.

There's the spring, derived rather than guessed: a quadratic anchor to theta*_A, with per-coordinate stiffness F_i — the diagonal Fisher, i.e. the local curvature of the A-loss, i.e. how much A cares about weight i. Important weights (large F_i) are held nearly rigid; unimportant ones (F_i ~ 0) are left free for B. The 1/2 is just the Gaussian quadratic-form factor that fell out of the Taylor expansion; I'll keep it because it's the honest coefficient, and because it makes the penalty's gradient exactly F_i (theta_i - theta*_{A,i}) with no stray factor of two. And lambda: the clean derivation actually says the Fisher should be multiplied by the *sample size* of task A (the Hessian of the summed log-likelihood is N times the per-example expected Hessian), but the diagonal Laplace approximation tends to be overconfident — a point estimate of curvature underestimates the true posterior spread — so rather than nail lambda to N I let it be a tunable knob that sets how much I weigh old task A against new task B. One scalar per importance term, chosen by search.

Let me make sure I can actually compute that diagonal Fisher, because the elegance is worthless if the computation is expensive or wrong. F_i = E_{x} E_{y ~ p_theta(y|x)}[ (d log p_theta(y|x)/dtheta_i)^2 ] at theta*_A. The expectation over x is just an average over a sample of the task's inputs — I'll draw a hundred or so examples, that's plenty for a stable diagonal. The inner expectation over y is the subtle one, and I want to get it right. The Fisher's definition averages the squared score over y drawn from the *model's own predictive distribution* p_theta(y|x), NOT over the true labels in the data. So for each input x, I run a forward pass to get the predicted class distribution p = softmax(logits); then for each possible class label k I compute the per-class negative log-likelihood -log p_theta(y=k|x), backprop it to get its gradient, square the gradient coordinatewise, and weight that by p_k — the model's predicted probability of class k — and sum over k. That weighted sum is the inner expectation E_{y ~ p_theta}[(score)^2] done exactly (the expectation over a categorical is just the probability-weighted sum over its outcomes). Average over the x's and I have the diagonal Fisher.

I want to be sure about that weighting, because there's a tempting shortcut. Why weight by the predicted p_k and not just plug in the true label, or the model's top prediction? Because only the expectation over y ~ p_theta is the thing that equals the expected Hessian — the derivation above used exactly E_{y ~ p}, the model's distribution, when it turned sum_y p · d^2p/dtheta^2 into d^2/dtheta^2(sum p) = 0. If I average over the true labels instead, I get the empirical Fisher, a useful but different object whose expectation is over the data labels rather than the model distribution. If I take the argmax label, I get a deterministic pseudo-label shortcut, not a sample from p_theta and not an unbiased Monte Carlo estimate of the Fisher. Drawing a label from p_theta would be a one-sample Monte Carlo estimate; summing over all classes weighted by p_k is the exact categorical expectation when the number of classes is small, which it is here, so I'll do the exact sum.

Now the practical wrinkles, because this runs on a real network. I have to put the network in eval mode for the Fisher pass — I don't want dropout or batchnorm noise corrupting the curvature estimate, and I want clean per-example gradients, so I process one input at a time. Each named parameter accumulates its running sum of p_k · (grad_i)^2 across the classes and across the sampled inputs; at the end I divide by the number of samples to make it an average. And I zero the model's gradients before each per-class backward so the squared gradients don't contaminate each other. The cost is: a hundred forward passes, and for each, one backward per class — linear in parameters, linear in examples, no matrix ever formed. That's the scaling win over the older quadratic-penalty methods that had to invert a parameter-sized matrix or recompute curvature per sample; this is just augmented backprop.

The per-step penalty is then trivial and cheap, which it has to be since it's evaluated every single training step: for each parameter, take its stored importance F_i and its anchor prev_i, add up F_i (theta_i - prev_i)^2 across all coordinates, multiply by 1/2, and finally scale by the strength lambda. A scalar tensor, added to the task loss, no data pass -- just an elementwise op over the current weights and the stored summary.

One more thing I have to settle before the code is the multi-task case, because the sequence doesn't stop at two. When task C arrives after A and B, I want to stay close to what *both* A and B needed. The cleanest move is to note that the sum of two quadratic penalties is itself a quadratic penalty, so I can keep one penalty per past task and add them all: sum over past tasks t of sum_i (lambda_t/2) F_{t,i} (theta_i - theta*_{t,i})^2, each anchored at the parameters as they stood when task t finished, each with its own stored diagonal Fisher. That keeps a separate record of every task's measured curvature and anchor, at the cost of memory that grows with the number of tasks.

Let me write it as the two routines the harness asks for: one that summarizes a finished context into the diagonal Fisher, and one that turns that stored record into the per-step term.

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def summarize_finished_context(model, dataset, device):
    """Diagonal Fisher information at the just-finished context's optimum.

    F_i = E_x E_{y ~ p_theta(y|x)} [ (d log p_theta(y|x)/d theta_i)^2 ],
    the model-distribution expected Hessian of the NLL -> per-weight curvature.
    """
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    anchor = {n: p.detach().clone() for n, p in params.items()}        # theta*_i
    est_fisher = {n: torch.zeros_like(p) for n, p in params.items()}   # F_i

    mode = model.training
    model.eval()                                     # clean gradients: no dropout/BN noise

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # per-example scores
    n_samples = min(len(data_loader), 200)           # ~hundreds of x's is enough for a diagonal

    for idx, (x, _) in enumerate(data_loader):
        if idx >= n_samples:
            break
        x = x.to(device)
        output = model(x)                            # logits for this input
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1).detach()
        for label_index in range(output.shape[1]):   # exact inner expectation: sum over classes
            label = torch.tensor([label_index], device=device, dtype=torch.long)
            negloglikelihood = F.cross_entropy(output, label)   # -log p_theta(y=k|x)
            model.zero_grad()
            negloglikelihood.backward(               # one backward per class -> the score
                retain_graph=True if (label_index + 1) < output.shape[1] else False
            )
            for n, p in params.items():
                if p.grad is not None:
                    # accumulate p_k * (score_i)^2: the y-expectation weighted by p_k
                    est_fisher[n] += label_weights[0, label_index] * p.grad.detach().pow(2)

    est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}  # average over x

    model.train(mode=mode)                           # restore training mode
    return {"fisher": est_fisher, "anchor": anchor}


def extra_training_loss(model, summary_state, strength=1.0):
    """EWC penalty: 0.5 * sum_i F_i (theta_i - theta*_i)^2.

    Sum one quadratic record per past context. `strength` is lambda unless a
    record carries its own value. Cheap: no data pass.
    """
    device = next(model.parameters()).device
    if not summary_state:
        return torch.zeros((), device=device)

    records = summary_state if isinstance(summary_state, (list, tuple)) else [summary_state]
    total = torch.zeros((), device=device)
    for record in records:
        fisher = record["fisher"]
        anchor = record["anchor"]
        scale = record.get("strength", strength)
        record_total = torch.zeros((), device=device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in fisher:
                record_total = record_total + (fisher[n] * (p - anchor[n]) ** 2).sum()
        total = total + 0.5 * scale * record_total   # lambda/2 times the diagonal quadratic
    return total
```

So the causal chain, end to end. SGD forgets because nothing in the B-loss protects the weights A needed, and the freedom that lets B learn is the same freedom that loses A. Replay respects nothing about the network and costs data that grows with the sequence; a uniform anchor can't distinguish A-critical weights from free ones, so it either freezes the net or fails to hold A -- which told me the answer is a quadratic spring with *per-weight* stiffness. Reading training as MAP and peeling task B off with Bayes showed that all of A's knowledge lives in the posterior p(theta | D_A), and that posterior is the summary I need; a Laplace approximation turns it into a Gaussian centered at theta*_A whose precision is the Hessian of the old negative log posterior. The full Hessian is unaffordable, but the model-distribution Fisher equals the expected Hessian of the negative log-likelihood -- a fact I checked by watching the second-derivative-of-the-density term integrate to zero -- so the likelihood curvature can be estimated by squared first-order gradients, cheaply and with a PSD matrix. Keeping only the diagonal makes it linear in the parameters, and the result is the penalty L_B(theta) + sum_i (lambda/2) F_i (theta_i - theta*_{A,i})^2: a spring that's stiff on the weights A cared about and loose elsewhere, with lambda absorbing sample-size and prior/approximation scaling. The Fisher diagonal is computed by averaging, over a sample of inputs, the probability-weighted squared per-class scores under the model's own predictive distribution -- the exact expectation that makes it the curvature. And the whole thing drops into the training loop as one post-context backprop pass and one elementwise penalty per step, holding multiple tasks by summing one quadratic per past task.
