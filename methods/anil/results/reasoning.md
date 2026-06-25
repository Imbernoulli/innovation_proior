Let me start from the exact place where the cost enters. A task clone begins at the meta-initialization $\theta$, takes support-set steps, and the query loss is differentiated back through those steps. For one ordinary full-network step, the adapted point is $\theta'=\theta-\alpha\nabla_\theta L_S(f_\theta)$, so the query gradient contains $(I-\alpha\nabla_\theta^2L_S)^\top\nabla_{\theta'}L_Z$. That is not a symbolic nicety; it is the extra graph-retention and Hessian-vector work I pay inside every meta-batch. If I want to simplify this honestly, I cannot just say "first order" and detach everything. I need to know which inner-loop degrees of freedom are actually doing useful work, and that is an empirical question before it is a design question.

There is an immediate asymmetry in the architecture. The last layer is not just another layer. In each $N$-way episode the output coordinates are relabelled to a fresh set of classes, so a fixed classifier head cannot be right for every task's arbitrary alignment. The head has a legitimate reason to move. The earlier layers are different: edges, textures, parts, and higher visual features can be shared. So the question I should be asking is not whether any parameter should adapt; it is whether the body adapts in a functional sense. Rapid learning says yes, the body should change substantially for each task. Feature reuse says no, the body should already be a useful representation and the task-specific work should concentrate at the head. I do not get to assume which one is true.

I need measurements before cutting anything. The first diagnostic is blunt: train the usual full meta-learner, then at test time freeze a contiguous prefix of body layers and see what accuracy does. The logic is that if the body is rapidly specializing per task, then denying it the inner-loop updates should cost real accuracy. The MiniImageNet numbers do not behave that way. On 5-way 1-shot, no freezing gives $46.9\pm0.2$, while freezing all four convolutional layers gives $46.3\pm0.4$. On 5-way 5-shot, the drop from no freezing to freezing all four convolutional layers is $63.1\pm0.4$ to $61.0\pm0.6$. Both gaps sit inside or barely outside the reported error bars; a body doing the main task adaptation should not survive being frozen this gracefully. The reading is that the trained body can be reused almost as-is.

The freezing test is indirect — it tells me the frozen body is good enough, not that the unfrozen body stays put. So I check the represented function directly, comparing activations before and after the inner loop with CCA and CKA. The body layers come out at CCA similarity above $0.9$, with CKA near $1$; the head is below $0.5$. Weight movement tells the same story: the average Euclidean distance between initialization and post-adaptation weights is tiny for every layer except the last, even though the body holds far more parameters. One worry remains, that this is an artifact of a fully converged model where everything has stopped moving. The timing-through-training check addresses it: the high body similarity and freezing robustness are already visible around 10,000 iterations, well before convergence. During training as well as at test time, the inner loop barely touches the body.

The cut these point toward is to leave the body fixed during the inner loop and update only the head. I have to be careful with the word "fixed." The body is fixed only inside each task's fast adaptation. It is still learned by the outer loop, because the query loss flows through the body features. For task $b$, with $\theta=(\theta_1,\ldots,\theta_{l-1},\theta_l)$ and $\theta_l$ the head, the inner recurrence becomes
$$
\theta^{(b)}_m
= \left(\theta_1,\ldots,\theta_{l-1},
(\theta_l)^{(b)}_{m-1}
-\alpha\nabla_{(\theta_l)^{(b)}_{m-1}}
L_{S_b}(f_{\theta^{(b)}_{m-1}})\right).
$$
The sign is the usual descent sign, $-\alpha g$. The body entries are copied from the meta-initialization at every inner step; only the final component descends the support loss. The outer loss and outer update stay the same.

My first reflex is that this looks suspiciously like first-order MAML — both throw away second-order work, so maybe I have just re-derived FOMAML with extra steps and the "no inner loop" framing is cosmetic. I should not trust that either way by inspection; I will work it out on the smallest model that still has a body and a head. Take $\hat y(x;\theta)=\theta_2(\theta_1 x)$ with scalar weights, $\theta_2$ the head, squared-error loss $L=\tfrac12(\hat y-y)^2$, one support point $(x_1,y_1)$, one query point $(x_2,y_2)$, one inner step. The support residual at the initialization is $r_1=\theta_2\theta_1 x_1-y_1$, and $\partial L_S/\partial\theta_2 = r_1\,\theta_1 x_1$. The head-only update keeps $\theta_1$ at its init and moves the head to
$$
\theta_2' = \theta_2 - \alpha\,r_1\,\theta_1 x_1 .
$$
The query prediction is $q=\theta_2'\,\theta_1 x_2$, and the outer gradient I care about is $\partial L_Z/\partial\theta_1=(q-y_2)\,\partial q/\partial\theta_1$.

The question is whether $\partial q/\partial\theta_1$ contains a second-order term routed through the head update. It will, only if $\theta_2'$ itself depends on $\theta_1$. It does: $\theta_1$ appears inside the support residual $r_1$ used to form the head step. Differentiating,
$$
\frac{\partial\theta_2'}{\partial\theta_1}
= -\alpha\,\frac{\partial}{\partial\theta_1}\!\left(r_1\theta_1 x_1\right)
= -\alpha\left(2\theta_1\theta_2 x_1^2 - x_1 y_1\right),
$$
and $2\theta_1\theta_2 x_1^2 - x_1 y_1$ is exactly $\partial^2 L_S/\partial\theta_1\partial\theta_2$. So $\partial\theta_2'/\partial\theta_1 = -\alpha\,\partial^2 L_S/\partial\theta_1\partial\theta_2$: the head update carries the cross-Hessian of the support loss. Let me confirm the algebra and then see whether it actually moves the outer gradient, rather than just being a term that happens to be present.

I verify it symbolically and put numbers in. With sympy, the identity $\partial\theta_2'/\partial\theta_1 = -\alpha\,\partial^2 L_S/\partial\theta_1\partial\theta_2$ checks out exactly. Now take a concrete point $\theta_1=1,\ \theta_2=0.5,\ \alpha=0.4,\ x_1=2,\ y_1=1,\ x_2=1.5,\ y_2=0.8$. The cross-Hessian $\partial^2 L_S/\partial\theta_1\partial\theta_2 = 2\theta_1\theta_2 x_1^2-x_1y_1 = 2(0.5)(4)-2 = 2.0$, so $\partial\theta_2'/\partial\theta_1 = -0.4\cdot 2.0 = -0.8 \neq 0$. The full ANIL outer gradient at this point is $\partial L_Z/\partial\theta_1 = 0.0225$. If I detach the head update with respect to the body — which is precisely what first-order MAML does to kill the second-order graph — the same gradient becomes $-0.0375$. They differ by $0.06$, the entire contribution of the surviving cross-Hessian path. So this is not first-order MAML: FOMAML would have returned $-0.0375$ here, ANIL returns $0.0225$. The reduction is in the set of adapted parameters, not in the derivative order through the parameters that still adapt.

That distinction dictates the implementation. I must compute gradients only with respect to the current head parameters, and I must keep `create_graph=True` so the outer backward can differentiate through the head update — dropping it would silently collapse ANIL into the FOMAML number I just computed. Then I pass a full update list to `update_module`: zero tensors for body parameters, $-\alpha g$ for head parameters. Passing zeros for the body is doing real work, not bookkeeping: it must leave the body parameters in the clone as a `param + 0` node so the query features still trace back to the meta-initialization, while injecting no support-gradient body update.

I want to be sure the zeros-for-body trick actually preserves the body's path to the outer gradient, because if `param + 0` were optimized away or detached, the body would stop learning entirely and I would not necessarily notice from the loss. I trace a head-only step on a tiny two-layer linear net in torch, with a stand-in for `update_module` that functionally replaces each parameter by `param + update`. After one head-only inner step and a query backward, the meta-gradient with respect to the body initialization is non-`None` with norm $0.2626$ — the body does receive a meta-gradient even though its inner update was zero. As a second check that the surviving path is the second-order one, I rerun with the inner learning rate set to $0$ and to $0.5$. If the head update were detached from the body, the body's meta-gradient would not depend on the inner step at all. It does: the body meta-gradient norm is $0.3076$ at $\alpha=0$ and $0.2626$ at $\alpha=0.5$. The dependence on $\alpha$ is the same cross-Hessian path the scalar example exposed, now showing up through real autograd.

There is a code trap I have to respect for this to keep working across steps. `learn2learn.update_module` replaces parameter tensors with `p + update`, so after one inner step the parameter objects are new tensors. If I cached head tensor objects once in `__init__`, the next step could point at stale tensors detached from the live graph. Names are stable, object IDs are not, so I store head parameter names and re-collect the current tensors from `model.named_parameters()` at every step. The learn2learn reference expresses the same idea in a split model: pass data through `features`, wrap only `classifier` in `MAML`, clone and adapt that classifier, then evaluate on frozen features. In the single-module scaffold, re-identifying the final head by name or by the final linear module is the equivalent.

The train/test asymmetry around the head is worth pushing on. At test time, if the body features are good enough — and the freezing and CCA evidence says they are — I can go further and drop the head entirely, classifying by cosine similarity between support and query representations, essentially the Matching-Networks rule over frozen features. That gives a no-inner-loop test-time variant. But I should not carry that move into training. The training-regime comparisons argue against it: task-specific training with a head learns stronger features, while multitask training without that task-specific alignment can do worse than random features. The body learns reusable features precisely because the outer loop sees many different task-specific heads forcing the alignment; once the body is learned, the head matters less at test time than during training. So the design stays narrow: remove the body inner loop during training and testing, keep the head inner loop for the main algorithm, and optionally replace the head at test time with a cosine rule as a separate endpoint.

Putting the chain together: the expensive full-network inner loop is justified only if the body rapidly specializes. Freezing, CCA/CKA, Euclidean movement, and the early-training check all point the other way — the body is reused, the head moves. The cut that survives the math is to adapt only the head, with the same descent sign and the same outer objective, while retaining second-order differentiation through that head update, which I confirmed is genuinely nonzero both symbolically and by tracing autograd. The implementation follows: current head params each step, `autograd.grad(..., create_graph=True)`, full-length updates with zeros for the body and $-\alpha g$ for the head, no extra meta-parameters.

## Minimal ANIL implementation stub

```python
import torch
import torch.nn.functional as F
import learn2learn as l2l


def adapt_head_only(model, head_names, support_x, support_y,
                    inner_lr: float = 0.5, n_steps: int = 1):
    for _ in range(n_steps):
        loss = F.cross_entropy(model(support_x), support_y)
        named = dict(model.named_parameters())
        head_params = [named[n] for n in head_names]
        grads = torch.autograd.grad(loss, head_params, create_graph=True)

        updates = []
        for name, param in model.named_parameters():
            if name in head_names:
                updates.append(-inner_lr * grads[head_names.index(name)])
            else:
                updates.append(torch.zeros_like(param))
        l2l.update_module(model, updates=updates)
    return model
```
