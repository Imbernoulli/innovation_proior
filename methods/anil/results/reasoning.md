Let me start from the exact place where the cost enters. A task clone begins at the meta-initialization $\theta$, takes support-set steps, and the query loss is differentiated back through those steps. For one ordinary full-network step, the adapted point is $\theta'=\theta-\alpha\nabla_\theta L_S(f_\theta)$, so the query gradient contains $(I-\alpha\nabla_\theta^2L_S)^\top\nabla_{\theta'}L_Z$. That is not a symbolic nicety; it is the extra graph-retention and Hessian-vector work I pay inside every meta-batch. If I want to simplify this honestly, I cannot just say "first order" and detach everything. I need to know which inner-loop degrees of freedom are actually doing useful work.

There is an immediate asymmetry in the architecture. The last layer is not just another layer. In each $N$-way episode the output coordinates are relabelled to a fresh set of classes, so a fixed classifier head cannot be right for every task's arbitrary alignment. The head has a legitimate reason to move. The earlier layers are different: edges, textures, parts, and higher visual features can be shared. So the real question is not whether any parameter should adapt; it is whether the body adapts in a functional sense. Rapid learning says yes, the body should change substantially for each task. Feature reuse says no, the body should already be a useful representation and the task-specific work should concentrate at the head.

I need measurements before cutting anything. The first diagnostic is blunt but decisive: train the usual full meta-learner, then at test time freeze a contiguous prefix of body layers and see what accuracy does. If the body is rapidly specializing, freezing it should hurt. The MiniImageNet numbers do not behave that way. On 5-way 1-shot, no freezing gives $46.9\pm0.2$, while freezing all four convolutional layers gives $46.3\pm0.4$. On 5-way 5-shot, the drop from no freezing to freezing all four convolutional layers is $63.1\pm0.4$ to $61.0\pm0.6$. That is far too small for a body that is supposedly doing the main task adaptation. It says the trained body can be reused almost as-is.

The freezing test is still indirect, so I check the representation itself. I compare activations before and after the inner loop with CCA and CKA. The body layers sit at CCA similarity above $0.9$, with CKA near $1$; the head is below $0.5$. Weight movement gives the same story: the average Euclidean distance between initialization and post-adaptation weights is tiny for every layer except the last, even though the body has more parameters. The timing-through-training check matters too. The high body similarity and freezing robustness are already visible around 10,000 iterations. So this is not just a converged-model artifact. During training as well as testing, the inner loop is doing very little to the body.

Now the tempting cut is clear: leave the body fixed during the inner loop and update only the head. I have to be careful with the phrase "fixed." The body is fixed only inside each task's fast adaptation. It is still learned by the outer loop because the query loss flows through the body features. For task $b$, with $\theta=(\theta_1,\ldots,\theta_{l-1},\theta_l)$ and $\theta_l$ the head, the inner recurrence becomes
$$
\theta^{(b)}_m
= \left(\theta_1,\ldots,\theta_{l-1},
(\theta_l)^{(b)}_{m-1}
-\alpha\nabla_{(\theta_l)^{(b)}_{m-1}}
L_{S_b}(f_{\theta^{(b)}_{m-1}})\right).
$$
The sign is the usual descent sign, $-\alpha g$. The body entries are copied from the meta-initialization at every inner step; only the final component descends the support loss. The outer loss and outer update stay the same.

I should also check that I have not accidentally turned the method into first-order MAML. A two-layer scalar example is the cleanest check: $\hat y(x;\theta)=\theta_2(\theta_1x)$, where $\theta_2$ is the head. With one support point $(x_1^{(t)},y_1^{(t)})$ and one query point $(x_2^{(t)},y_2^{(t)})$, full MAML would use
$$
\theta_1^{(t)}=\theta_1-\alpha\frac{\partial L_S}{\partial\theta_1},
\qquad
\theta_2^{(t)}=\theta_2-\alpha\frac{\partial L_S}{\partial\theta_2}.
$$
The head-only rule uses $\theta_{\mathrm{fast}}^{(t)}=[\theta_1,\theta_2^{(t)}]$. The query prediction is
$$
\hat y(x_2^{(t)};\theta_{\mathrm{fast}}^{(t)})
=\left[\theta_2-\alpha\frac{\partial L(\hat y(x_1^{(t)};\theta),y_1^{(t)})}{\partial\theta_2}\right]\theta_1x_2.
$$
When I differentiate this query loss with respect to $\theta_1$, the explicit $\theta_1x_2$ factor gives the direct outer-loop feature-gradient, and the bracket also depends on $\theta_1$ through the support forward pass used to compute $\partial L_S/\partial\theta_2$. Differentiating the bracket contributes a $-\alpha\,\partial^2L_S/(\partial\theta_1\partial\theta_2)$ cross-Hessian term. So one second-order path is removed, the one through a body inner update, but the second-order path through the head update remains. That settles it: this is not first-order MAML. It cuts the set of adapted parameters, not the derivative order through the surviving adapted parameters.

That second-order check dictates the implementation. I must compute gradients only with respect to the current head parameters, and I must keep `create_graph=True` so the outer backward can differentiate through the head update. Then I pass a full update list to `update_module`: zero tensors for body parameters, $-\alpha g$ for head parameters. Passing zeros for the body is not just a convenience; it preserves the direct graph from the adapted clone's body parameters back to the meta-initialization while deliberately omitting any support-gradient body update.

There is a code trap. `learn2learn.update_module` replaces parameter tensors with `p + update`. After one inner step, object identities are different. If I cached head tensor objects once in `__init__`, the next step could point at stale tensors. Names are stable, object IDs are not, so I should store head parameter names and re-collect the current tensors from `model.named_parameters()` at every step. The learn2learn reference expresses the same idea in a split model: pass data through `features`, wrap only `classifier` in `MAML`, clone and adapt that classifier, then evaluate on frozen features. In the single-module scaffold, re-identifying the final head by name or by the final linear module is the equivalent.

The train/test asymmetry around the head is also important. At test time, if the body features are good enough, I can go even further: remove the head and classify by cosine similarity between support and query representations, essentially the Matching-Networks rule over frozen features. That gives a no-inner-loop test-time variant. But I should not remove the task-specific head from training. The training-regime comparisons show why: task-specific training with the head learns stronger features, while multitask training without that task-specific alignment can be worse than random features. The body learns reusable features because the outer loop sees many task-specific heads; after the body is learned, the head is less important at test time. So the final design is narrow: remove the body inner loop during training and testing, keep the head inner loop for the main algorithm, and optionally replace the head at test time with a cosine rule as a separate endpoint.

Putting the chain together, the expensive full-network inner loop was justified only if the body rapidly specialized. Freezing, CCA/CKA, Euclidean movement, and early-training checks all point the other way: the body is reused, the head moves. The mathematically safe cut is therefore to adapt only the head, with the same descent sign and the same outer objective, while retaining second-order differentiation through that head update. The implementation follows directly: current head params each step, `autograd.grad(..., create_graph=True)`, full-length updates with zeros for the body and $-\alpha g$ for the head, no extra meta-parameters.

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
