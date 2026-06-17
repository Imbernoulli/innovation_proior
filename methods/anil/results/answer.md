# ANIL (Almost No Inner Loop)

ANIL is the head-only inner-loop version of MAML. It keeps the usual outer-loop meta-objective, but during task adaptation it freezes the network body and applies gradient descent only to the final task-specific head. The body is still learned by the outer loop.

## Update

With $\theta=(\theta_1,\ldots,\theta_{l-1},\theta_l)$ and $\theta_l$ the head, task $b$ uses
$$
\theta^{(b)}_m
= \left(\theta_1,\ldots,\theta_{l-1},
(\theta_l)^{(b)}_{m-1}
-\alpha\nabla_{(\theta_l)^{(b)}_{m-1}}
\mathcal L_{S_b}(f_{\theta^{(b)}_{m-1}})\right).
$$
The body entries stay fixed through the inner loop; the head takes the ordinary $-\alpha g$ descent step. The meta-loss remains
$$
\mathcal L_{\mathrm{meta}}(\theta)
=\sum_b\mathcal L_{Z_b}(f_{\theta^{(b)}_m}),
\qquad
\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L_{\mathrm{meta}}(\theta).
$$

ANIL is not first-order MAML. In the two-layer example $\hat y(x)=\theta_2(\theta_1x)$, the ANIL query prediction after one support step is
$$
\left[\theta_2-\alpha\frac{\partial L(\hat y(x_1;\theta),y_1)}{\partial\theta_2}\right]\theta_1x_2.
$$
The bracket still depends on $\theta_1$ through the support forward pass, so the outer gradient for $\theta_1$ still differentiates through the head update and keeps a cross-Hessian term. ANIL removes the body inner update; it does not detach the surviving head update.

## Implementation

The learn2learn ANIL reference wraps only the classifier in `MAML` and feeds it frozen features. This scaffold-compatible version is the same operation inside one module: identify the current head parameters, compute head-only gradients with graph retention, and give `update_module` zero updates for the body.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Set

import learn2learn as l2l

INNER_LR = 0.5


class InnerLoopOptimizer:
    """ANIL: adapt only the final task-specific head in the inner loop."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        self._head_param_names = self._infer_head_param_names(model)
        if not self._head_param_names:
            raise ValueError("could not identify final classifier/head parameters")

    @staticmethod
    def _infer_head_param_names(model: nn.Module) -> Set[str]:
        named = list(model.named_parameters())
        all_names = {name for name, _ in named}

        for token in ("classifier", "head"):
            selected = {
                name for name in all_names
                if token in name.lower().split(".")
            }
            if selected:
                return selected

        last_linear = None
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                direct = list(module.named_parameters(recurse=False))
                if direct:
                    last_linear = (module_name, direct)
        if last_linear is not None:
            module_name, direct = last_linear
            return {
                f"{module_name}.{param_name}" if module_name else param_name
                for param_name, _ in direct
            }

        last_param_module = None
        for module_name, module in model.named_modules():
            direct = list(module.named_parameters(recurse=False))
            if direct:
                last_param_module = (module_name, direct)
        if last_param_module is None:
            return set()
        module_name, direct = last_param_module
        return {
            f"{module_name}.{param_name}" if module_name else param_name
            for param_name, _ in direct
        }

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            named_params = list(model.named_parameters())
            head = [
                (name, param) for name, param in named_params
                if name in self._head_param_names
            ]
            if not head:
                raise RuntimeError("head parameters not found on adapted clone")

            head_names = [name for name, _ in head]
            head_params = [param for _, param in head]
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(loss, head_params, create_graph=True)
            grad_by_name = dict(zip(head_names, grads))

            updates = [
                -self.inner_lr * grad_by_name[name]
                if name in grad_by_name else torch.zeros_like(param)
                for name, param in named_params
            ]
            l2l.update_module(model, updates=updates)
        return model

    def meta_parameters(self) -> List[Tensor]:
        return []
```

Key faithfulness checks:

- The head update has the descent sign `-inner_lr * grad`.
- The body receives zero inner updates but remains in the outer graph through `param + 0`.
- `create_graph=True` keeps the second-order path through the head update.
- Head tensors are re-collected every step because `update_module` replaces parameter objects.
- There is no learned optimizer state, unlike Meta-SGD.

At test time, the same feature-reuse logic can be pushed further into NIL: discard the head, embed support and query examples with the body, and classify queries by a softmax over cosine similarities to support embeddings. That is a test-time endpoint, not the training rule; task-specific heads during training are still what pressure the body to learn strong reusable features.
