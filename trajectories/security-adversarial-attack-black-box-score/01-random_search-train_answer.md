I am attacking a classifier I can only query: I send it an image and read back logits, with no weights and no gradient. For a correctly classified $(x, y)$ I want a nearby image it gets wrong, $\arg\max_k f_k(x_{\text{adv}}) \neq y$, with $\|x_{\text{adv}} - x\|_\infty \le \varepsilon$ and $x_{\text{adv}} \in [0,1]$. The white-box reference, PGD, would step along the gradient of the margin and project back into the box — but the gradient is exactly what I am denied. Everything I can do reduces to one lever: propose a perturbed image, query, and decide whether to keep it. Before reaching for anything clever about *where* or *how big* the moves should be, I want to establish the floor — the crudest honest use of that lever — so I know what every later method has to beat.

The method is greedy uniform random search. The first decision is what scalar to hill-climb, and I want the cheapest one that tracks progress using only the logits I already pay for. That is the correct-class score $f_y(x_{\text{adv}})$: it starts high because the model is confident in $y$, and the prediction flips once some other class overtakes it. I do not even need the full margin to descend — I just need a number that falls as I approach the boundary, and $f_y$ is the simplest such number, recovered with a single `gather` on the logits. So the floor minimizes $f_y$ by greedy accept-if-better: a candidate is kept only when it lowers the correct-class logit.

With the objective fixed, the only remaining design questions are the proposal and the budget bookkeeping, and the floor answers both as bluntly as possible. The proposal is the textbook random-search move: draw isotropic uniform noise in a small box, $\text{noise} \sim \mathcal{U}(-\text{step}, \text{step})$ with $\text{step} = \varepsilon/2$, add it to the current best, then re-project into the feasible set with $\text{clamp}(x + \text{clamp}(\text{cand} - x, -\varepsilon, \varepsilon),\, 0,\, 1)$. There is no structure in this whatsoever — no notion that the model is convolutional, no notion that successful $L_\infty$ perturbations sit at corners of the box, no decay of the step size, no concentration of the change into any region. It scatters tiny nudges across all $C \cdot H \cdot W$ coordinates at once. The $\text{step} = \varepsilon/2$ choice is itself timid: a typical proposal lands well *inside* the box rather than on its boundary, so the floor never even spends its full per-component budget.

The budget bookkeeping is where the floor quietly throws away most of its allowance, and this is the detail that decides its number. The harness gives a per-sample budget $n_{\text{queries}}$ (the runs use 1000), and the oracle scores the *entire batch* as a failure the instant the running query count crosses $\text{batch\_size} \cdot n_{\text{queries}}$. The floor does not walk anywhere near that line: it runs a fixed $n_{\text{steps}} = \max(1, \min(n_{\text{queries}}, 64))$ iterations, each costing one query per sample, and it queries *only the candidate* each step — it never re-queries the current best, because it carries the best score forward from the previous accept. So with a 1000 budget it caps itself at 64 candidate queries plus one initial query of the clean image, about 65 total, deliberately leaving roughly 935 of every 1000 queries on the table. That self-imposed ceiling, far below the real budget, is the first of its two crippling weaknesses.

The one piece of vectorized care worth naming is that the accept rule operates per sample. I compute the candidate's correct-class score for every image in the batch, form $\text{improve} = \text{cand\_score} < \text{best}$, and update only the rows where it holds via a masked `torch.where`; an image whose candidate got worse simply keeps its previous best. So a single batched query advances every still-improving sample at once, the budget shared across the batch but the keep/reject decision independent per image. That structure is right and is the one thing later rungs inherit unchanged.

What this floor must do is the entire point of running it. Greedy random search with isotropic interior noise in a roughly 3000-dimensional space (CIFAR is $3 \times 32 \times 32$) is the classic high-dimensional failure: a random direction is almost orthogonal to whatever direction would actually lower $f_y$, so most proposals are rejected and the few that are accepted make tiny progress. With only 64 such steps the perturbation barely moves off the clean image. On the easier (model, dataset) pairs — where the boundary is close enough that even crude noise stumbles across it — the floor should flip a fair fraction of images; on the harder pairs, a more robust architecture or CIFAR-100 with its 99 competing classes, the same 64 timid moves should mostly fail and $\text{asr}$ should sag well below half. The spread across scenarios will be wide and the mean modest, precisely because nothing here adapts to the model or spends the budget it was handed. That diagnosis already names the two diseases the next rung must cure: an *economic* one (quit at 65 queries when 1000 are available) and a *directional* one (unstructured interior nudges almost orthogonal to any descent direction), and the cleaner fix is to reconstruct an actual descent direction from the very queries this floor was wasting.

```python
import torch
import torch.nn as nn


def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    n_queries: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    _ = (device, n_classes)
    model.eval()

    adv_images = images.detach().clone()
    step = eps / 2.0
    n_steps = max(1, min(int(n_queries), 64))

    with torch.no_grad():
        best = model(adv_images).gather(1, labels.view(-1, 1)).squeeze(1)

        for _ in range(n_steps):
            noise = torch.empty_like(adv_images).uniform_(-step, step)
            cand = adv_images + noise
            cand = torch.clamp(images + torch.clamp(cand - images, -eps, eps), 0.0, 1.0)

            cand_score = model(cand).gather(1, labels.view(-1, 1)).squeeze(1)
            improve = cand_score < best

            if improve.any():
                mask = improve.view(-1, 1, 1, 1)
                adv_images = torch.where(mask, cand, adv_images)
                best = torch.where(improve, cand_score, best)

    delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    return torch.clamp(images + delta, 0.0, 1.0).detach()
```
