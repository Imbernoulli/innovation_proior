We are handed a model already trained on a dataset $D$, and a request to remove the influence of a forgetting set $D_f$, leaving the remaining data $D_r = D \setminus D_f$. The clean reference is the model we would get by retraining the same architecture from scratch on $D_r$ alone — but that retrain is exactly what we cannot afford, which is why we want an approximate edit. A good edit has to satisfy several objectives at once: it must lose the special influence of $D_f$, keep utility on $D_r$ and on held-out test data, and not leave the forgotten examples looking so anomalous that a membership-inference attack can still single them out. No single metric captures this; the real target is closeness to the retrain reference across unlearning accuracy on $D_f$, remaining accuracy on $D_r$, test accuracy, attack behavior, and runtime.

The standard moves each fall short in a way that turns out to share a single root cause. Finetuning on $D_r$ is cheap and utility-friendly but applies no direct pressure to change predictions on $D_f$. Gradient ascent reverses the training loss on the forget set, but flipping the sign on cross-entropy makes the objective unbounded and acutely sensitive to learning rate and step count. Random labeling — relabel each forget example and train on the wrong labels — at least gives a bounded target. Influence- and curvature-based methods try to estimate the retraining displacement directly, but those estimates are brittle in deep networks and bring extra hyperparameters. The structural habit they all share is that once a loss is chosen, the update is applied to *every* parameter. The same filters, features, and classifier weights serve both $D_f$ and $D_r$, so a whole-model step that damages $D_f$ behavior also drags the features $D_r$ depends on, and a step that protects $D_r$ damps the forgetting signal. That is the familiar under-forgetting / over-forgetting tradeoff, and tuning retain terms, schedules, and regularizers only ever narrows it — the update *footprint* stays broad. The generative setting makes the same point harder to hide: for a conditional diffusion model the target is a class, object, style, or unsafe concept, and ascent or random relabeling erase too much generative ability while ordinary finetuning leaves the concept intact.

The way out is to stop merging two questions that the loss-only methods conflate. First: which parameters are actually involved in the forget-set behavior at the trained model? Second: once those are identified, what bounded objective should we run on them? I propose SalUn — saliency-based unlearning — which answers the first question with a derivative and the second with a bounded random-label (or, generatively, pseudo-conditioning) objective restricted to the salient coordinates. The first question has a cheap answer: at the trained weights $\theta_o$, differentiate a forgetting loss $\ell_f$ with respect to every parameter. For classification $\ell_f$ is ordinary cross-entropy on $D_f$; for diffusion it is the ordinary denoising MSE on the forget concept. A large absolute derivative means a small movement of that coordinate strongly changes the forget loss; a small one means the coordinate is not doing much for *this* request. The coordinate score is $\lvert \nabla_\theta \ell_f(\theta_o; D_f) \rvert$, and the sign of the loss is immaterial — an implementation may compute $-\mathrm{CE}$ or $-\mathrm{MSE}$ in gradient-ascent language while displaying positive loss, but the absolute value after accumulation is unchanged by multiplying $\ell_f$ by $-1$.

Turning scores into a hard decision gives the saliency mask, a binary entry per parameter coordinate,
$$m_S = \mathbb{1}\!\left( \left\lvert \nabla_\theta \ell_f(\theta; D_f)\big|_{\theta=\theta_o} \right\rvert \ge \gamma \right).$$
The threshold $\gamma > 0$ is not a reusable absolute number, because gradient magnitudes vary across architectures; instead we rank all absolute scores *globally* and keep a fixed top fraction — the default keeps the largest $50\%$, the mask file written as `with_0.5.pt`. The choice to rank globally rather than per layer is load-bearing: a layerwise median would force every layer to move half its weights even when the forget signal is concentrated in a few layers, defeating the purpose of a forget-specific footprint. The masked model is then
$$\theta_u = m_S \odot (\Delta\theta + \theta_o) + (1 - m_S)\odot\theta_o,$$
which makes precise that the mask is *not* pruning the forward pass — the full network still computes all features and predictions. The mask constrains the *update*: masked-in coordinates are free to move from $\theta_o$ by $\Delta\theta$, while masked-out coordinates are pinned to their original values. The direct realization multiplies each parameter's gradient by its mask entry before `optimizer.step()`. That alone is not enough, though, because with SGD momentum or weight decay a zero gradient does not guarantee a coordinate stays put — momentum can carry it, and decay shrinks it. So after the step we explicitly restore masked-out coordinates to the saved $\theta_o$ and zero their momentum buffers. This guard is what enforces the exact freezing the equation describes.

With the footprint fixed, the forgetting objective is chosen to be bounded. For classification the core is random labeling: draw a label $y' \neq y$ for each forget example, minimize cross-entropy on $D_f$ under those random labels, and add a retain cross-entropy term weighted by $\alpha > 0$,
$$\min_{\Delta\theta}\ \mathbb{E}_{(x,y)\in D_f,\, y'\neq y}\,\mathrm{CE}(\theta_u; x, y') \;+\; \alpha\,\mathbb{E}_{(x,y)\in D_r}\,\mathrm{CE}(\theta_u; x, y).$$
This beats gradient ascent because cross-entropy to a wrong-but-valid label is bounded below, so there is no runaway and far less learning-rate sensitivity; and combined with the mask, the relabeling pressure can only reach the salient coordinates, sparing the shared features that $D_r$ needs. I keep the artifact faithful to the actual reference rather than to the idealized formula: the classifier code samples random labels from *all* classes via `torch.randint` (CIFAR-10, SVHN) or `np.random.randint` (CIFAR-100, TinyImageNet), so it does not enforce $y' \neq y$ on every draw, and it realizes the retain term as masked retain updates in the same epoch (CIFAR-10, SVHN) or as training over the concatenation of randomized-forget and retain data (CIFAR-100, TinyImageNet) — both express the same pressure balance even though neither is a single combined minibatch with an explicit $\alpha$. For diffusion the random-label idea becomes pseudo-conditioning: for a forget concept $c$, make the denoiser under $c$ imitate the denoiser under another condition $c' \neq c$ on the same noised input, plus a retain denoising term weighted by $\beta > 0$,
$$\min_{\Delta\theta}\ \mathbb{E}_{(x,c)\in D_f,\,t,\,\epsilon,\,c'\neq c}\,\big\lVert \epsilon_{\theta_u}(x_t \mid c') - \epsilon_{\theta_u}(x_t \mid c) \big\rVert_2^2 \;+\; \beta\,\ell_{\mathrm{MSE}}(\theta_u; D_r).$$
The DDPM and Stable Diffusion class-removal paths fix the pseudo class to $(\texttt{label\_to\_forget}+1)\bmod 10$ and detach the pseudo output; NSFW removal uses the clothed-person prompt as both pseudo and retain prompt against the nude forget prompt. The mask is built the same way, over `model.model.diffusion_model` parameters, applied by stripping the `model.diffusion_model.` prefix from parameter names.

One shortcut I deliberately reject is computing the mask lazily from the first forget minibatch. The definition and the reference both compute the mask in a setup pass over the whole forget loader at the original weights $\theta_o$; a first-minibatch estimate is a plausible engineering approximation but it is not the method. So the recipe is: score forget gradients at the original model, keep the top fraction of coordinates as the update region, freeze the rest exactly, and run the bounded random-label or pseudo-conditioning objective with a retain term. The insight worth carrying away is that forgetting need not be made safer only by tuning the loss — it can be made safer by letting the forget loss move only the coordinates the forget set actually activates.

```python
import torch


def build_salun_mask(model, forget_loader, criterion, keep_ratio=0.5, device="cuda"):
    gradients = {
        name: torch.zeros_like(param, device="cpu")
        for name, param in model.named_parameters()
    }
    model.eval()

    for image, target in forget_loader:
        image = image.to(device)
        target = target.to(device)
        loss = -criterion(model(image), target)  # sign is irrelevant after abs()
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.detach().cpu()

    with torch.no_grad():
        gradients = {name: value.abs() for name, value in gradients.items()}
        flat = -torch.cat([value.flatten() for value in gradients.values()])
        threshold_index = int(flat.numel() * keep_ratio)
        ranks = torch.argsort(torch.argsort(flat))

        mask = {}
        start = 0
        for name, value in gradients.items():
            n = value.numel()
            local_ranks = ranks[start : start + n]
            mask[name] = (local_ranks < threshold_index).reshape(value.shape)
            start += n
        return mask


def apply_mask_to_grads(model, mask):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.mul_(mask[name].to(param.device, dtype=param.grad.dtype))


def restore_masked_params(model, mask, theta0, optimizer):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in mask:
                continue
            m = mask[name].to(param.device, dtype=param.dtype)
            param.data.mul_(m).add_(theta0[name].to(param.device) * (1 - m))
            state = optimizer.state.get(param)
            if state is not None and "momentum_buffer" in state:
                state["momentum_buffer"].mul_(m)


def salun_cifar10_epoch(model, forget_loader, retain_loader, criterion, optimizer, mask, num_classes):
    theta0 = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name in mask
    }
    model.train()

    for image, target in forget_loader:
        image = image.cuda()
        target = torch.randint(0, num_classes, target.shape, device=image.device)
        loss = criterion(model(image), target)
        optimizer.zero_grad()
        loss.backward()
        apply_mask_to_grads(model, mask)
        optimizer.step()
        restore_masked_params(model, mask, theta0, optimizer)

    for image, target in retain_loader:
        image = image.cuda()
        target = target.cuda()
        loss = criterion(model(image), target)
        optimizer.zero_grad()
        loss.backward()
        apply_mask_to_grads(model, mask)
        optimizer.step()
        restore_masked_params(model, mask, theta0, optimizer)
```
