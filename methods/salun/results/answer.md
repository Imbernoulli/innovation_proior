# SalUn, Corrected

SalUn is a hard-masked approximate unlearning method. It first builds a binary parameter mask from the
absolute forget-gradient at the original model, then applies a bounded unlearning objective while only
the masked-in coordinates are allowed to move.

## Core Equations

Given trained weights `theta_o`, forget set `D_f`, retain set `D_r`, and forgetting loss `ell_f`,

```text
m_S = 1(abs(grad_theta ell_f(theta; D_f) | theta = theta_o) >= gamma)
theta_u = m_S * (Delta theta + theta_o) + (1 - m_S) * theta_o
```

`gamma > 0` is a global hard threshold; in practice the default mask file `with_0.5.pt` keeps the top
50% largest absolute forget-gradient coordinates. The sign used to compute the setup gradient is
immaterial because the mask uses absolute values.

For classification, the objective is

```text
min_Delta E_{(x,y) in D_f, y' != y} CE(theta_u; x, y')
        + alpha E_{(x,y) in D_r} CE(theta_u; x, y),    alpha > 0.
```

In practice the classifier implementation samples random labels from all classes (`torch.randint` or `np.random.randint`),
so it does not enforce `y' != y` in every draw. It also realizes the retain term as masked retain updates
or a randomized-forget-plus-retain concatenated dataset, rather than always as one combined minibatch.

For diffusion unlearning, the objective is

```text
min_Delta E_{(x,c) in D_f, t, eps, c' != c}
          || eps_theta_u(x_t | c') - eps_theta_u(x_t | c) ||_2^2
        + beta * ell_MSE(theta_u; D_r),    beta > 0.
```

For DDPM and Stable Diffusion I use a fixed pseudo class `(label_to_forget + 1) % 10` for class
unlearning; the NSFW path uses `"a photo of a person wearing clothes"` as the pseudo/retain prompt for
the forget prompt `"a photo of a nude person"`.

## Code-Faithful Classifier Skeleton

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

This skeleton captures the full classifier path: mask precomputation over the forget loader,
global top-ratio thresholding, random labels drawn from all classes, gradient masking before the step,
and exact post-step restoration of masked-out coordinates.
