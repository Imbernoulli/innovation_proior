## Task And Gold Standard

A vision model has already been trained on a dataset `D`. A request arrives with a forgetting set
`D_f`, and the remaining data are `D_r = D \ D_f`. The clean reference behavior is the model obtained by
training the same architecture from scratch on `D_r`; that reference is expensive, but it defines what a
successful edit is trying to approximate. The new model should lose the special influence of `D_f`,
keep useful behavior on `D_r` and test data, and avoid making the forgotten examples look so unusual
that a privacy attack can still recognize them.

The practical evaluation is therefore multi-objective. Classification work usually reports unlearning
accuracy on `D_f`, remaining accuracy on `D_r`, test accuracy, membership-inference attack behavior, and
runtime. No single metric is enough: a model can destroy forget-set accuracy by damaging everything, or
keep utility by barely forgetting. The useful target is closeness to the retrain reference across all of
these views.

## Classification Baselines

Finetuning on `D_r` is cheap and utility friendly, but it has no direct pressure to change predictions
on `D_f`. Random labeling gives direct bounded pressure: replace the labels of the forget examples with
arbitrary labels and train on those relabeled examples. Gradient ascent reverses the ordinary training
loss on the forget set, but its sign makes the objective unbounded and sensitive to learning-rate and
step-count choices. Influence-style methods try to approximate the retraining displacement from
gradients or curvature, but their estimates are brittle in deep networks and require extra
hyperparameters.

These methods share a structural habit: once they decide on a loss, they apply the update to the whole
model. Retain terms, schedules, and regularizers can reduce collateral damage, but the update footprint
itself is still broad.

## Why The Whole-Model Footprint Is The Bottleneck

The same filters, features, and classifier weights serve examples from both `D_f` and `D_r`. A whole
parameter update that makes the model worse on `D_f` can also move features needed by `D_r`; a whole
parameter update that preserves `D_r` can damp the forgetting signal. This explains the familiar
under-forgetting and over-forgetting tradeoff in approximate unlearning.

There is relevant prior structure but no complete answer yet. Input-gradient explanation methods show
that a derivative can localize which input coordinates affect a decision. Pruning and sparsity work
show that only some parameters are important for a network's behavior or efficiency. Sparse-unlearning
work shows that model-side structure can help approximate unlearning. What is missing is a dense-model,
forget-set-specific way to decide which coordinates an unlearning update should be allowed to move.

## Generative Unlearning Raises The Same Issue

Conditional diffusion models make the same problem harder to hide. The target may be a class, object,
style, or unsafe concept rather than a labeled training subset. A successful edit should stop the model
from generating the unwanted concept under its prompt while preserving generation quality for normal
prompts. Existing classification moves do not transfer cleanly: ascent and random relabeling can erase
too much generative ability, while ordinary finetuning can leave the unwanted concept intact.

The diffusion training loss is an MSE on predicted noise at sampled time steps. Any unlearning rule in
this setting has to decide both what alternative conditioning signal should replace the forgotten
concept and how to protect the model's remaining denoising ability.

## Implementation Frame

The available implementation slot starts from a trained model, a forget loader, a retain loader, a loss,
and an optimizer. It may run a setup pass before unlearning, and then it performs a small number of
unlearning epochs or iterations. The slot must specify three things: the forgetting loss, the retain
loss, and whether every parameter update is applied everywhere or only to selected coordinates.

```python
def unlearn(model, forget_loader, retain_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for images, labels in forget_loader:
            optimizer.zero_grad()
            loss = ...  # forgetting-side objective
            loss.backward()
            optimizer.step()

        for images, labels in retain_loader:
            optimizer.zero_grad()
            loss = ...  # retain-side objective
            loss.backward()
            optimizer.step()
```

The unresolved design question is how to make this update forget-specific without turning the whole
network into collateral damage.
