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

Finetuning on `D_r` is cheap and utility friendly. Random labeling gives direct bounded pressure:
replace the labels of the forget examples with arbitrary labels and train on those relabeled examples.
Gradient ascent reverses the ordinary training loss on the forget set. Influence-style methods try to
approximate the retraining displacement from gradients or curvature.

These methods apply updates to the whole model. Retain terms, schedules, and regularizers can modulate
the update, and the update footprint spans all parameters.

## Generative Unlearning

Conditional diffusion models present a related problem. The target may be a class, object, style, or
unsafe concept rather than a labeled training subset. A successful edit should stop the model from
generating the unwanted concept under its prompt while preserving generation quality for normal prompts.

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
