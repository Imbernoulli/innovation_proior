**Problem.** A 3D Gaussian scene fit by gradient descent on a pure photometric loss
(`0.8·L1 + 0.2·(1−SSIM)`) is under-determined: many Gaussian configurations render the training views
identically, so the optimizer settles into wasteful ones — floaters, needles, oversized splats — that
pass on training views but hurt held-out novel views. A regularizer must add the preference the data
term lacks; this rung adds none, to fix the floor.

**Key idea (the floor).** Leave `compute_regularizer` at its scaffold default: return a scalar zero, so
the loop adds nothing to the photometric loss and the Gaussians are supervised by the photographs alone.
The renderer, optimizer, photometric loss, and `DefaultStrategy` densification are all fixed; only this
function is editable, and here it is empty by construction.

**Step-1 edit.** The single editable region returns `torch.zeros(())` on the splats' device — no
penalty, no schedule, no constants.

**Why / hyperparameters.** None. This is the lower bound every regularizer must beat. It is weakest by
construction: there is no mechanism to prefer compact, well-shaped Gaussians, so the under-determined fit
keeps the pathologies the data term is indifferent to.

**What to watch.** The unbounded outdoor scenes (garden, bicycle, stump) have large weakly-observed
regions where the data term is flat and floaters/needles accumulate — most headroom for a regularizer.
The indoor scene (bonsai) is more fully observed, so the bare fit is already high and offers less slack.
That under-constraint is what forces a parameter penalty at step 2.

```python
# EDITABLE region of gsplat/custom_regularizer.py — step 1: no regularization (photometric loss only)
import torch
import torch.nn.functional as F


def compute_regularizer(splats, step, scene_scale):
    """No regularization — zero added to the photometric loss."""
    return torch.zeros((), device=splats["means"].device)
```
