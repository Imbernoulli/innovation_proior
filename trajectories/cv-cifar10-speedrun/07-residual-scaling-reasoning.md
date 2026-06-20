airbench94 is done: 94.01% in 3.83 seconds, 3.29 compiled. But the speedrun has a second, harder bar that
I've been ignoring — 96% mean accuracy — and the question now is not "how do I shave seconds off 94%" but
"what is the *fastest way to reach a higher accuracy at all*." That is a different problem. At 94% the
binding constraint was *time*: the network had ample capacity and I was racing to use it efficiently. At
96% the binding constraint flips to *capacity and regularization*: I need a model that can represent a
more accurate function and a training that won't overfit while it does. So I should expect to spend more
epochs and more FLOPs here, and the design goal is to spend them as efficiently as possible.

Let me diagnose what stops airbench94 at ~94%. Two things. First, raw capacity: the network is small
(blocks of 64/256/256 channels, two convs per block) and trained for ~10 epochs — it's tuned to *just*
clear 94% fast, so it simply doesn't have the representational room or the training time to push to 96%.
Second, generalization: if I scale capacity and train longer, a small-data problem like CIFAR-10 will
start to overfit, and the gap between train and test accuracy will be what holds me below 96%. So the
96% recipe has to do three things at once: add capacity, add depth, and add regularization.

Capacity first, the easy part: widen the blocks (the third block from 256 up to 512, and the others up)
and train longer (tens of epochs instead of ten). That's tuning, and it gets me partway, but pure width
plateaus — beyond a point, more channels on the same shallow two-conv blocks stop helping because the
network can't compose features over enough nonlinear stages.

So, depth. I add a *third* convolution to each block, taking the network from 7 conv layers to 10. But I
already know from my own ladder what goes wrong when you naively deepen a conv stack: the deeper layers
have to relearn to pass signal through, and training conditioning degrades — that's the exact problem
Dirac init was invented to fix, and I'm still using Dirac init, so the deep stack starts near identity.
Yet identity init only sets the *starting* point; over a longer 96%-scale training the optimizer can
still walk a deep block into a poorly-conditioned configuration where gradients don't flow cleanly to the
earlier convs. The standard structural fix for that is a *residual connection*: wrap a skip around a
sub-stack so its output is `x + F(x)`, giving gradients a direct path back and making "do nothing" the
structural default, not just the initial one. So I add a residual connection across the last two convs of
each block: save the activation after the first conv, run the next two convs+norms, and add the saved
activation back before the final activation.

There's a tension worth naming: I am *already* using identity initialization, which is itself a
"start-near-identity" device, so do I even need an explicit residual skip on top of it? They're not
redundant. Dirac init makes the block identity-like at step zero; the residual skip makes it
identity-friendly *throughout* training and routes gradients structurally. In a short 94% run the init
alone was enough and a skip would have been dead weight. In a long, deep 96% run the structural skip
earns its keep by keeping the deeper stack trainable as the weights move far from their init. I keep
both.

Now regularization, because added capacity + depth + epochs will overfit. The augmentation I have (flip +
2-pixel translate) is light — appropriate for a 10-epoch run that can't overfit much, too weak for a long
one. The classic strong, cheap augmentation for CIFAR is **Cutout**: mask out a random square patch of
each training image (set it to zero), forcing the network to classify from partial views and not lean on
any single region. I add 12-pixel Cutout, and I strengthen the random translation from 2 to 4 pixels to
match the heavier-augmentation regime. Cutout is a per-batch masked fill — cheap — and it's the right tool
because it attacks exactly the overfitting that the bigger, longer-trained network would otherwise suffer.

```python
def make_random_square_masks(inputs, size):
    n,c,h,w = inputs.shape
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)
    corner_y_dists = torch.arange(h, device=inputs.device).view(1,1,h,1) - corner_y.view(-1,1,1,1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1,1,1,w) - corner_x.view(-1,1,1,1)
    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)
    return mask_y * mask_x

def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)
```

and the residual block:

```python
def forward(self, x):
    x = self.conv1(x); x = self.pool(x); x = self.norm1(x); x = self.activ(x)
    x0 = x
    x = self.conv2(x); x = self.norm2(x); x = self.activ(x)
    x = self.conv3(x); x = self.norm3(x)
    x = x + x0           # residual over the last two convs
    x = self.activ(x)
    return x
```

I also let the learning-rate schedule decay all the way to zero at the end (rather than to a small floor)
and shorten the warmup, both appropriate for a longer run, and I keep alternating flip — which Table 2
shows still helps in the 96% configuration.

The prediction. This isn't a seconds-down rung; it's a *capability* rung — it answers a question 94%
couldn't, namely how fast I can reach 96% at all. The bet is that width + a third conv per block +
residual skips + Cutout together break through the ~94% ceiling to 96% mean accuracy, and that the FLOPs/
error tradeoff stays on the same favorable log-log line the 94 and 95 points sit on (so 96% costs more but
not pathologically more). The cost is real: tens of epochs, several × the FLOPs and wall-clock of
airbench94. The risk is that the residual skip is redundant with Dirac init and adds nothing — but the
longer training is exactly where init-only conditioning decays, so I expect the skip to earn its place.
The result is the airbench96 training; code in the answer.
