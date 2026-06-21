airbench94 is done — 94.01% in 3.83 A100-seconds, 3.29 compiled — but the speedrun has a second, harder bar I have been ignoring, 96% mean accuracy, and reaching it is a different problem from shaving seconds off 94%. At the 94% bar the binding constraint was *time*: the network had ample capacity and I was racing to use it efficiently. At 96% the binding constraint flips to *capacity and regularization*. Two things stop airbench94 at ~94%. First, raw capacity: the network is small (blocks of 64/256/256 channels, two convs per block) and trained for ~10 epochs — tuned to *just* clear 94% fast, with no representational room or training time to push to 96%. Second, generalization: if I scale capacity and train longer, a small-data problem like CIFAR-10 starts to overfit, and the train/test gap is what holds me below 96%. So the 96% recipe has to do three things at once — add capacity, add depth, and add regularization.

I propose the **airbench96** capacity-and-regularization recipe, built from a residual-scaling architecture plus Cutout. Capacity first, the easy part: widen the blocks (the third block from 256 up to 512, the others up) and train for tens of epochs instead of ten. That is tuning, and it gets me partway, but pure width plateaus — beyond a point, more channels on the same shallow two-conv blocks stop helping because the network cannot compose features over enough nonlinear stages. So I add depth: a *third* convolution to each block, taking the network from 7 conv layers to 10. But I already know from this ladder what goes wrong when you naively deepen a conv stack — the deeper layers have to relearn to pass signal through and training conditioning degrades, the exact problem Dirac init was built to fix. I keep Dirac init, so the deep stack starts near identity, but identity init only sets the *starting* point: over a longer 96%-scale run the optimizer can walk a deep block into a poorly-conditioned configuration where gradients no longer flow cleanly back to the earlier convs. The standard structural fix is a **residual connection**: wrap a skip around a sub-stack so its output is $x + F(x)$, giving gradients a direct path back and making "do nothing" the structural default rather than merely the initial one. So I add a residual connection across the last two convs of each block — save the activation after the first conv, run the next two convs and norms, and add the saved activation back before the final activation.

Dirac init and the residual skip are not redundant, and it is worth being precise about why I keep both. Dirac init makes the block identity-like *at step zero*; the residual skip makes it identity-friendly *throughout* training and routes gradients structurally. In a short 94% run the init alone was enough and a skip would have been dead weight, which is why airbench94 has none. In a long, deep 96% run the structural skip earns its keep precisely because that is where init-only conditioning decays — as the weights move far from their init, the skip is what keeps the deeper stack trainable. The third ingredient is regularization, since added capacity, depth, and epochs will overfit. The augmentation I had (flip + 2-pixel translate) is light, appropriate for a 10-epoch run that cannot overfit much but too weak for a long one. The classic strong, cheap CIFAR augmentation is **Cutout**: mask out a random square patch of each training image (set it to zero), forcing the network to classify from partial views and not lean on any single region. I add 12-pixel Cutout and strengthen the random translation from 2 to 4 pixels to match the heavier-augmentation regime. Cutout is a per-batch masked fill — `make_random_square_masks` picks a random corner per image and builds the square mask by thresholding distance from it, and `batch_cutout` zeros it in — so it is cheap, and it attacks exactly the overfitting the bigger, longer-trained network would otherwise suffer. I keep alternating flip, which still helps in the 96% configuration, let the learning-rate schedule decay all the way to zero at the end rather than to a small floor, and shorten the warmup — both appropriate for a longer run of about 37 epochs. This is not a seconds-down rung but a capability rung: width plus a third conv per block plus residual skips plus Cutout break through the ~94% ceiling to 96% mean accuracy, and the three airbench points lie on a single log-log FLOPs$\leftrightarrow$error line, so 96% costs more FLOPs but not pathologically more.

```python
class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out); self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out); self.norm2 = BatchNorm(channels_out)
        self.conv3 = Conv(channels_out, channels_out); self.norm3 = BatchNorm(channels_out)
        self.activ = nn.GELU()
    def forward(self, x):
        x = self.conv1(x); x = self.pool(x); x = self.norm1(x); x = self.activ(x)
        x0 = x
        x = self.conv2(x); x = self.norm2(x); x = self.activ(x)
        x = self.conv3(x); x = self.norm3(x)
        x = x + x0                 # residual over the last two convs
        x = self.activ(x)
        return x

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

# hyp: widths block1/2/3 = 128/384/512, train_epochs = 37, translate = 4, cutout = 12,
#      warmup 0.1, LR decays all the way to zero at the end
```
