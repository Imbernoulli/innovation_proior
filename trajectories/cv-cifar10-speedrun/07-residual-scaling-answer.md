**Problem (from step 6).** airbench94 is tuned to *just* clear 94% fast, with no room for the harder 96%
bar. Reaching 96% is a different problem: the binding constraint flips from time to capacity +
regularization. The small, ~10-epoch network lacks the representational room and training time for 96%, and
naively scaling width/epochs on a small dataset overfits.

**Key idea.** A capacity-and-regularization recipe (`airbench96`): (1) widen the blocks (third block 256→512)
and train ~37 epochs; (2) add a *third* conv to each block (7→10 conv layers) for compositional depth; (3)
wrap a **residual connection** over the last two convs of each block (`x + x0`) so the deeper stack stays
trainable as weights move far from the Dirac init and gradients have a structural path; (4) add **12-pixel
Cutout** augmentation and raise random translation 2→4 pixels to fight the overfitting the bigger, longer
training would otherwise suffer. Keep alternating flip (Table 2 shows it still helps at 96%).

**Why it works.** Width + extra depth add the representational room 94% lacked; the residual skip keeps the
deep stack conditioned *throughout* the long run (Dirac init only fixes the *start*, and over many epochs the
weights wander away from it — the skip earns its keep precisely here); Cutout supplies the strong, cheap
regularization a high-capacity model needs on small data. The three airbench points lie on a single log-log
FLOPs↔error line, so 96% costs more FLOPs but not pathologically more.

**Change / code.** The residual ConvGroup (three convs, skip over the last two) and the Cutout augmentation.

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
