The task is to make a small decoder-only transformer generalize from short multi-digit addition to much longer numbers. When I train a standard model on operands up to twenty digits and test it on one-hundred-digit numbers, the failure is sharp but not random: the outputs are shifted by a place, or have one digit too many, as if the model were adding the unit digit of one operand to the tens digit of the other. Single-digit addition with carry is easy; the model is failing at bookkeeping. It cannot keep the columns aligned, because standard positional encodings do not give it a stable handle on a digit's significance.

Absolute positional embeddings are the worst tool here. The unit digit of a five-digit number and the unit digit of a fifteen-digit number sit at completely different absolute indices, so absolute position scrambles significance even in-distribution. Beyond the training length the embedding rows are untrained and the model collapses. Relative schemes such as RoPE, ALiBi, or FIRE generalize better because they depend only on query-key distance, but they are still computed over the whole sequence and have no notion of where a number begins; the model must still infer that two digits share a column. Index hints can force alignment by inserting shared labels like a6b7c5, but they roughly double the sequence length and the reported generalization is brittle across random seeds. What is needed is a way to hand the model the column structure through the position channel itself, without extra tokens.

The method is Abacus Embeddings. The core idea is to index each digit not by its position in the whole sequence, but by its offset from the start of its own number. With operands written least-significant-digit-first, as is already done so that generation order matches carry propagation, offset one is the units digit, offset two is the tens digit, and so on. Every digit of the same significance, across both operands and the answer, therefore receives exactly the same learned positional vector. The transformer can attend "unit to unit" and "ten to ten" by matching positional vectors, which is precisely the human column-stacking prior built into the input representation. The signal is added to the token embedding before the first layer, exactly like a learned absolute embedding, except that the index is per-number rather than per-sequence.

This per-number counter fixes the alignment problem, but by itself it still walks into the coverage problem that sinks absolute embeddings: a hundred-digit number needs offsets up to a hundred, and if the model trains only on twenty-digit numbers then the rows for offsets above twenty are never updated. The fix is to randomize the starting point of the count during training. Once per batch, draw a shift uniformly from zero to a chosen maximum, and add that same shift to every positive offset. The within-number step stays plus one, so adjacency and significance are preserved. Sharing the shift across the whole batch is load-bearing: if different numbers in the same example started at different shifts, their unit digits would no longer share an embedding row and the column-alignment property would break. With one shared shift, every number in a batch starts at the same offset, so same-significance digits still match within every example, while over many batches embedding rows far beyond the training length get exercised. At evaluation the shift is set to zero, so the digits use the most-trained rows one, two, three, and so on.

The largest sampled shift bounds the extrapolation reach: rows up to roughly the longest training number plus the maximum shift are trained, and rows beyond that remain untrained. This is extended coverage, not unbounded extrapolation, but it is enough to move from twenty-digit training to hundred-digit testing with a modest increase in the shift range. The offset computation is linear-time and applies only to digit tokens, so it composes cleanly with a relative positional scheme such as FIRE or RoPE that handles the rest of the vocabulary. It also transfers beyond addition: the same within-run offset idea helps multiplication, sorting, and similar alignment tasks where the latent variable is position inside a contiguous unit.

```python
import random
import torch
import torch.nn as nn


class Abacus(nn.Module):
    """Abacus Embeddings: learned positional embeddings indexed by a digit's
    offset from the start of its own number. Operands are assumed reversed
    (least-significant digit first) so that offset equals significance."""

    def __init__(self, digit_tokens, embedding_dim, max_seq_length=1024, max_k=99):
        super().__init__()
        # Row 0 is reserved for non-digit tokens; rows 1..max_seq_length-1
        # hold the within-number offsets.
        self.embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)
        self.max_k = max_k

    def helper(self, mask, device):
        # mask[b, t] == 1 means token t is a digit. Convert each maximal run
        # of consecutive digits into 1, 2, 3, ... and leave everything else 0.
        mask_shape = mask.shape
        shifted_mask = torch.cat(
            [torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype),
             mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask

        segment_ids = torch.cumsum(starts, dim=1)
        index = torch.arange(mask.size(1), device=device).unsqueeze(0).expand_as(mask)

        reset_index = torch.zeros_like(mask, dtype=torch.long)
        reset_index = reset_index.scatter_add(1, segment_ids, index * starts.long())

        positions = index - reset_index.gather(1, segment_ids) + 1
        return positions * mask

    def forward(self, input_ids):
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)

        if self.training:
            shift = random.randint(0, self.max_k)
            output[output > 0] += shift

        return self.embedding(output)
```
