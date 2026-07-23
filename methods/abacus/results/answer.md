# Abacus Embeddings

## Problem
Decoder-only transformers trained on short multi-digit addition fail to extrapolate to longer
numbers at test time. The bottleneck is not the arithmetic but **digit alignment**: the model
cannot reliably represent a digit's *significance* (its position relative to the start of its
own number), so it misaligns columns. Absolute positional embeddings scramble significance
(the units digit sits at different absolute indices in numbers of different length) and break
on out-of-range positions at test; relative embeddings generalize better but are blind to
where a number begins; index hints supply the columns but bloat the sequence and are
seed-brittle.

## Key idea
Index each digit by its **offset from the start of the current number**, not from the start of
the sequence, and pass that offset through a learned embedding table added to the token
embeddings. With operands written least-significant-digit-first (reversed), offset 1 = units,
offset 2 = tens, and so on, so **all digits of the same significance — across both operands
and the answer — receive the same positional vector**. This hands the model the column
structure directly, through the position channel, with no extra tokens.

To keep this absolute-style scheme from failing on long inputs (the large embedding rows would
otherwise never be trained), the start of the per-number count is **randomly shifted during
training**. In paper notation, draw an offset `β` from `{1, ..., k}` (default `k = 100`),
shared across the whole batch, and give a number's digits indices `β, β+1, β+2, ...`. The
+1 step is kept rigid (preserving significance/adjacency); only the start is randomized.
Sharing `β` across the batch keeps all same-significance digits on the same row within any
example. At test, `β = 1`.

The standalone reference code implements the same idea with a zero-based shift:
`s = random.randint(0, max_k)` is added to offsets that already start at 1. Thus standalone
`max_k = 99` corresponds to paper starts `{1, ..., 100}`. The integrated training config names
the field `max_abacus_len` and passes it as the inclusive `max_k`, so read that path with the
same convention. Over many batches this trains embedding rows beyond the training lengths, so
test positions within the sampled range are no longer out-of-distribution. The largest sampled
shift controls extrapolation reach: larger shift trains rows further out but updates each far row
less often.

This is an *absolute* embedding applied only to numeric tokens, computed in linear time, so it
composes with a relative embedding (FIRE / RoPE) handling non-numeric tokens in a general
model. On addition it lifts a small from-scratch model (trained on ≤20-digit operands, one GPU
for a day) to strong length generalization, and the same signal transfers to multiplication
and sorting.

## Algorithm
For each sequence, with `digit_tokens` the set of digit token ids:
1. `mask[t] = 1` iff token `t` is a digit.
2. Within each maximal run of consecutive digits, assign 1-based offsets `1, 2, 3, …`
   (run-starts detected where `mask` rises from 0 to 1; offsets via cumulative segment ids and
   gather of the run-start index). Non-digit tokens get index 0 (a reserved row).
3. Training only: draw `shift ~ U{0, ..., max_k}` once per batch and add it to every positive
   offset (digits become `shift+1, shift+2, ...`; offsets already start at 1). Test: add 0.
4. Look up the indices in a learned embedding table and add the result to the token embeddings
   before the first decoder layer.

## Reference implementation
```python
import random
import torch


class Abacus(torch.nn.Module):
    """Abacus Embeddings: learned positional embeddings indexed by a digit's offset
    from the start of its own number, reused for every number. Integers must be
    written reversed (least-significant digit first) for offsets to equal significance.
    """

    def __init__(self, digit_tokens, embedding_dim, max_seq_length=1024, max_k=99):
        """
        digit_tokens (list): token ids of the ten digits '0'..'9'.
        embedding_dim (int): embedding dimension.
        max_seq_length (int): number of trainable positional rows (must exceed
            longest test offset + max_k); row 0 is reserved for non-digit tokens.
        max_k (int): the training shift is drawn from U{0..max_k};
            standalone max_k=99 gives paper starts 1..100.
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)
        self.max_k = max_k

    def helper(self, mask, device):
        """Convert a binary digit-mask into per-number 1-based offsets (0 elsewhere)."""
        mask_shape = mask.shape
        # detect run starts: mask is 1 here but was 0 at the previous position
        shifted_mask = torch.cat(
            [torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype),
             mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask

        # segment id increments at each new run, constant within a run
        segment_ids = torch.cumsum(starts, dim=1)
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)

        # reset_index[segment] = absolute index where that run started.
        # (a one-token all-digit tensor would need one extra bin)
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)

        # offset within the run = index - run-start index + 1, then zero non-digits
        positions = index - reset_index.gather(1, segment_ids) + 1
        result = positions * mask
        return result

    def forward(self, input_ids):
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)

        k = 0
        if self.training:
            k = random.randint(0, self.max_k)   # one shared shift for the whole batch
            output[output > 0] += k             # shift digit offsets only (>= 1)

        return self.embedding(output)           # add onto the token embeddings
```

## Notes
- **Reverse the integers** (LSD first) so offset = significance and generation order matches
  carry propagation.
- **Composable**: applies only to digit tokens, so it can be added alongside a relative scheme
  (FIRE/RoPE) for the rest of a general model; the offset computation is linear-time.
- **Bounded extrapolation**: this is an absolute embedding, so it cannot use rows beyond the
  largest trained (`longest digit run + largest sampled zero-based shift`). Increase the shift
  range to extend reach.
- **Edge case**: the scatter helper above is faithful for ordinary arithmetic
  sequences, including multiple digit runs and non-digit gaps. A degenerate one-token all-digit
  tensor would need an extra reset bin in a hardened standalone utility.
- Pairs naturally with input injection and looped (weight-tied) transformer layers, which
  further reduce out-of-distribution error once positions are resolved.
