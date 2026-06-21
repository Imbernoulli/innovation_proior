A masked diffusion language model only gives us one primitive: feed it a partially masked sequence and it returns, in a single forward pass, a categorical distribution over the vocabulary for every masked position. Generation is the reverse process that starts from a fully masked response and gradually unmasks it. The difficulty is that the exact reverse kernel must only fill a schedule-fixed fraction of the masked positions at each step; committing too many at once breaks the tokenwise-independence approximation and degrades the joint sample, while committed tokens are frozen forever by the carry-over property of absorbing-state diffusion. So the practical decoder has to decide, after every forward pass, which masked positions to fill and what token to write.

The obvious baselines fall short. Random remasking preserves the intended masking rate by choosing surviving positions uniformly, but it ignores everything the predictor already produced. It can just as easily freeze a low-confidence guess as a high-confidence one, and those early errors poison the context for every later step. Confidence-keeping iterative decoding from masked image transformers uses the same intuition of keeping the most confident predictions, but it was tuned for image synthesis: it samples tokens stochastically with temperature annealing and follows a cosine remasking schedule, neither of which is right for a likelihood-trained masked diffusion language model with a linear schedule and a preference for deterministic accuracy on math and code. What is needed is a version that respects the linear schedule, uses greedy token assignment by default, and still commits the positions the model is most sure about.

The method is confidence-greedy decoding, also called low-confidence remasking. At each reverse step we run the predictor once to obtain logits for every position. We choose a token at each masked position by taking the argmax of the logits, which is greedy decoding at temperature zero. The confidence of that choice is the model's own probability mass on the selected token, computed from a softmax over the original logits. We then commit the top-k masked positions in the active block by that confidence, where k is the uniform per-step budget dictated by the linear masking schedule, and leave every other masked position untouched for the next step. Because already-unmasked and out-of-block positions are assigned negative-infinite confidence, they are never selected again. Repeating this process gives an easy-first curriculum: the model locks in the tokens its current context already determines, those tokens then sharpen the context for the remaining ambiguous positions, and the next step commits the next-most-confident subset.

This rule is motivated by two sources of error in the fast decoder. First, carry-over makes every commitment irreversible, so the cost of a mistake is the same wherever it happens, but the model-assigned probability of error is smallest where the distribution is peaked. Committing the highest-confidence positions therefore minimizes the chance of freezing a wrong token. Second, the parallel step treats proposals at different positions as conditionally independent given the current context, which is only an approximation to the true joint conditional. The perturbation caused by committing one position is smallest when that position's selected token already carries most of the probability mass, which is exactly the high-confidence case. Greedy token assignment reinforces the same accuracy-first behavior: suppressing diversity is harmful for open-ended generation but helpful when there is a single correct answer. A temperature knob remains available for the open-ended setting by adding Gumbel noise before the argmax, while the confidence itself is still read from the original unnoised logits. The decoding can run fully parallel when the block spans the whole response, or semi-autoregressively in left-to-right blocks for long structured outputs, with the total step budget divided evenly across blocks. One practical refinement is to suppress the end-of-sequence token's confidence for instruction-tuned models that were fine-tuned with heavy padding, preventing premature termination.

```python
import torch
import torch.nn.functional as F


def get_num_transfer_tokens(mask_index, steps):
    """Linear noise schedule -> equal expected number of tokens unmasked per step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, :remainder[i]] += 1
    return num


def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling; temperature == 0 reduces to plain argmax."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class DemaskDecoder:
    """Confidence-greedy / low-confidence remasking decoder for a masked diffusion LM."""

    def __init__(self, mask_id, temperature=0.0, eos_id=None, suppress_eos=False):
        self.mask_id = mask_id
        self.temperature = temperature
        self.eos_id = eos_id
        self.suppress_eos = suppress_eos

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        mid = self.mask_id
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length  # == 1 -> fully parallel
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        used = 0
        for b in range(num_blocks):  # blocks left-to-right
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens((x[:, bs:be] == mid), steps_per_block)
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m  # masks in the current block only
                if not mask_idx.any():
                    break

                logits = model(x).logits  # one forward pass: all positions
                logits_noised = add_gumbel_noise(logits, self.temperature)
                x0 = torch.argmax(logits_noised, dim=-1)  # greedy or sampled token

                p = F.softmax(logits, dim=-1)
                conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)  # prob of selected token
                if self.suppress_eos and self.eos_id is not None:
                    conf = torch.where(x0 == self.eos_id,
                                       torch.full_like(conf, -float("inf")), conf)

                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(conf.shape[0]):
                    c = conf[j].clone()
                    c[~mask_idx[j]] = -float("inf")  # frozen / out-of-block: never selected
                    _, topk = torch.topk(c, int(num_xfer[j, step].item()))  # most-confident k
                    xfer[j, topk] = True
                x = torch.where(xfer, x0, x)  # commit; carry-over keeps the rest masked
                used += 1
        return x, used
```
