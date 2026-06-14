Let me start from the thing that actually bites when I decode one of these masked-diffusion language models. The setup is clean: the generation region starts as all `[MASK]`, the model is a mask predictor — feed it the partially-filled sequence and in one forward pass it hands me, for every masked position at once, a full categorical distribution over the vocabulary — and I have a budget of `steps` denoising iterations to turn every mask into a real token. The forward noise schedule even tells me the pace: going from time `t` down to `s`, I should unmask in expectation a fraction `1 - s/t` of whatever is still masked, so on a given step I know I'm allowed to fill, say, sixteen of the two hundred blanks. What it does not tell me — and what the training objective is completely silent about, because the cross-entropy loss only ever asks the model to predict the clean token at a masked slot, never which slots to fill in what order — is *which* sixteen. Any order that ends with everything unmasked is a legal sampler. That freedom is the whole problem, and it's not cosmetic, because committing a token here is not a private act: the attention is bidirectional, so the moment I freeze a token at position `j` it becomes conditioning context for every other masked position on every later step, forward and backward. If I'm in the regime where committed tokens never get revisited, then a token I freeze wrong is wrong forever, and worse, it actively corrupts the predictions for everything I decode after it. So the question sharpens to: out of the one-pass logits, manufacture a per-position scalar whose largest values mark the positions that are safest to commit right now, plus a token to write there. Cheap — a vector op over the logits, no extra forward passes, because the number of forward passes is literally the efficiency I'm graded on.

The dumbest thing that respects the schedule is to pick the budgeted number of masked positions uniformly at random and fill each with the model's prediction. And it's not crazy — it's actually the faithful reverse-process sampler, the one that matches the forward corruption exactly, so in expectation it's unbiased. But think about what it does step to step. It will just as happily commit a token at a position where the model has essentially no opinion — a near-uniform predicted distribution, a coin flip dressed up as a prediction — as at a position where the model is screaming a single answer. And since that committed coin-flip token then conditions the entire rest of the decode, I've injected noise into the context that everything downstream has to denoise around. Early random commitments are early mistakes, and early mistakes here are expensive. So randomness is a wall: faithful to the marginal process, but it throws away the one piece of information I have for free on every step, which is *how sure the model is at each position*. I want to commit where the model is sure and leave the uncertain positions masked, so that by the time I'm forced to decide them, more of the context has been resolved and the model is surer. That's the less-to-more intuition: get the easy, decisive positions out of the way first; let them firm up the hard ones.

Now, "commit where the model is sure" is exactly the move MaskGIT made for image-token grids, and I should lean on it because the structure is identical: predict every masked location in parallel, attach a confidence to each, keep the most confident, re-mask the rest, repeat, growing the number kept along a schedule. Their confidence is the probability of the token they end up placing — the single top probability, `conf_i = max_v P_i(v)`. And LLaDA carries the very same rule into the language setting under the name low-confidence remasking: keep the highest-probability predictions, re-mask the lowest, explicitly the same approach as MaskGIT. So the obvious first instantiation of my slot is: `conf_i = max_v P_i(v)`, unmask the positions with the largest `conf_i`, write the argmax token. Let me adopt that as a working hypothesis and then try to break it, because that's the only way I'll know if it's actually the right scalar or just the first one that came to hand.

Here's where I want to stress-test it. Picture two masked positions on the same step. At position A the model's distribution has its top token at probability 0.45 and everything else scattered tiny — the runner-up is down at 0.02. At position B the top token is also at 0.45, but the second-place token is at 0.44 — a dead heat between two candidates, with the rest of the mass negligible. Max-probability confidence scores both positions at 0.45. It calls them equally safe. But they are not remotely equally safe. At A the model is decisively committed: it has one answer and a forty-fold margin over the next. At B the model is genuinely torn between two near-identical options; whichever I freeze, there was an almost-equally-good alternative I'm permanently killing, and under greedy argmax with no revisiting that's a coin flip I've just frozen into the context. The whole point of "commit where the model is sure" was to avoid exactly B, and max-probability confidence walks straight into it, because `0.45` doesn't know the difference between "0.45 versus 0.02" and "0.45 versus 0.44." This is the wall with max-probability: it's a function of a single number, the winning mass, and the safety of a commitment plainly depends on more than that one number. The information that distinguishes A from B — the *runner-up* — is sitting right there in the logits and the rule is throwing it away.

So what's the minimal thing I have to add back? Not the whole distribution — just the piece that A and B differ on, which is the second-place probability. The honest measure of "is the top token clearly the winner here, or is it in a fight" is the *gap* between first and second:

  margin_i = P_i(top1) - P_i(top2),

the top-1 probability minus the top-2 probability at position `i`. Read it on my two cases: A gets `0.45 - 0.02 = 0.43`, B gets `0.45 - 0.44 = 0.01`. Now A scores forty-three times higher than B, which is exactly the ranking I wanted — commit A, leave B masked until its context firms up. A large gap means the top token dominates its nearest rival, so freezing it costs me nothing I'd plausibly have wanted; a small gap means a near-tie, the one case where my irreversible commit is a gamble. The scalar to maximize for "safest to commit" is the margin. And the extra cost is small: one descending sort gives me the first two probabilities needed for the score, while the token assignment can still be the ordinary argmax.

I want to make sure I'm not reinventing a thing that has a known failure mode I'm forgetting, so let me place it. This gap is precisely the quantity active learning calls *margin*: rank items by `P(y_1 | x) - P(y_2 | x)`, the spread between the two most probable classes. There it's used to find the *most uncertain* items to query a human about — you ask about small margins, the ambiguous ones — and it was introduced exactly to fix the shortcoming of the least-confident criterion, which looks only at the top class and "throws away information about the remaining label distribution." That critique is word-for-word my A-versus-B complaint about max-probability. So the margin is the standard repair for the disease I diagnosed, which is reassuring — except I have to be careful about the *direction*. In active learning you query the *smallest* margin (most ambiguous, most worth labeling). I want the opposite: I'm not asking a human to resolve uncertainty, I'm committing the positions I'm already certain about and leaving the uncertain ones alone. So I take the *largest* margin. Same scalar, opposite end. It's the certainty reading of the same quantity — best-versus-second-best, but I keep the bests, not the worsts.

Should I instead just use the full distribution and go for entropy, which uses *everything*, not only the top two? Let me actually think about what entropy does at a position. `-H_i = sum_v P_i(v) log P_i(v)`; low entropy means peaked, peaked means confident, so unmask the lowest-entropy positions. In a tiny label space that can be a reasonable confidence proxy because there is not much tail mass to account for. But my vocabulary is not tiny — it is tens of thousands of classes. Stare at the entropy sum in that regime. The overwhelming majority of the terms are tail tokens at near-zero probability, each contributing a tiny `P log P`, and there are tens of thousands of them. That huge pile of small contributions can shift the entropy as much as the contest at the very top does. Two positions could have identical top-two structure — same `P(top1)`, same `P(top2)` — and entropy could still rank them differently purely because of how the negligible mass is smeared across the long tail, which is noise I do not care about. The thing that actually predicts whether my commit is correct is whether the top token clearly beats its nearest rival; entropy dilutes that signal with tail bookkeeping. This is the many-class argument for best-versus-second-best: when there are lots of classes, the spread over the tail is uninformative relative to the gap between the two front-runners, so the two-best gap is the robust certainty signal and full entropy is the noisy one. So I don't want entropy either. Margin is the cheap, robust middle: it adds back exactly the one piece max-prob was missing (the runner-up) and ignores exactly the tail entropy gets distracted by.

Good — the position-selection signal is the margin, maximized. The token assignment is the easy half: once I've decided a position is decisive, the natural commit is its most probable token, the argmax. On a position I've specifically judged safe, I do not want to inject variance; the whole reason I chose it is that its top token dominates, so I take it.

Let me now build the actual step, because the abstraction has to become tensor ops. The model gives me logits over the whole sequence; I want probabilities, and I want them in enough precision that the top-two gap of two close masses doesn't get crushed by float rounding, so I'll softmax in float64. From `p = softmax(logits)` I get the argmax token `x0 = argmax(p)` for the assignment. For the score I sort `p` descending along the vocabulary axis and read the first two columns: `margin = sorted_p[..., 0] - sorted_p[..., 1]`. That's `P(top1) - P(top2)` at every position in one shot.

Now the selection. Only masked positions inside the current block are eligible — I must not "re-decide" a position that's already committed, and I must not reach outside the block I'm working on. The clean way to enforce that with a top-k over the whole row is to disqualify the ineligible positions by force: set their margin to `-inf` so the top-k can never pick them. Then `topk(margin, k)` with `k` = the budget for this step (`get_num_transfer_tokens` hands me how many to unmask per step from the uniform schedule) returns exactly the indices I should commit, and I write the argmax tokens there with `x = where(selected, x0, x)`. Everything else stays masked for the next iteration. One forward pass per step, one sort, one top-k, one masked write.

The block structure falls out without any new idea. If `block_length == gen_length` there's a single block and every masked position competes on every step — fully parallel decoding, which is the open-ended-text regime. Otherwise the generation region is `gen_length / block_length` blocks and I walk them left to right, and within block `b` the eligibility mask restricts the top-k to `[bs, be)`; that's the semi-autoregressive regime for the accuracy tasks. The constraint `gen_length % block_length == 0` is just so the blocks tile exactly, and I split the step budget evenly across blocks (`steps % num_blocks == 0`). The same margin rule serves both — the only thing that changes is which positions are marked eligible.

Let me write it as the decoder that fills the one empty slot in the harness — build the masked working sequence, walk the blocks, and in each step do the predict / score-by-margin / top-k-eligible / commit:

```python
import torch
import torch.nn.functional as F


class DemaskDecoder:
    """topk_margin: each step, unmask the eligible masked positions whose
    top1-prob minus top2-prob (the margin) is largest, writing the argmax token.
    Works fully-parallel (block_length == gen_length) or block-by-block."""

    def __init__(self, mask_id, temperature=0.0,
                 conf_threshold=0.9, kl_threshold=0.01, history_length=2):
        self.mask_id = mask_id
        self.temperature = temperature

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        mid = self.mask_id
        # working sequence = prompt then a fully-masked generation region
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        used = 0
        for b in range(num_blocks):                          # blocks, left to right
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(
                (x[:, bs:be] == mid), steps_per_block)        # per-step budget k
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m                 # eligible: masked & in-block
                if not mask_idx.any():
                    break
                logits = model(x).logits                      # one pass scores all blanks
                p_curr = F.softmax(logits.to(torch.float64), dim=-1)   # hi-precision probs
                x0 = torch.argmax(p_curr, dim=-1)             # token to write = argmax
                sorted_probs, _ = torch.sort(p_curr, dim=-1, descending=True)
                margin = sorted_probs[..., 0] - sorted_probs[..., 1]   # P(top1) - P(top2)
                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(margin.shape[0]):
                    m = margin[j].clone()
                    m[~mask_idx[j]] = -float("inf")           # disqualify ineligible positions
                    _, topk = torch.topk(m, int(num_xfer[j, step].item()))  # largest margins
                    xfer[j, topk] = True
                x = torch.where(xfer, x0, x)                  # commit the chosen positions
                used += 1
        return x, used                                       # used = forward passes
```

So the causal chain, end to end. The one-pass-many-positions decode leaves me a real degree of freedom the training objective never constrains — which masked positions to commit each step — and because a committed token becomes irreversible bidirectional context, getting that choice wrong is permanently expensive. Random selection is the faithful marginal sampler but discards the per-position certainty I get for free and so commits coin flips early. Max-probability confidence, the MaskGIT/LLaDA rule, uses that certainty but reads it off a single number, the winning mass, which cannot tell a decisive top token from one in a dead heat with its runner-up — and the near-tie is exactly the dangerous commit. Adding back the one missing piece, the second-place probability, gives the margin `P(top1) - P(top2)`: large when the top token clearly dominates (safe to freeze), small at a near-tie (leave masked until context resolves it). That is the active-learning margin criterion read in the certainty direction — keep the largest margins rather than query the smallest — and it beats entropy here because over a tens-of-thousands-token vocabulary the full-distribution sum is swamped by tail noise while the two-best gap stays informative, the best-versus-second-best argument for many classes. The token written is the argmax, and the selection runs as predict / softmax in high precision / margin from the top two sorted probabilities / disqualify ineligible positions with `-inf` / top-k the budget / commit — one forward pass per step, serving both fully-parallel and blockwise decoding by only changing which positions are eligible.
