# Toolformer

## Problem

LMs excel at soft language tasks but fail at crisp ones that small dedicated programs handle perfectly: precise arithmetic, current facts, the date, factual lookup, low-resource translation. The fix is to let the model call external tools and weave their results into its text — but it must learn *when* to call, *which* tool, and *what arguments* on its own, with almost no human annotation, and without losing its general language-modeling ability.

## Key idea

Self-supervised tool learning by self-annotation + perplexity-based filtering + plain-LM finetuning. A tool call is a tuple c = (a_c, i_c) (name, input) linearized into text with special markers:

  e(c)    = `<API>` a_c `(` i_c `)` `</API>`
  e(c, r) = `<API>` a_c `(` i_c `)` → r `</API>`

Three stages turn a plain corpus into a corpus with helpful calls inlined:

1. **Sample.** For each tool, prompt M with a few in-context demonstrations P(x). Find positions where M wants to open a call: p_i = p_M(`<API>` | P(x), x_{1:i-1}); keep positions with p_i > τ_s (top k). At each, sample up to m candidate calls by continuing from `<API>` to `</API>`. Execute them to get results r_i.

2. **Filter** (the core contribution). With weights w_t = w̃_t / Σ_s w̃_s, w̃_t = max(0, 1 − 0.2·t) (full credit at the call, decaying to 0 by t = 5), define the weighted continuation loss
   L_i(z) = − Σ_{j=i}^{n} w_{j−i} · log p_M(x_j | z, x_{1:j-1}),
   and compare
   L_i^+ = L_i(e(c_i, r_i))   (call **and** result given)
   L_i^− = min( L_i(ε), L_i(e(c_i, ε)) )   (best of: no call / call without result).
   Keep the call iff **L_i^− − L_i^+ ≥ τ_f** — i.e. having the *result* cuts the next-token loss, beyond both not calling and calling without a result, by margin τ_f. (Loss is computed with the call as a *prefix*, not spliced mid-text, because M is not yet finetuned on inline calls.)

3. **Finetune.** Splice each surviving call into its position, x* = x_{1:i-1}, e(c_i, r_i), x_{i:n}, giving an augmented corpus C* that is the original text with calls inserted. Finetune M on C* with the standard LM objective: same content as C (so generality is preserved) plus the learned "call here helps."

**Inference.** Decode normally until M emits →; interrupt, run the tool, insert the result and `</API>`, resume. Start a call when `<API>` is among the top-k most likely tokens (k > 1 makes M more willing to use tools), at most one call per input.

Defaults: τ_s = 0.05, τ_f = 1.0, k = 5, m = 5; for sparse-use tools (calculator, MT) τ_s = 0, k = 20, m = 10, τ_f = 0.5. Tools: QA (retrieval-augmented LM), calculator (four ops, 2 decimals), Wikipedia BM25 search, MT (multilingual, target English, auto source detection), calendar (current date). Base model: a ~6.7B GPT-J; finetuned with batch size 128, learning rate 1e-5, linear warmup over the first 10% of training.

## Code

The filter is the whole method: run M on the plain tokens, the tokens-with-call-no-response, and the tokens-with-call-and-response; weight next-token log-probs with the decaying weights; keep the passage iff min(no-call, call-without-result) loss minus call-with-result loss clears the threshold.

```python
import torch
from functools import partial

def log(t, eps=1e-20):
    return torch.log(t + eps)

def default_weight_fn(t):
    # w~_t = max(0, 1 - 0.2 t): full weight at the call, linearly to 0 by t = 5
    return (1. - t * 0.2).clamp(min=0.)

def get_pred_prob(token_ids, logits):
    logits = logits[:, :-1]
    token_ids = token_ids[:, 1:]
    probs = logits.softmax(dim=-1)
    return probs.gather(-1, token_ids[..., None]).squeeze(-1)

def get_arange_start_at_token_id(token_ids, token_id, pad_id=-1):
    is_mask = token_ids == token_id
    arange = (is_mask.cumsum(dim=-1) > 0).cumsum(dim=-1)   # 0 before marker, 1,2,... after
    before = arange == 0
    arange = arange - 1                                   # -> -1 before, 0,1,2,... after
    return arange.masked_fill(before, pad_id)

def weight_and_mask(token_ids, token_id, pad_id=-1, weighting_fn=default_weight_fn):
    t = get_arange_start_at_token_id(token_ids, token_id, pad_id)
    weights = weighting_fn(t)
    return weights.masked_fill(t == pad_id, 0.)

def filter_tokens_with_api_response(
    model, *, tokens, tokens_without_api_response, tokens_with_api_response,
    api_start_token_id, api_end_token_id, filter_threshold=1.,
    weighting_fn=default_weight_fn):

    with torch.no_grad():
        model.eval()
        logits, logits_wo, logits_w = map(
            model, (tokens, tokens_without_api_response, tokens_with_api_response))

    probs    = get_pred_prob(tokens, logits)                          # no call at all
    probs_wo = get_pred_prob(tokens_without_api_response, logits_wo)  # call, no response
    probs_w  = get_pred_prob(tokens_with_api_response,  logits_w)     # call + response

    wm = partial(weight_and_mask, weighting_fn=weighting_fn)
    weight_wo = wm(tokens_without_api_response[:, :-1], api_end_token_id)
    weight_w  = wm(tokens_with_api_response[:, :-1],  api_end_token_id)
    weight    = wm(tokens_without_api_response[:, 1:], api_start_token_id)[:, :probs.shape[-1]]

    def loss_fn(weight, probs):                # L_i(z) = sum w * -log p
        return (weight * -log(probs)).sum(dim=-1)

    loss    = loss_fn(weight,    probs)         # L_i(epsilon)
    loss_wo = loss_fn(weight_wo, probs_wo)      # L_i(e(c, epsilon))
    loss_w  = loss_fn(weight_w,  probs_w)       # L_i(e(c, r))

    loss_plus  = loss_w                         # L_i^+
    loss_minus = torch.minimum(loss_wo, loss)   # L_i^- = min(no-call, call-without-result)

    selected_mask = (loss_minus - loss_plus) >= filter_threshold   # keep iff helped by >= tau_f
    return selected_mask
```

An example QA-tool sampling prompt P(x): a few demonstrations of inlining calls, ending with the target text.

```
Your task is to add calls to a Question Answering API to a piece of text. The questions
should help you get information required to complete the text. You can call the API by
writing "[QA(question)]" where "question" is the question you want to ask. Here are some
examples of API calls:
Input: Joe Biden was born in Scranton, Pennsylvania.
Output: Joe Biden was born in [QA("Where was Joe Biden born?")] Scranton, [QA("In which
state is Scranton?")] Pennsylvania.

Input: x
Output:
```
