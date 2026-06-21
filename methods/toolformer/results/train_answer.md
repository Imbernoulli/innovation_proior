A large language model is simultaneously remarkable and embarrassingly weak. It generalizes from a handful of demonstrations or a plain instruction, yet it fails at the crisp, hard things a tiny dedicated program does perfectly: it will not add two numbers reliably, it does not know today's date, it invents facts, it cannot translate a low-resource phrase. Scaling barely helps here — a bigger model still cannot know the current date or compute arithmetic dependably, because the missing information was never frozen into its weights. The obvious remedy is to let the model call a calculator, a search engine, a translation system, a calendar, and splice the returned text back into its own generation. The genuine difficulty is not whether a call can be made — of course it can, if I hand-script it — but how the model learns, on its own, *when* a call is worth making, *which* tool, and *what arguments* to pass. The two existing roads both dead-end. Collecting human demonstrations of tool calls is expensive, and worse, it teaches what *humans* judge to be a useful call, which is not the same as what actually helps *this model* predict the next tokens. Writing a task-specific few-shot prompt ("for this task, use the calculator like so") does not generalize and presupposes I already know which tool the task needs — precisely the decision I want the model to make for itself. What I need is self-supervision: the model annotates its own data with tool calls, and an automatic criterion keeps only the annotations that demonstrably help, all without degrading general language modeling.

I propose Toolformer. The defining move is to make a tool call ordinary text. A call is a tuple $c = (a_c, i_c)$ — the tool's name and its input — and I linearize it with special marker tokens in two forms, one for the call alone and one carrying the returned result $r$ after an arrow marker:
$$e(c) = \texttt{<API>}\ a_c\,(\,i_c\,)\ \texttt{</API>}, \qquad e(c, r) = \texttt{<API>}\ a_c\,(\,i_c\,)\ \rightarrow\ r\ \texttt{</API>}.$$
In practice these markers need no new vocabulary — they can be spelled with existing tokens like `[`, `]`, `->` — but conceptually they are three special tokens. The payoff of inlining is that "use a tool" becomes "predict the right text," so an ordinary language-modeling loss can supervise it, *if* I had a corpus with helpful calls inserted at the right positions. I do not have one. The entire problem therefore reduces to manufacturing such a corpus from plain text, then finetuning on it with the standard LM objective.

The construction has three stages. First, sampling. Starting from a plain passage $x = x_1\ldots x_n$, for each tool I write one short prompt $P(x)$ containing a few in-context demonstrations of the tool being inlined into text, ending with the target input. I do not ask for a call at every position, because most positions do not want one; instead I let the model tell me where it is inclined to open a call by reading off the probability it assigns to emitting the start marker there,
$$p_i = p_M\!\left(\texttt{<API>} \mid P(x), x_{1:i-1}\right).$$
I keep positions with $p_i > \tau_s$, and if too many survive I keep the top $k_{\text{pos}}$. At each kept position $i$ I sample up to $m$ candidate calls by continuing generation from $[P(x), x_{1:i-1}, \texttt{<API>}]$ and stopping at `</API>` (a candidate that never closes is discarded). Then I execute every candidate to obtain its text result $r_i$.

The second stage, the filter, is the core of the method — it is where "self-supervised, model-defined usefulness" is made concrete. The only thing the model is ultimately judged on is predicting the upcoming tokens, so a call is useful exactly when giving the model the call *and its result* makes the following text easier to predict. I write the weighted continuation loss for a prefix $z$ as
$$L_i(z) = -\sum_{j=i}^{n} w_{j-i}\,\log p_M\!\left(x_j \mid z, x_{1:j-1}\right),$$
and I compare two prefixes. With the call and its result available, $L_i^{+} = L_i\!\left(e(c_i, r_i)\right)$. For the baseline I take the *better* of the two ways a call could fail to help — doing no call at all, or doing the call but receiving no useful result:
$$L_i^{-} = \min\!\left(L_i(\varepsilon),\; L_i\!\left(e(c_i, \varepsilon)\right)\right),$$
where $\varepsilon$ is the empty sequence. Including the second term is what stops me from rewarding a call merely for the *act* of calling: by taking the minimum, the call must earn its keep specifically through the *content* of the result, beating both "nothing" and "call-but-no-answer." I keep the call iff it reduces the loss by at least a margin $\tau_f$,
$$L_i^{-} - L_i^{+} \ge \tau_f.$$
The sign is the thing to get right: $L$ is a loss, so lower is better; $L_i^{+}$ is small when the call helps and $L_i^{-}$ is larger, so $L_i^{-} - L_i^{+}$ is large and positive exactly when the result slashed the loss, and requiring it to clear $\tau_f > 0$ keeps only calls that bought at least $\tau_f$ nats of improvement.

The weights $w_t$ encode the locality of a result's usefulness: a calculator answer helps the next few tokens, not the token forty words later, so the weighting should be largest at the call and decay to zero. I take a raw weight that ramps down linearly,
$$\tilde w_t = \max(0,\, 1 - 0.2\,t),$$
which gives $\tilde w_0 = 1, \tilde w_1 = 0.8, \ldots$ and hits $0$ at $t = 5$, spreading the call's credit over roughly the next five tokens. Its mass is $1 + 0.8 + 0.6 + 0.4 + 0.2 = 3.0$, so I normalize over future-token offsets $t = 0, 1, 2, \ldots$:
$$w_t = \tilde w_t \Big/ \sum_s \tilde w_s.$$
The $\max(0, \cdot)$ is load-bearing: it clamps distant tokens to exactly zero rather than letting the linear function go negative, which would absurdly reward making far-future tokens *less* likely. One honest subtlety in $L_i^{+}$: I prefix $e(c_i, r_i)$ to the continuation rather than splicing it mid-sentence, because the model has not yet been finetuned and has never seen an inline call, so inserting one mid-text would spike perplexity for reasons unrelated to whether the result helped. Prefixing keeps the measured continuation as natural text and isolates the question.

The third stage merges and finetunes. I gather the surviving executed calls from all tools, drop any text where every candidate was filtered out, sort the survivors of a given text by position, and *now* splice each call-with-result into the position where it was proposed — for a single kept call at $i$,
$$x^{*} = x_{1:i-1},\, e(c_i, r_i),\, x_{i:n}.$$
The quietly crucial property is that the augmented corpus $C^{*}$ is the *exact same text* as the original $C$ with API calls woven in, so finetuning $M$ on $C^{*}$ with the ordinary LM objective exposes it to the same underlying language — preserving general language modeling — while additionally teaching the pattern "at positions like this, with arguments like these, calling this tool helps predict what comes next." The mechanism is tool-agnostic: once a call is rendered as text and executed, the filter sees only token probabilities and the returned string. At inference I decode normally until the model emits the arrow marker — its way of saying "I expect a result here" — then interrupt, actually call the tool with the arguments it wrote, insert the result and `</API>`, and resume. With pure greedy decoding the model only calls when `<API>` is literally the single most likely token, which underuses tools, so I relax it to start a call whenever `<API>` is among the top $k_{\text{decode}}$ tokens ($k_{\text{decode}}=1$ recovers greedy, and I use $k_{\text{decode}}=10$); at most one call per input prevents looping.

The remaining knobs make "self-supervised" concrete. By default $\tau_s = 0.05$, $\tau_f = 1.0$, $k_{\text{pos}} = 5$, $m = 5$. For tools useful on only a tiny slice of text — the calculator and translation — I open the gates with $\tau_s = 0$, $k_{\text{pos}} = 20$, $m = 10$ to harvest enough candidates, and since that yields fewer survivors I loosen $\tau_f = 0.5$. I also pre-filter the corpus per tool with cheap heuristics (run the calculator sampler only on texts where one number is the result of an operation on two others; the calendar sampler only where the document's date is recoverable from its URL; MT only on paragraphs with a non-English chunk between English text). The tools are a QA system (Atlas, a retrieval-augmented LM), a calculator (four ops, two decimals), Wikipedia BM25 search, a 600M NLLB translation model with fastText source detection and English target, and a calendar returning the current date. The base model is 6.7B GPT-J. For the finetune I take up to 25k examples per API, cap sequences at length 1{,}024, use effective batch size 128, learning rate $1\mathrm{e}{-5}$, linear warmup over the first 10% of training, ZeRO-3 with BF16 on eight A100 40GB GPUs, up to 2k steps, and keep the checkpoint with the best held-out CCNet perplexity.

The filter is the whole method in code: run $M$ on three versions of the passage — the plain tokens, the tokens with the call but no response, and the tokens with the call and response — read off the predicted probability of each actual next token, apply the normalized decaying weights, form the three losses, and keep the passage iff $\min(\text{no-call}, \text{call-without-result})$ minus call-with-result clears the threshold.

```python
import torch
from functools import partial

def log(t, eps=1e-20):
    return torch.log(t + eps)

def default_weight_fn(t):
    # w_t = max(0, 1 - 0.2 t) / sum_s max(0, 1 - 0.2 s); denominator is 3.0
    return (1. - t * 0.2).clamp(min=0.) / 3.

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
