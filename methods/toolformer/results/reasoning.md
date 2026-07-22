Let me be precise about what I'm trying to fix, because the failure is so specific it almost dictates the solution. The model is brilliant at the soft stuff and terrible at the hard, crisp stuff: it won't add two numbers reliably, it doesn't know today's date, it invents facts, it can't translate a low-resource phrase. Every one of these is a thing a tiny dedicated program does perfectly — a calculator, a clock, a lookup, a translation model. So I don't need to make the LM better at arithmetic. I need to let it *ask the calculator* and weave the answer back into its text. The model should be a generalist that knows when to phone a specialist.

The deep question isn't "can it call a tool" — of course it can, if I hand-script the call. The question is how it learns, on its own, *when* a call is worth making, *which* tool, and *what arguments*, without me annotating mountains of examples. Two existing roads both look unappealing for me. Road one: collect tons of human demonstrations of tool calls. Expensive, and worse, it teaches the model what *humans* think is a useful call, which is not the same as what actually helps *this model* predict the next tokens. Road two: write a task-specific prompt that says "for this task, use the calculator like so." That doesn't generalize and it presupposes I already know which tool the task needs — exactly the decision I want the model to make. So I want self-supervision: the model annotates its own data, and some automatic criterion keeps only the good annotations. That criterion is the part I don't have yet, and I suspect it's where most of the difficulty lives.

Can a tool call even live inside text? It can, as long as the tool's input and output are both text. Then I can represent a call as a tuple c = (a_c, i_c), the tool name and its input, and linearize it with special marker tokens. Two forms — without and with the result r:

  e(c)    = <API> a_c ( i_c ) </API>
  e(c, r) = <API> a_c ( i_c ) → r </API>

so e(c) is "I'm calling tool a_c on input i_c," and e(c,r) additionally carries the returned result r after a → marker. (In practice I don't even need new vocabulary — I can spell <API>, </API>, → with existing tokens like "[", "]", "->". But conceptually they're three special tokens.) The point of inlining is that now a tool call is just *text*, so an ordinary language-modeling loss can supervise it — if only I had training sequences with calls inserted in the right places. I don't have those. The whole problem reduces to one thing: manufacture a corpus where helpful API calls are inlined at the right positions, then finetune on it with the plain LM objective.

Start with a plain text x = x_1…x_n from an ordinary corpus. For each tool I write one short prompt P(x) — a handful of in-context examples showing the tool being inserted into text, ending with "Input: x / Output:". The model's in-context-learning does the rest: it'll propose where and how to call the tool in x. But I shouldn't ask for a call at every position; most positions don't want one. So I let the model tell me where it *wants* to start a call. For each position i, I read off the probability the model assigns to emitting the start marker right there:

  p_i = p_M(<API> | P(x), x_{1:i-1}).

These are the positions the model itself is inclined to open a call. I threshold them at τ_s, keep positions with p_i > τ_s, and if too many survive I keep the top k. Then at each kept position i, I sample actual call candidates c_i^1…c_i^m by continuing generation from [P(x), x_{1:i-1}, <API>] and stopping at </API>. (If the model never closes with </API>, I throw that candidate away.) Now I have, scattered through x, up to k positions each with up to m candidate calls.

Execute them — run each calculator/QA/search/MT/calendar call to get its text result r_i. Fine. Now the crux: which of these calls do I keep? This is the filter, and it's where the whole "self-supervised, model-defined usefulness" idea has to be made concrete. I need an automatic score that says "this call helped." Helped *with what*? With the only thing the model is ultimately judged on: predicting the upcoming tokens x_i, x_{i+1}, … . So a natural criterion is: does giving the model the call *and its result* make the following text easier to predict than not having it?

Let me write that as a loss. The natural quantity is the cross-entropy the model pays on the continuation x_i…x_n when it's prefixed with some sequence z:

  L_i(z) = − Σ_{j=i}^{n} w_{j−i} · log p_M(x_j | z, x_{1:j-1}).

The w's are weights I'll pin down in a second; for now it's a weighted negative log-likelihood of the tokens from position i onward. I compare two prefixes. With the call and its result available:

  L_i^+ = L_i( e(c_i, r_i) ).

And the baseline — what the loss is if the call *doesn't* help. My first instinct is to make the baseline simply "no call at all," L_i(ε), and keep the call when L_i(ε) − L_i^+ is large. But that obvious choice has a hole.

Imagine a position right after a number where the model has learned, from the in-context prompt, that calculator calls *tend* to appear. Even an empty or useless call — "<API> Calculator(2 + 2) </API>" with no answer, in a spot that has nothing to do with 2+2 — changes the local distribution: the model now expects the surrounding tokens to be the kind of arithmetic-flavored text that follows calls, and that alone can nudge log p of the continuation up a bit. With baseline L_i(ε), that nudge counts as "the call helped," even though the *result* contributed nothing. So a pure-L_i(ε) baseline would reward the model for the mere *act* of calling, independent of whether the returned string was any good. That's exactly backwards from what I want: I want to credit the content r_i, not the gesture.

So the baseline has to neutralize the gesture. There are two ways the call could fail to help: I could do no call at all, or I could do the call but get no useful result. Take the *better* (lower-loss) of those two as the bar to beat:

  L_i^− = min( L_i(ε), L_i( e(c_i, ε) ) ),

where ε is the empty sequence. The first term L_i(ε) is "predict the continuation with no API call." The second L_i(e(c_i, ε)) is "the model saw that a call was made but received no response." By taking the min, I force the call to earn its keep specifically through the *content of the result*: e(c_i, r_i) has to beat both "nothing" and "call-but-no-answer." A call is genuinely useful only if the result r_i itself made the future tokens easier. This is a strictly harder bar than my first-instinct baseline — min(L_i(ε), L_i(e(c_i,ε))) ≤ L_i(ε) always — so it can only reject more calls, never accept ones the simpler version would have rejected. That asymmetry is the right side to err on; a corpus full of cosmetic calls would teach the model to call uselessly.

Then keep the call iff it reduces loss by at least a margin τ_f:

  L_i^− − L_i^+ ≥ τ_f.

Concretely: at the position in "...a passing rate of " with continuation "29", no-call spreads mass thin — say p(29 | ε, …) = 0.1, −log p ≈ 2.30. Call-without-result doesn't supply the digits either, p ≈ 0.12, −log p ≈ 2.12. Call-with-result, where the calculator returned 29.0, puts the answer right in context: p(29 | e(c,r), …) = 0.7, −log p ≈ 0.36. Then L_i^+ ≈ 0.36, L_i^− = min(2.30, 2.12) = 2.12, and L_i^− − L_i^+ ≈ 1.76 ≥ τ_f = 1.0 — kept. If the result hadn't actually helped, p-with-result would sit near 0.1 too, L_i^+ ≈ 2.30, the difference ≈ −0.2, and the call is dropped — exactly the case the min-baseline exists to catch, since a no-result call at −log p ≈ 2.12 is nearly indistinguishable from no call at all.

Now those weights w. The information a call returns is useful for predicting tokens *near* the call, and its relevance should fade as I move further past it — a calculator result helps the next few tokens, not the token forty words later. So I want a weighting that's largest right at the call and decays to zero. Take a raw weight that ramps down linearly,

  w̃_t = max(0, 1 − 0.2·t),

so w̃_0 = 1, w̃_1 = 0.8, … and it hits 0 at t = 5: the call's "credit" spans roughly the next five tokens. Let me actually total the raw schedule: w̃_0 + w̃_1 + w̃_2 + w̃_3 + w̃_4 = 1 + 0.8 + 0.6 + 0.4 + 0.2 = 3.0, and w̃_5 = max(0, 1 − 1.0) = 0, w̃_6 = max(0, 1 − 1.2) = 0, and everything beyond is 0 too, so the finite mass is exactly 3.0. I normalize over future-token offsets t = 0, 1, 2, …:

  w_t = w̃_t / Σ_s w̃_s = w̃_t / 3.0,

giving w = (0.333, 0.267, 0.200, 0.133, 0.067, 0, 0, …), which sums to 1 — so L_i is a genuine weighted average of next-token surprisals, comparable across positions. The max(0, ·) is the load-bearing part: without it the linear weight would go negative at t ≥ 5 (1 − 0.2·6 = −0.2), and a negative weight on −log p means the filter would *reward* making a far-future token *less* likely — absurd. The clamp pins distant tokens to exactly zero instead. So the filter scores a call almost entirely by its effect on the immediately following few tokens (the t = 0,1,2 offsets alone carry 0.333 + 0.267 + 0.200 = 0.8 of the total weight), which is the locality I wanted.

There's a subtlety in L_i^+ I should be honest about: when I compute the loss with the call, I prefix e(c_i, r_i) to the continuation rather than splicing it into the middle of x at position i. Why? Because the model hasn't been finetuned yet — it has never seen an API call mid-sentence — so inserting one inside x would interrupt the natural flow and spike perplexity for reasons that have nothing to do with whether the result is useful. Prefixing keeps the measured continuation as natural text and isolates the question "did the result help."

After filtering, I merge the surviving executed calls from all tools. If a text has several survivors, I sort them by position and *now* splice each call-with-result into the text at the position where it was proposed. For x with one kept call (c_i, r_i) at i, the local form is

  x* = x_{1:i-1}, e(c_i, r_i), x_{i:n}.

Do this over the corpus, drop texts where every candidate call was filtered out, and I get an augmented corpus C*. What C* actually is, token for token, is what determines whether finetuning on it will wreck plain language modeling. Take any text x and its augmented version x*. Delete the spans between (and including) every <API> and </API> from x* — that removes exactly e(c_i, r_i) at each spliced position and leaves x_{1:i-1} followed by x_{i:n}, i.e. the original x verbatim. So C* is C with API spans interleaved and *nothing else changed*: same words, same order, same content, just calls woven in at positions where their executed results demonstrably helped. That's reassuring — finetuning M on C* with the ordinary LM objective exposes it to the same underlying text it would have seen from C (so general language modeling should be preserved) while additionally teaching the pattern "at positions like this, with arguments like these, calling this tool helps me predict what comes next." And the mechanism stayed tool-agnostic the whole way: after the call has been rendered as text and executed, the filter only ever sees token probabilities and the returned string — it never inspects what kind of tool produced r_i.

For the concrete finetune, I also have to keep the augmented corpus balanced and computationally bounded. I take up to 25k examples per API, cap sequences at length 1,024, train with an effective batch size of 128 and learning rate 1e-5, and warm up linearly over the first 10% of training. The engineering choice is ZeRO-3 with BF16 on eight A100 40GB GPUs; I train up to 2k steps, check held-out CCNet perplexity every 500 steps, and keep the checkpoint with the best perplexity. These settings don't change the objective; they make the self-generated corpus trainable without letting one prolific tool dominate the mixture.

After finetuning, I decode normally until the model produces the → marker — its way of saying "I expect a tool result here." At that point I interrupt decoding, actually call the tool with the arguments it just wrote, insert the returned result and the </API> marker, and resume decoding. The model itself chose to call, chose the tool, chose the arguments; I just service the request.

One decoding tweak matters. With pure greedy decoding the model only calls a tool when <API> is literally the single most likely next token, which is conservative — it underuses tools it would benefit from. So I relax it: start a call whenever <API> is among the top `k_decode` most likely tokens; `k_decode=1` recovers greedy, and I use `k_decode=10` to make the model more willing to reach for a tool. This is separate from the top-position cap used while sampling candidates. I also cap decoding at one call per input so it can't fall into a loop of calling forever without producing output.

Let me also pin the thresholds, since "self-supervised" still leaves knobs. By default τ_s = 0.05 (only consider positions where the model gives the start marker at least 5% probability), τ_f = 1.0 (keep calls that cut the loss by at least 1.0), `k_pos = 5` top positions, m = 5 candidates per position. Some tools — the calculator, translation — are useful on only a tiny slice of text, so for those I open the gates: τ_s = 0, `k_pos = 20`, m = 10 to harvest enough candidates, and since that yields fewer survivors I loosen τ_f to 0.5. I also pre-filter the corpus per tool with cheap heuristics (only run the calculator sampler on texts with several numbers where one is the result of an operation on two others; only run the calendar sampler where I can recover the document's date from its URL; only run MT on paragraphs with a non-English chunk between English text), so I'm spending sampling budget where a tool is plausibly relevant.

The filter is the whole method, so here is exactly that computation. Given a passage, I run the model on three versions — the plain tokens, the tokens with the call but no response, and the tokens with the call and response — read off the predicted-probability of each actual next token, apply the decaying weights, and form the three losses; then I keep a passage iff (min of the two baselines) minus (loss-with-response) clears the threshold.

```python
import torch
from functools import partial

def log(t, eps=1e-20):
    return torch.log(t + eps)

def default_weight_fn(t):
    # normalized w_t from raw max(0, 1 - 0.2 t); denominator is 3.0
    return (1. - t * 0.2).clamp(min=0.) / 3.

def get_pred_prob(token_ids, logits):
    logits = logits[:, :-1]          # drop last logit
    token_ids = token_ids[:, 1:]     # the next-token targets
    probs = logits.softmax(dim=-1)
    return probs.gather(-1, token_ids[..., None]).squeeze(-1)   # p of the actual next token

def get_arange_start_at_token_id(token_ids, token_id, pad_id=-1):
    is_mask = token_ids == token_id
    arange = (is_mask.cumsum(dim=-1) > 0).cumsum(dim=-1)
    before = arange == 0
    arange = arange - 1
    return arange.masked_fill(before, pad_id)

# distance t counts up from the marker; tokens before it get weight 0
def weight_and_mask(token_ids, token_id, pad_id=-1, weighting_fn=default_weight_fn):
    t = get_arange_start_at_token_id(token_ids, token_id, pad_id)  # -1 before marker, 0,1,2,... after
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

    probs    = get_pred_prob(tokens, logits)                              # no call at all
    probs_wo = get_pred_prob(tokens_without_api_response, logits_wo)      # call, no response
    probs_w  = get_pred_prob(tokens_with_api_response,  logits_w)         # call + response

    wm = partial(weight_and_mask, weighting_fn=weighting_fn)
    weight_wo = wm(tokens_without_api_response[:, :-1], api_end_token_id) # credit after </API>
    weight_w  = wm(tokens_with_api_response[:, :-1],  api_end_token_id)
    weight    = wm(tokens_without_api_response[:, 1:], api_start_token_id)[:, :probs.shape[-1]]

    def loss_fn(weight, probs):                  # L_i(z) = sum w * -log p
        return (weight * -log(probs)).sum(dim=-1)

    loss    = loss_fn(weight,    probs)           # L_i(ε)
    loss_wo = loss_fn(weight_wo, probs_wo)        # L_i(e(c, ε))
    loss_w  = loss_fn(weight_w,  probs_w)         # L_i(e(c, r))

    loss_plus  = loss_w                           # L_i^+
    loss_minus = torch.minimum(loss_wo, loss)     # L_i^- = min(no-call, call-without-result)

    selected = (loss_minus - loss_plus) >= filter_threshold   # keep iff it helped by >= tau_f
    return selected
```

The double cumsum in `get_arange_start_at_token_id` is doing something specific: the first cumsum turns the marker indicator into a step function (0 before the marker, 1 from the marker on), the second counts positions past that step, and subtracting 1 shifts it so the result reads t = −1 before the marker and t = 0, 1, 2, … immediately after. Offset t = 0 lands on the first continuation token *after* `</API>`, not on the marker itself — exactly the token whose surprisal w_0 = 0.333 is supposed to weight most heavily.

One more thing the code makes me notice: `weight` (the no-call baseline) is anchored at the *start* marker via `api_start_token_id`, while `weight_wo`/`weight_w` are anchored at the *end* marker `api_end_token_id`. That's deliberate, not a typo — in the no-call sequence there is no </API> to key off of, so I count offsets from where the call *would* have begun, and the trailing `[:, :probs.shape[-1]]` slice keeps the three weight tensors the same length so they line up token-for-token against the same continuation. With all three anchored to the same physical continuation tokens, the three losses are measuring the same future under three different prefixes, which is the only way the comparison L_i^− − L_i^+ means anything.
