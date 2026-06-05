Let me be precise about what I'm trying to fix, because the failure is so specific it almost dictates the solution. The model is brilliant at the soft stuff and terrible at the hard, crisp stuff: it won't add two numbers reliably, it doesn't know today's date, it invents facts, it can't translate a low-resource phrase. Every one of these is a thing a tiny dedicated program does perfectly — a calculator, a clock, a lookup, a translation model. So I don't need to make the LM better at arithmetic. I need to let it *ask the calculator* and weave the answer back into its text. The model should be a generalist that knows when to phone a specialist.

The deep question isn't "can it call a tool" — of course it can, if I hand-script the call. The question is how it learns, on its own, *when* a call is worth making, *which* tool, and *what arguments*, without me annotating mountains of examples. Two existing roads both dead-end for me. Road one: collect tons of human demonstrations of tool calls. Expensive, and worse, it teaches the model what *humans* think is a useful call, which is not the same as what actually helps *this model* predict the next tokens. Road two: write a task-specific prompt that says "for this task, use the calculator like so." That doesn't generalize and it presupposes I already know which tool the task needs — exactly the decision I want the model to make. I want self-supervision: the model annotates its own data, and some automatic criterion keeps only the good annotations.

First, can a tool call even live inside text? It can, as long as the tool's input and output are both text. Then I represent a call as a tuple c = (a_c, i_c), the tool name and its input, and I linearize it with special marker tokens. Two forms — without and with the result r:

  e(c)    = <API> a_c ( i_c ) </API>
  e(c, r) = <API> a_c ( i_c ) → r </API>

so e(c) is "I'm calling tool a_c on input i_c," and e(c,r) additionally carries the returned result r after a → marker. (In practice I don't even need new vocabulary — I can spell <API>, </API>, → with existing tokens like "[", "]", "->". But conceptually they're three special tokens.) The point of inlining is that now a tool call is just *text*, so an ordinary language-modeling loss can supervise it — if only I had training sequences with calls inserted in the right places. I don't have those. The whole problem reduces to: manufacture a corpus where helpful API calls are inlined at the right positions, then finetune on it with the plain LM objective.

So, three stages: sample candidate calls into plain text, filter to the helpful ones, finetune on the survivors. Let me build each.

Sampling. Take a plain text x = x_1…x_n from an ordinary corpus. For each tool I write one short prompt P(x) — a handful of in-context examples showing the tool being inserted into text, ending with "Input: x / Output:". The model's in-context-learning does the rest: it'll propose where and how to call the tool in x. But I shouldn't ask for a call at every position; most positions don't want one. So I let the model tell me where it *wants* to start a call. For each position i, I read off the probability the model assigns to emitting the start marker right there:

  p_i = p_M(<API> | P(x), x_{1:i-1}).

These are the positions the model itself is inclined to open a call. I threshold them at τ_s, keep positions with p_i > τ_s, and if too many survive I keep the top k. Then at each kept position i, I sample actual call candidates c_i^1…c_i^m by continuing generation from [P(x), x_{1:i-1}, <API>] and stopping at </API>. (If the model never closes with </API>, I throw that candidate away.) Now I have, scattered through x, up to k positions each with up to m candidate calls.

Execute them — run each calculator/QA/search/MT/calendar call to get its text result r_i. Fine. Now the crux: which of these calls do I keep? This is the filter, and it's where the whole "self-supervised, model-defined usefulness" idea has to be made concrete. I need an automatic score that says "this call helped." Helped *with what*? With the only thing the model is ultimately judged on: predicting the upcoming tokens x_i, x_{i+1}, … . So the criterion should be: does giving the model the call *and its result* make the following text easier to predict than not having it?

Let me write that as a loss. The natural quantity is the cross-entropy the model pays on the continuation x_i…x_n when it's prefixed with some sequence z:

  L_i(z) = − Σ_{j=i}^{n} w_{j−i} · log p_M(x_j | z, x_{1:j-1}).

The w's are weights I'll pin down in a second; for now it's a weighted negative log-likelihood of the tokens from position i onward. I compare two prefixes. With the call and its result available:

  L_i^+ = L_i( e(c_i, r_i) ).

And the baseline — what the loss is if the call *doesn't* help. There are two ways the call could fail to help: I could do no call at all, or I could do the call but get no useful result. So I take the *better* (lower-loss) of those two as the bar to beat:

  L_i^− = min( L_i(ε), L_i( e(c_i, ε) ) ),

where ε is the empty sequence. The first term L_i(ε) is "predict the continuation with no API call." The second L_i(e(c_i, ε)) is "the model saw that a call was made but received no response." Why include that second one? Because I don't want to reward a call merely for the *act* of calling — sometimes just emitting "<API> Calculator(…) </API>" with no answer shifts the loss for spurious reasons. By taking the min, I force the call to earn its keep specifically through the *content of the result*: e(c_i, r_i) has to beat both "nothing" and "call-but-no-answer." A call is genuinely useful only if the result r_i itself made the future tokens easier.

Then keep the call iff it reduces loss by at least a margin τ_f:

  L_i^− − L_i^+ ≥ τ_f.

Let me sanity-check the sign, because it's easy to flip and the whole thing breaks if I do. L is a loss — lower is better. L_i^+ is the loss *with* the helpful prefix; if the call helps, L_i^+ is small. L_i^− is the loss without help; it's larger. So L_i^− − L_i^+ is positive and large exactly when the call slashed the loss. Requiring it ≥ τ_f > 0 keeps only calls that bought at least τ_f nats of improvement. Good, the direction is right: large positive difference = very helpful = keep.

Now those weights w. The information a call returns is useful for predicting tokens *near* the call, and its relevance should fade as I move further past it — a calculator result helps the next few tokens, not the token forty words later. So I want a weighting that's largest right at the call and decays to zero. Take a raw weight that ramps down linearly,

  w̃_t = max(0, 1 − 0.2·t),

so w̃_0 = 1, w̃_1 = 0.8, … and it hits 0 at t = 5: the call's "credit" spans roughly the next five tokens. Then normalize across the sequence so the weights sum to one:

  w_t = w̃_t / Σ_s w̃_s.

The max(0, ·) is the load-bearing part — it clamps the contribution of distant tokens to exactly zero rather than letting a linear function go negative, which would absurdly reward making far-future tokens *less* likely. So the filter scores a call almost entirely by its effect on the immediately following few tokens, which is exactly the locality I argued for.

There's a subtlety in L_i^+ I should be honest about: when I compute the loss with the call, I prefix e(c_i, r_i) to the continuation rather than splicing it into the middle of x at position i. Why? Because the model hasn't been finetuned yet — it has never seen an API call mid-sentence — so inserting one inside x would interrupt the natural flow and spike perplexity for reasons that have nothing to do with whether the result is useful. Prefixing keeps the measured continuation as natural text and isolates the question "did the result help."

Finetuning. After filtering, I take each surviving call and *now* splice it into the text at its position: for x with a kept call (c_i, r_i) at i, build

  x* = x_{1:i-1}, e(c_i, r_i), x_{i:n}.

Do this over the corpus and I get an augmented corpus C* that is — and this is the quietly crucial property — *the exact same text as the original corpus C, with API calls inserted*. Same words, same content, just calls woven in at positions where they demonstrably helped. So finetuning M on C* with the ordinary LM objective exposes it to all the same content it would have seen from C, which means it doesn't lose any general language ability; the only thing it additionally learns is the pattern "at positions like this, with arguments like these, calling this tool helps me predict what comes next." It learns when and how to use tools purely from its own feedback signal. (I also drop examples where every candidate call got filtered out, and I assume the surviving distribution is close enough to the original that language modeling stays intact — something I'd want to verify by checking perplexity with tools disabled.)

Inference. After finetuning, I decode normally until the model produces the → marker — its way of saying "I expect a tool result here." At that point I interrupt decoding, actually call the tool with the arguments it just wrote, insert the returned result and the </API> marker, and resume decoding. The model itself chose to call, chose the tool, chose the arguments; I just service the request.

One decoding tweak matters. With pure greedy decoding the model only calls a tool when <API> is literally the single most likely next token, which is conservative — it underuses tools it would benefit from. So I relax it: start a call whenever <API> is among the top k most likely tokens (k=1 recovers greedy; I push k up to make the model more willing to reach for a tool). And I cap it at one call per input so it can't fall into a loop of calling forever without producing output.

Let me also pin the thresholds, since "self-supervised" still leaves knobs. By default τ_s = 0.05 (only consider positions where the model gives the start marker at least 5% probability), τ_f = 1.0 (keep calls that cut the loss by at least 1.0), k = 5 top positions, m = 5 candidates per position. Some tools — the calculator, translation — are useful on only a tiny slice of text, so for those I open the gates: τ_s = 0, k = 20, m = 10 to harvest enough candidates, and since that yields fewer survivors I loosen τ_f to 0.5. I also pre-filter the corpus per tool with cheap heuristics (only run the calculator sampler on texts with several numbers where one is the result of an operation on two others; only run the calendar sampler where I can recover the document's date from its URL; only run MT on paragraphs with a non-English chunk between English text), so I'm spending sampling budget where a tool is plausibly relevant.

The filter is the whole method, so here is exactly that computation. Given a passage, I run the model on three versions — the plain tokens, the tokens with the call but no response, and the tokens with the call and response — read off the predicted-probability of each actual next token, apply the decaying weights, and form the three losses; then I keep a passage iff (min of the two baselines) minus (loss-with-response) clears the threshold.

```python
import torch
from functools import partial

def log(t, eps=1e-20):
    return torch.log(t + eps)

# w̃_t = max(0, 1 - 0.2 t): full credit at the call, linearly to 0 by t=5
def default_weight_fn(t):
    return (1. - t * 0.2).clamp(min=0.)

def get_pred_prob(token_ids, logits):
    logits = logits[:, :-1]          # drop last logit
    token_ids = token_ids[:, 1:]     # the next-token targets
    probs = logits.softmax(dim=-1)
    return probs.gather(-1, token_ids[..., None]).squeeze(-1)   # p of the actual next token

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

So the causal chain: the model fails exactly where crisp external tools succeed, so it should learn to call them; learning that from human labels is costly and misaligned with the model's own notion of usefulness, so make it self-supervised; tool calls inline into text as markered spans e(c) / e(c,r), turning "use a tool" into "predict text"; sample candidate calls where the model is inclined to open one (p_i > τ_s); keep a call only if having its *result* reduces the weighted next-token loss versus both no-call and call-without-result, by a margin τ_f — with weights that decay to zero a few tokens out so credit is local; splice the survivors back into the otherwise-unchanged corpus and finetune with the plain LM objective, so generality is preserved while the model learns when/which/how to call; and at inference, interrupt on the → marker to run the call and resume.
