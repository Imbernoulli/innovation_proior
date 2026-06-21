A masked-diffusion language model such as LLaDA generates by iterative denoising: it starts from a sequence of [MASK] tokens and, over many steps, predicts and commits a few tokens at a time while remasking the rest. The predictor is a bidirectional Transformer, so every position attends to every other position at every step, including positions that are still masked. This bidirectional attention is what gives the model its infilling ability, but it also makes inference expensive, because a standard decoder recomputes queries, keys, and values for all tokens at all layers at every denoising step. The problem is to cut this per-step QKV recomputation drastically while preserving final-task accuracy, using only signals the model already produces and with no retraining.

Existing ideas fall short because they refresh on a fixed schedule rather than on the model's actual state. Fast-dLLM decodes in fixed blocks and recomputes the whole cache at block boundaries, which wastes work when nothing has changed and holds stale KV through a block when the model is revising rapidly; it also recomputes all layers uniformly and freezes masked tokens near block edges that still influence the current prediction. dKV-Cache delays caching a decoded token by one step and refreshes on a fixed interval, so it still misses mid-interval drift and treats every layer the same. dLLM-Cache refreshes prompt features on a long fixed interval and recomputes generated rows whose raw features moved, but the interval is still a clock and the similarity test is on features rather than on the attention signal that drives KV change. What is missing is a policy that refreshes when and only when the state has moved, and refreshes only the layers and positions where it has moved.

The method I propose is called Elastic-Cache. It is a training-free, architecture-agnostic KV-cache policy for masked-diffusion language models. Instead of a rigid block or fixed interval, it uses three coupled mechanisms: a sliding prediction window that keeps only the leftmost live masks current while block-caching distant masked tokens; an attention-drift trigger that decides when a cached layer has gone stale; and a depth-aware reset that recomputes only from the first stale layer onward. Together these cut the per-step forward on three axes at once: fewer positions are forwarded, refresh is triggered only by actual movement, and when refresh happens it does not touch layers whose KV has already settled.

The sliding window works as follows. At each step only the leftmost beta masked positions are treated as live, with beta set to 16 by default. Distant masked tokens outside this window receive very little attention and act essentially as a length-bias prior on the sequence, so their KV can be frozen in the cache with little harm. Nearby masked positions that strongly attend to one another stay live and current, unlike a rigid block schedule that freezes masks at a block boundary even when they still matter. After step zero the query set is therefore only the tracked probe tokens, the tokens decoded in the previous step, and the current window of leftmost masks.

The attention-drift trigger decides when to refresh. In bidirectional attention the change in attention weights is upstream of the change in KV states, because a newly decoded token rewrites the attention output that earlier tokens computed back when it was still masked. Attention weights are also bounded probability vectors, so their movement is a clean scale-free signal. Per layer, Elastic-Cache tracks the most-attended already-decoded token. The most-attended token is the conservative choice because it has the smallest drift: if even its attention pattern has shifted past a threshold, then every less-stable token has shifted at least as much. The trigger compares the cosine similarity of this token's current attention-weight row to its row from the previous step; if the similarity falls below gamma, set by default to 0.9, the attention pattern has broken and a refresh is needed.

The depth-aware reset answers where to refresh. Transformer layers behave differently during denoising: early layers settle quickly onto local lexical structure, while deep layers keep shifting as global semantics revise. When the attention trigger fires at some layer, that layer is the last one allowed to finish in reuse mode; the next layer and all deeper layers recompute KV for all tokens from the cached full-sequence hidden state, exactly as at the initial step. In the implementation a trigger at zero-based block index b mutates the first-update boundary to b plus one, so deeper layers see the new boundary and switch to full recompute. This boundary adapts per step and per input on its own, rather than being hand set.

For token commitment inside the window, Elastic-Cache keeps confidence-aware parallel decoding. Each masked position in the window receives a softmax distribution over tokens; every position whose top probability clears epsilon, set to 0.9, is committed in parallel, and if none clears the threshold the cutoff is clamped to the current window maximum so at least one position is committed. This is orthogonal to caching and is what lets each step commit several tokens where the model is confident.

```python
import torch
import torch.nn.functional as F

MASK_ID, EOS_ID = 126336, 126081


@torch.no_grad()
def generate_with_elastic_cache(model, prompt, gen_length=512, window_length=16,
                                mask_id=MASK_ID, eos_id=EOS_ID, threshold=0.9,
                                tokens_per_iter=1, gamma=0.9, track_num=1,
                                block_caching=True):
    """Elastic-Cache decoding for a bidirectional masked-diffusion LM.
       gamma: attention-similarity refresh trigger; window_length: sliding window beta;
       track_num: number of most-attended tokens used as the drift probe."""
    for block in model.model.transformer.blocks:
        block.x_cache = block.q_cache = block.k_cache = block.v_cache = None
        block.track_token = None

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    query_position = torch.arange(x.shape[1], device=model.device)
    track_position = query_position[:0].clone()
    new_decoded_position = query_position[:prompt.shape[1]].clone()
    masked_position = query_position[prompt.shape[1]:].clone()
    L = len(model.model.transformer.blocks)
    i, nfe, num_computed, total_computed = 0, 0, 0, 0
    decoded_eos = False

    while True:
        query_masked_position = masked_position[:window_length] if block_caching else masked_position

        if i == 0:
            x_query, start_reset = x, -1
        else:
            query_position = torch.cat([track_position, new_decoded_position, query_masked_position], dim=0)
            x_query, start_reset = x[:, query_position], L

        positions = [query_position, track_position, query_masked_position, masked_position]
        lengths = [x.shape[1], start_reset, gamma, track_num]

        logits = model(x_query, use_cache=True, lengths=lengths, positions=positions).logits
        logits = (logits[:, query_masked_position, :] if logits.shape[1] == x.shape[1]
                  else logits[:, -query_masked_position.shape[0]:, :])
        if not block_caching:
            query_masked_position = query_masked_position[:window_length]
            logits = logits[:, :window_length]

        track_position = torch.cat([b.track_token for b in model.model.transformer.blocks],
                                   dim=0).unique(sorted=False)

        if threshold is not None:
            x, new_decoded_position, eos_pos = _commit_confident(
                logits, query_masked_position, x, threshold, eos_id)
        else:
            x, new_decoded_position, eos_pos = _commit_topk(
                logits, query_masked_position, x, tokens_per_iter, eos_id)
        masked_position = masked_position[~torch.isin(masked_position, new_decoded_position)]

        nfe += 1
        if not decoded_eos and eos_pos.shape[0] > 0:
            eos_pos = eos_pos.min().item()
            decoded_eos = True
            masked_position = masked_position[masked_position <= eos_pos]

        num_computed += L - lengths[1]
        total_computed += L
        if masked_position.numel() == 0:
            break
        i += 1
    return x, nfe, num_computed / total_computed


def _commit_confident(logits, query_masked, x, threshold, eos_id=EOS_ID):
    p = F.softmax(logits.to(torch.float64), dim=-1)
    conf, pred = torch.max(p, dim=-1)
    keep = (conf >= min(threshold, conf.max()))
    commit = query_masked[keep[0]]
    pred = pred[:, keep[0]]
    x[:, commit] = pred
    return x, commit, commit[pred.eq(eos_id)[0]]


def _commit_topk(logits, query_masked, x, num_transfer_tokens=1, eos_id=EOS_ID):
    p = F.softmax(logits.to(torch.float64), dim=-1)
    conf, pred = torch.max(p, dim=-1)
    _, keep = torch.topk(conf[0], k=min(num_transfer_tokens, pred.shape[1]), largest=True)
    commit = query_masked[keep]
    pred = pred[:, keep]
    x[:, commit] = pred
    return x, commit, commit[pred.eq(eos_id)[0]]


@torch.no_grad()
def elastic_attention(block, x, positions, lengths, block_idx, qkv_fn, attn_fn, scale):
    """Per-layer KV caching with the attention-drift trigger.
       lengths[1] is start_reset, the first zero-based layer to full-recompute."""
    query_position, track_position, query_masked, masked_position = positions
    key_len, start_reset, gamma, track_num = lengths

    if block_idx > start_reset:
        block.x_cache = x
    else:
        block.x_cache[:, query_position, :] = x
        if block_idx == start_reset:
            x = block.x_cache

    q, k, v = qkv_fn(x)

    if block_idx >= start_reset:
        block.q_cache, block.k_cache, block.v_cache = q, k, v
    else:
        past_k = block.k_cache.clone()
        block.k_cache[:, :, query_position, :] = k
        block.v_cache[:, :, query_position, :] = v
        past_q = block.q_cache[:, :, track_position, :].clone()
        block.q_cache[:, :, query_position, :] = q
        k, v = block.k_cache, block.v_cache

    att, att_weight = attn_fn(q, k, v, need_weights=True)

    if block_idx >= start_reset:
        masked_att = att_weight[:, :, query_masked, :]
    else:
        masked_att = att_weight[:, :, -query_masked.shape[0]:, :]
        cur_track_att = att_weight[:, :, :track_position.shape[0], :]
        past_att = torch.softmax(past_q @ past_k.transpose(-2, -1) * scale, dim=-1)
        sim = F.cosine_similarity(past_att, cur_track_att, dim=1).mean()
        if sim < gamma:
            lengths[1] = block_idx + 1

    masked_att = masked_att.sum(dim=(0, 1, 2))
    masked_att[masked_position] = 0.0
    block.track_token = masked_att.topk(k=track_num, dim=0, largest=True)[1]
    return x, att
```
