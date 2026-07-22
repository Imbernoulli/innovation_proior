Let me start from what actually hurts when I try to deploy this fine-tuned vision-language-action model on a real robot. I have a 7B-parameter policy that takes an image and a language instruction and emits a robot action, and it already follows language and generalizes across tasks — that part is good. But it predicts the action the way a language model predicts text: the action is a 7-dimensional delta end-effector pose, each dimension normalized to [-1,+1] and chopped into 256 bins, each bin assigned a token, and the model spits out those tokens one at a time, left to right, under a causal mask. So to produce a single timestep's action I have to run the decoder seven times in sequence, each pass conditioned on the tokens already emitted. On an A100 that single-timestep generation takes about a third of a second. Three to five actions per second. My controller wants twenty-five to fifty, and if it's a bimanual arm with double the action dimensions it wants even more headroom. I am an order of magnitude too slow before I even start.

And the slowness isn't just an inconvenience I can paper over with a faster GPU, because it blocks the one thing I know would make the policy *better*. It's well documented that predicting a chunk of several future actions per query and executing them — action chunking — raises manipulation success rates: it shrinks the effective horizon of the task, which directly attacks the compounding-error problem of behavioral cloning, where each small prediction error nudges the robot a little further off the demonstrated distribution until it lands in a state it has never seen and can't recover from. Chunking also lets the policy express temporally-correlated structure that a single-step Markovian policy fundamentally can't — a pause in the middle of a motion, say. So I want chunking for quality. But under autoregressive decoding a chunk of K timesteps costs K times D sequential passes. With D=7 and even a modest K=8 that's fifty-six sequential decoder calls per query. The latency was already unacceptable at one timestep; chunking multiplies it eightfold. The decoding scheme and the thing I most want are in direct conflict.

So the real question underneath the speed complaint is: why am I decoding sequentially at all? In language, left-to-right is forced — token t+1 genuinely depends on what token t turned out to be, you're sampling from a chain. But is that true of my action tokens? The seven dimensions of one timestep's delta pose are just seven coordinates of one vector; there is no deep sense in which the gripper dimension must be "generated after" the z-position dimension. I'm imposing an autoregressive ordering on a quantity that doesn't have one, purely because I inherited the language-model machinery. If I could predict all the action positions at once, in a single forward pass, the K·D factor collapses to one.

What would that take? Two things. First, the decoder needs something to put in the action-token positions as *input*, since in the autoregressive setup the input to position t is the (teacher-forced or previously-generated) token at t-1 — and if I'm not generating sequentially, there's no previous token to feed. Second, the causal mask has to go: a causal mask exists precisely to enforce left-to-right dependence, so each position can only see earlier ones. If I want every action position predicted simultaneously and informed by all the others and by the full observation, I need every action position to attend to every other one. So: replace the K·D action-input slots with placeholder embeddings that carry no action content — they just mark "an action goes here, at this position in the chunk" — and swap the causal mask for a bidirectional one over those slots. Now one forward pass: the observation tokens and the K·D empty action slots all go in, bidirectional attention lets each action slot read the observation and the other slots, and the decoder produces a hidden state at each of the K·D action positions in a single shot.

This is exactly the trick a from-scratch chunking policy I know already uses — its transformer decoder reads K fixed learned query embeddings, one per output timestep, and emits all K actions non-causally in one pass; the queries differ only in which output position they stand for. The placeholder action embeddings here play the same role: they differ from each other only in their positional encoding, the marker of *where in the chunk* this slot sits. So I'll feed empty action embeddings distinguished by position, and use bidirectional attention. There's a worry I should name: parallel prediction is in principle less expressive than autoregressive, because I've thrown away the ability to condition action position j on the realized value of position i. Maybe the chunk needs that internal coherence. But I have a high-capacity 7B trunk with full bidirectional attention across the whole chunk and the observation, so each slot still sees everything *except the sampled values* of the others. That is an expressivity question to validate, not a reason to keep an already-too-slow left-to-right decode.

And here's the part that turns a speed fix into a quality path: once decoding is a single pass, chunking is no longer multiplied by K in decoder calls. Adding more action slots — go from D slots to K·D slots — does lengthen the attention sequence, so it is not literally zero-cost, but it stays one forward pass instead of fifty-six sequential calls for D=7, K=8. So the very thing autoregressive decoding made impractical becomes affordable, and with it the prior action-chunking benefits: reduced effective horizon, less compounding error, and access to temporally correlated motion structure. The single-pass decode and chunking aren't two separate ideas; the first is what makes the second deployable. That's the first pillar.

Now, what comes out of that forward pass at each action position is a hidden state, a vector the width of the LLM. I still have to turn those hidden states into actions, and here I have a choice I should think through rather than inherit. The native recipe would linearly project each action-position hidden state to 256 logits and pick a bin — the discrete representation. But binning is a lossy bottleneck I keep paying for: 256 bins is a coarse grid over [-1,+1], and fine manipulation lives in the gaps between grid points. I could add more bins for finer resolution, but then each individual bin-token appears more rarely in the few hundred demonstrations I have, and rare tokens generalize worse — I'd be trading precision against data efficiency along a curve where neither end is good. The cleaner move is to not discretize at all: let the model output the continuous action directly. Replace the token-logit output layer with a small head that maps the action-position hidden states straight to real-valued, normalized actions. No grid, no bottleneck; the precision ceiling that discretization imposed just disappears. That's the second pillar — a continuous action representation via a regression head.

If I'm regressing continuous actions, I need a loss on them, and this is where I should be careful, because the loss choice interacts with the messy reality of human demonstration data. The obvious default is squared error, L2: minimize the mean of (predicted − ground_truth)². But demonstrations are noisy, and at a given observation a human might have done slightly different things on different episodes, so the target is effectively a spread of plausible actions rather than a point. L2 penalizes large residuals quadratically, so it's pulled hard toward outliers and its optimum is the *mean* of the conditional action distribution — and the mean of a spread of distinct valid motions can be a blended action that's valid under none of them. L1, the mean absolute error, penalizes linearly, so it's far less swayed by the occasional large deviation and its optimum is the *median*. For action prediction the median is the better summary: it's robust to noisy demonstration outliers, and the from-scratch chunking policy I keep coming back to reports exactly this — they switched their reconstruction term from the common L2 to L1 and found L1 models the action sequence more precisely. So my objective is the mean L1 distance between the predicted action chunk and the ground-truth chunk, on the normalized [-1,+1] actions: L = mean |predicted − ground_truth| over all K timesteps and all D dimensions. That's the third pillar.

Let me sit with the alternative for a second, because there's a more powerful tool on the table and I want to know why I'm *not* reaching for it. I could model the action chunk with a conditional denoising diffusion process — learn to predict the noise added to ground-truth actions under a forward schedule, then at inference start from Gaussian noise and iteratively denoise down to an action chunk, conditioned on the observation. Diffusion's selling point is genuine: it represents *multimodal* action distributions, so when there really are several distinct valid ways to act, it can capture all of them instead of collapsing to a median. That's a real expressivity advantage L1 doesn't have. So why not diffusion? Two reasons, and they're decisive for *this* setting. First, cost: diffusion needs many sequential denoising steps at inference — tens of them — so even riding on top of my nice single-pass parallel decode, each chunk now costs tens of passes, and I'm right back in the latency hole I climbed out of; training also converges slower. Second, and this is the bet: my trunk is a 7B model with enormous capacity. The reason simple regression usually loses to diffusion is under-capacity — a small network forced to output one vector hedges by averaging modes. A high-capacity model conditioned on a rich observation can often pin down which mode the current situation calls for and regress it cleanly, so the multimodality that diffusion buys may be largely redundant here. If that's right, L1 should roughly *match* diffusion's success rate while being dramatically faster to train and to run. I'll proceed with L1 as the primary objective on that reasoning, fully aware of its honest limitation — if the demonstrations are truly multimodal, where several genuinely different action sequences are all correct for the same input, the median-seeking L1 head can't represent that and diffusion would have the edge. For focused, consistent-strategy demonstrations, the median is exactly what I want.

So the recipe is taking shape from the pain backward: single-pass parallel decode (so chunking is affordable and compounding error drops), continuous actions (so discretization stops capping precision), L1 regression (robust, precise, single-pass-cheap, and good enough given the capacity). Now I need to actually build the regression head and wire it into the forward pass, and there are concrete decisions there too.

How does the head consume the hidden states? After the forward pass I have a hidden state at each of the K·D action positions — shape (batch, K·D, hidden). I want one D-dimensional action per timestep, so K actions total. The natural grouping is by timestep: the D=7 action positions belonging to timestep k together determine that timestep's 7-dim action, so I gather those D hidden states for each timestep and feed them jointly. Concretely, reshape (batch, K·D, hidden) to (batch, K, D·hidden) — each of the K rows is the concatenation of that timestep's D action-position hidden states — and map each row through the head to D outputs. So the head's input width is D·hidden and its output width is D=action_dim, applied across the K timesteps.

What should the head itself be? It sits on top of the (LoRA-adapted) frozen-ish trunk, reading high-dimensional features and regressing a low-dimensional target on a few hundred demonstrations — so I want something expressive enough to extract the action but stable and not prone to overfitting. A plain stack of linear+ReLU layers can be finicky to train at this width. Residual connections make deeper MLPs trainable by giving gradients a clean path, and layer normalization stabilizes the activation scale, so an MLP-ResNet — a few residual feedforward blocks — is the right shape. For the block ordering I'll put the normalization *before* the transformation rather than after: pre-normalization (normalize, then linear, then ReLU, then add the input back) is known to keep the residual stream's gradients well-behaved and training stable, the same reasoning that favors pre-norm feedforward blocks in transformers. So each block is: LayerNorm → Linear → ReLU, added to its input via the residual. Around the stack I'll bracket with an input projection (LayerNorm → Linear → ReLU) that lifts the D·hidden input to the working width, then two such residual blocks, then a final LayerNorm and a linear projection down to action_dim. That's a head about four layers deep with ReLU activations — enough to extract the action, no more. It's small relative to the 7B trunk, so it adds negligible inference cost — which is the whole point of choosing a head over a heavier generative decoder.

Let me make sure the wiring at training time is right, because there's a subtlety in extracting the action positions. I run the trunk forward with the action-token slots filled by empty (zeroed) action-token features — they carry no action content, just hold the positions — under bidirectional attention. I take the last layer's hidden states, drop the vision-patch prefix to get the text-portion hidden states, and now I need exactly the K·D positions that correspond to action tokens. At training time I have the ground-truth label sequence, so I can build a boolean mask over the response positions that marks the current timestep's action tokens and the next (chunked) timesteps' action tokens — the union of "current action" and "next actions" masks — and select those. That gives me K·D hidden states, which I reshape to (batch, K·D, hidden) and hand to the head. At evaluation time I don't have labels, but I know the layout: the action positions start right after the prompt and run for K·D contiguous slots, so I slice them by position instead of by mask. Either way the head sees (batch, K·D, hidden), regroups to (batch, K, D·hidden), and predicts (batch, K, action_dim).

And the loss is then simply the mean absolute error between that (batch, K, action_dim) prediction and the ground-truth normalized action chunk of the same shape — one `L1Loss` call. Because the actions are normalized to [-1,+1], the loss is in a clean comparable scale; a convergence criterion like "train until mean normalized L1 falls below 0.01" is meaningful precisely because of that normalization.

One more set of choices that aren't the method itself but make it work on a few hundred demonstrations: I adapt the 7B trunk with LoRA — low-rank trainable matrices in the frozen model's linear layers — because full fine-tuning of 7B on so little data is both wasteful and prone to overfitting, and LoRA at a modest rank gives enough capacity to adapt while keeping the trainable parameter count and memory down. The regression head and (if used) a small projector for robot proprioceptive state are trained alongside the LoRA adapters. The action chunk size K is fixed by protocol — eight timesteps for the single-arm benchmark, to line up with the diffusion-policy comparison — predicted in one pass and executed open-loop before requerying. Learning rate, warmup, and the decay schedule are training-loop knobs around all this, not part of what makes the method a method.

Let me trace the whole causal chain once before I write it down. The native autoregressive-discrete decode is too slow (3-5 Hz, K·D sequential passes for a chunk) and its 256-bin quantization caps precision, and the slowness blocks action chunking, which I know would reduce compounding error and raise success. Asking *why decode sequentially* — the action coordinates have no intrinsic order — leads to replacing the per-position autoregressive inputs with positionally-marked empty action embeddings and the causal mask with bidirectional attention, so all K·D action positions are predicted in a single forward pass; that single pass makes action chunking essentially free, delivering both throughput and the chunking quality gain. Removing the discretization bottleneck means regressing continuous actions directly from the action-position hidden states with a small head instead of projecting to bin-logits. Choosing the regression loss, L1 over L2 because the median is robust to noisy multimodal-ish demonstrations and more precise; choosing L1 over diffusion because diffusion's iterative sampling reintroduces the latency I just eliminated and the trunk's capacity makes simple regression competitive, accepting that truly multimodal demonstrations are L1's weak spot. The head consumes the D per-timestep action hidden states jointly (reshape K·D → K rows of D·hidden), through a pre-norm MLP-ResNet small enough to add negligible cost, and is trained by the mean absolute error to the normalized ground-truth chunk. Now the code, filling the empty slots in the harness — the decode-driving (empty action features, bidirectional, parallel) and the action decoder (MLP-ResNet head + L1 loss):

```python
import torch
import torch.nn as nn

ACTION_DIM = 7          # delta end-effector pose dims (this robot)
NUM_ACTIONS_CHUNK = 8   # K: timesteps predicted per query, executed open-loop


class MLPResNetBlock(nn.Module):
    """Pre-norm residual feedforward block: LayerNorm -> Linear -> ReLU, + input.
    Pre-normalization keeps the residual stream's gradients well-behaved, so the
    head trains stably on top of the (LoRA-adapted) trunk."""
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    """Input projection -> num_blocks residual blocks -> output projection.
    Small relative to the 7B trunk, so it adds negligible inference cost."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.relu(self.fc1(x))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        return self.fc2(x)


class L1RegressionActionHead(nn.Module):
    """Continuous-action head: maps the action-position hidden states of a chunk
    to K continuous D-dim actions, trained by L1 regression."""
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=ACTION_DIM):
        super().__init__()
        self.action_dim = action_dim
        # input width is D * hidden: the D action-position hidden states of a
        # timestep are consumed jointly to predict that timestep's D-dim action.
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: (B, K * ACTION_DIM, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        # regroup the K*D action-position hidden states into K rows of D*hidden
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged)  # (B, K, action_dim)


def zero_action_token_embeddings(input_embeddings, all_actions_mask):
    """Inner trunk-forward step for the L1 path.
    Action token IDs still mark positions, but their input embeddings carry no
    action content; positional embeddings are added later by the language model."""
    return input_embeddings * ~all_actions_mask.unsqueeze(-1)


def training_step(vla, action_head, batch, num_patches):
    # Ground-truth chunk, normalized to [-1, +1]: (B, K, action_dim)
    ground_truth_actions = batch["actions"].to(torch.bfloat16)

    # Parallel decode: inside vla.forward, labels identify current/next action
    # token positions, and the L1 path zeroes those input embeddings before the
    # multimodal language-model forward. The design uses
    # bidirectional action-token attention, so all K*D positions are produced in
    # one pass.
    output = vla(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"].to(torch.bfloat16),
        labels=batch["labels"],
        output_hidden_states=True,
    )

    # Pull the K*D action-position hidden states out of the last layer.
    last_hidden_states = output.hidden_states[-1]          # (B, seq_len, hidden)
    text_hidden_states = last_hidden_states[:, num_patches:-1]  # drop vision prefix
    ground_truth_token_ids = batch["labels"][:, 1:]
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

    predicted_actions = action_head.predict_action(actions_hidden_states)  # (B, K, D)
    loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)      # mean |.|
    return loss
```
