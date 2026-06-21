The central puzzle is distinguishing interpolation from true rule learning. A small neural network can memorize any finite training set, including corrupted labels, so perfect training accuracy does not tell us whether the model has discovered the underlying rule. Natural datasets make this hard to isolate because the exact rule is unknown and train-test splits are entangled with distribution shifts. What is needed is a fully specified finite world where every possible query has a known answer, where train and validation are just disjoint cells of the same symbolic table, and where the input representation gives no notation shortcuts.

Existing ideas fall short in exactly this isolation. Random-label experiments show that networks can store arbitrary labels, but they do not create a setting where a discoverable rule predicts held-out cells. Double-descent and epoch-wise interpolation work show that longer training can improve generalization, but they typically vary capacity, noise, or natural-data complexity rather than fixing a tiny exact rule and watching what happens after the training set is already fit. Algorithmic benchmarks test symbolic procedures, but their emphasis is usually on architecture, length generalization, or unlimited generated data, not on the transition inside a fixed overparameterized model after memorization. The missing ingredient is a microscope that separates memorization time from rule-discovery time.

The method is Grokking. It builds a finite symbolic rule-learning testbed from complete binary operation tables and trains a small decoder-only transformer far past the point where training accuracy saturates. For a task such as modular division or permutation product, every element and operator is encoded as an opaque token. A random fraction of the table cells are revealed for training, and the remaining cells from the same table are held out. The model sees sequences of the form `<a> <op> <b> <=> <c>` and is trained to predict the right-hand side after the equals sign. Because the symbols are opaque, the model cannot exploit decimal digits or permutation notation; the only path to a held-out cell is through the pattern of interactions across the equations it has seen.

The empirical signature of grokking is a delayed generalization transition. Training accuracy becomes near-perfect within the first thousand or so optimization steps, while validation accuracy remains at chance. If training stopped there, the model would be labeled a memorizer. But with continued optimization, often for hundreds of thousands or millions of steps, validation accuracy suddenly begins to rise and eventually reaches the same near-perfect level as training accuracy. Validation loss first rises during the overfitting phase and then undergoes a second descent far after training loss has flattened. The key finding is that generalization can emerge long after overfitting appears complete.

Several probes support the interpretation that the optimizer moves from a sharp memorizing solution to a flatter rule-like solution. Weight decay is the strongest intervention, more than halving the amount of training data needed compared with most alternatives. Minibatch noise and injected Gaussian weight or update noise also help. Capacity is not the bottleneck: corrupting many training labels still leaves the network able to reach perfect training accuracy, showing it has enough parameters to memorize both the rule and explicit lies. Output-layer embeddings provide a structural signature, arranging into circular structures for modular addition or clustering into cosets for symmetric-group operations. On fixed-time runs, validation accuracy correlates negatively with a sharpness proxy, consistent with the idea that generalization appears once the weights reach flatter regions.

The training recipe for the main experiments uses a two-layer decoder-only transformer with width 128, four attention heads, causal masking, and about four hundred thousand non-embedding parameters. The optimizer is AdamW with learning rate 1e-3, betas (0.9, 0.98), epsilon 1e-8, linear warmup over ten updates, minibatch size min(512, ceil(train_size / 2)), weight decay 1, and up to 1e5 gradient updates for the reported learning-time curves. A dramatic modular-division demonstration uses 50 percent training data, Adam without weight decay, and 1e6 updates to make the late transition visible.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def make_modular_data(p, op, train_frac, seed=0):
    """Build train/val splits of a modular binary operation table.
    op: callable (a, b) -> c with 0 <= a, b, c < p.
    """
    rng = random.Random(seed)
    all_rows = []
    for a in range(p):
        for b in range(p):
            c = op(a, b) % p
            all_rows.append((a, b, c))
    rng.shuffle(all_rows)
    n_train = int(len(all_rows) * train_frac)
    return all_rows[:n_train], all_rows[n_train:]

class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4,
                 d_ff=512, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_emb', pe.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, norm_first=False,
            activation='relu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        T = x.size(1)
        h = self.token_emb(x) + self.pos_emb[:, :T, :]
        h = self.encoder(h, mask=self.mask[:T, :T], is_causal=True)
        return self.head(h)

def encode(rows, token_map):
    """Encode rows into input/target token ids.
    Sequence: EOS a op b = c EOS.
    Input is all tokens except final EOS; target is input shifted left by one.
    """
    eos = token_map['<eos>']
    eq = token_map['=']
    op_tok = token_map['op']
    inputs, targets = [], []
    for a, b, c in rows:
        seq = [eos, a, op_tok, b, eq, c, eos]
        inputs.append(seq[:-1])
        targets.append(seq[1:])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def train_grokking(p=97, train_frac=0.5, steps=100000, device='cpu'):
    def op(a, b):
        # modular division a / b mod p (b != 0)
        if b == 0:
            b = 1
        return (a * pow(b, -1, p)) % p

    train_rows, val_rows = make_modular_data(p, op, train_frac)
    special = ['<eos>', 'op', '=']
    tokens = special + list(range(p))
    token_map = {t: i for i, t in enumerate(tokens)}
    # map raw values to token ids for a, b, c
    def remap(rows):
        return [(token_map[a], token_map[b], token_map[c]) for a, b, c in rows]
    train_rows = remap(train_rows)
    val_rows = remap(val_rows)

    model = CausalTransformer(vocab_size=len(tokens)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  betas=(0.9, 0.98), eps=1e-8,
                                  weight_decay=1.0)
    x_train, y_train = encode(train_rows, token_map)
    x_val, y_val = encode(val_rows, token_map)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    # mask for answer positions: tokens after '='
    ans_mask_train = (y_train == token_map['=']).cumsum(dim=1) >= 1
    ans_mask_val = (y_val == token_map['=']).cumsum(dim=1) >= 1

    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = F.cross_entropy(logits.view(-1, len(tokens)),
                               y_train.view(-1),
                               reduction='none')
        loss = (loss * ans_mask_train.view(-1).float()).sum() / ans_mask_train.sum()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                train_logits = model(x_train)
                train_preds = train_logits.argmax(dim=-1)
                train_acc = ((train_preds == y_train) * ans_mask_train).sum().item() / ans_mask_train.sum().item()
                val_logits = model(x_val)
                val_preds = val_logits.argmax(dim=-1)
                val_acc = ((val_preds == y_val) * ans_mask_val).sum().item() / ans_mask_val.sum().item()
            print(f"step {step:6d}  train_acc {train_acc:.4f}  val_acc {val_acc:.4f}")
    return model

if __name__ == "__main__":
    train_grokking(p=97, train_frac=0.5, steps=100000, device='cpu')
```
