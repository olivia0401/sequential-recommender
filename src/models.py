import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset

from .data import PAD, Dataset


class SequenceDataset(TorchDataset):
    # (input, target) where target is input shifted by one step.
    # left-pad each sequence to max_len so the last position is the newest item.
    def __init__(self, sequences, max_len):
        self.rows = [s for s in sequences if len(s) >= 2]
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        seq = self.rows[i][-(self.max_len + 1):]
        inp, tgt = seq[:-1], seq[1:]
        pad = self.max_len - len(inp)
        inp = [PAD] * pad + inp
        tgt = [PAD] * pad + tgt
        return torch.tensor(inp), torch.tensor(tgt)


class GRU4Rec(nn.Module):
    def __init__(self, n_items, emb_dim=64, hidden=128, layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_items, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, n_items)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.gru(self.drop(self.emb(x)))
        return self.out(h)


class SASRec(nn.Module):
    # self-attention recommender (Kang & McAuley 2018), stripped down
    def __init__(self, n_items, max_len, emb_dim=64, blocks=2, heads=2, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=blocks)
        self.out = nn.Linear(emb_dim, n_items)
        self.max_len = max_len

    def forward(self, x):
        b, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(b, L)
        h = self.drop(self.item_emb(x) + self.pos_emb(pos))
        # causal mask so position t only attends to <= t
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        pad_mask = x == PAD
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad_mask)
        return self.out(h)


def train_model(model, data, max_len=50, epochs=20, batch_size=128, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ds = SequenceDataset(list(data.train.values()), max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        val = evaluate(model, data, split="val", max_len=max_len, device=device)
        print(f"epoch {ep:2d}  loss {total/len(loader):.4f}  "
              f"val Recall@10 {val['Recall@10']:.4f}  val NDCG@10 {val['NDCG@10']:.4f}")
    return model


@torch.no_grad()
def evaluate(model, data, split="test", max_len=50, n_neg=100, device=None,
             seed=42, batch_size=256):
    from .metrics import aggregate, rank_of_true
    from .data import sample_negatives

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rng = np.random.default_rng(seed)
    targets = data.val if split == "val" else data.test

    # build padded histories + candidate pools up front, then score in batches
    seqs, cands = [], []
    for user, true_item in targets.items():
        history = list(data.train[user])
        if split == "test":
            history = history + [data.val[user]]
        if not history:
            continue
        seen = set(history)
        negs = sample_negatives(true_item, data.n_items, seen, rng, k=n_neg)
        cands.append([true_item] + negs)          # index 0 is always the true item
        seq = history[-max_len:]
        seqs.append([PAD] * (max_len - len(seq)) + seq)

    ranks = []
    for i in range(0, len(seqs), batch_size):
        xb = torch.tensor(seqs[i:i + batch_size], device=device)
        cb = torch.tensor(cands[i:i + batch_size], device=device)
        logits = model(xb)[:, -1, :]              # (b, n_items) at the last step
        scores = torch.gather(logits, 1, cb).cpu().numpy()   # (b, 1 + n_neg)
        for row in scores:
            ranks.append(rank_of_true(row, true_index=0))

    return aggregate(ranks)
