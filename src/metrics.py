import numpy as np


def rank_of_true(scores, true_index):
    # 1-based position of the true item once candidates are sorted by score.
    # ties break against us (true item goes after equal-scoring ones).
    true_score = scores[true_index]
    return int((scores > true_score).sum()) + 1


def recall_at_k(rank, k):
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank, k):
    return 1.0 / np.log2(rank + 1) if rank <= k else 0.0


def mrr(rank):
    return 1.0 / rank


def aggregate(ranks, ks=(5, 10, 20)):
    ranks = np.asarray(ranks, dtype=np.float64)
    out = {}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(ranks <= k))
        gains = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
        out[f"NDCG@{k}"] = float(np.mean(gains))
    out["MRR"] = float(np.mean(1.0 / ranks))
    return out
