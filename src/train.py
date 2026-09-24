import argparse
import json
import random

import numpy as np

from .data import load_movielens_1m, load_retailrocket, sample_negatives
from .metrics import aggregate, rank_of_true


def eval_baseline(model, data, split="test", n_neg=100, seed=42):
    rng = np.random.default_rng(seed)
    targets = data.val if split == "val" else data.test
    ranks = []
    for user, true_item in targets.items():
        history = list(data.train[user])
        if split == "test":
            history = history + [data.val[user]]
        seen = set(history)
        negs = sample_negatives(true_item, data.n_items, seen, rng, k=n_neg)
        candidates = [true_item] + negs
        scores = model.score(history)[candidates]
        ranks.append(rank_of_true(scores, true_index=0))
    return aggregate(ranks)


def set_seed(seed):
    """Seed Python, NumPy and (if installed) torch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["popularity", "markov", "gru4rec", "sasrec"])
    ap.add_argument("--dataset", default="movielens",
                    choices=["movielens", "retailrocket"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--save", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    print(f"loading {args.dataset} ...")
    if args.dataset == "movielens":
        data = load_movielens_1m(args.data or "data/ml-1m/ratings.dat")
    else:
        data = load_retailrocket(args.data or "data/events.csv")
    print(f"users={len(data.train)}  items={data.n_items - 1}")

    if args.model in ("popularity", "markov"):
        from .baselines import MarkovModel, PopularityModel
        model = PopularityModel(data) if args.model == "popularity" else MarkovModel(data)
        metrics = eval_baseline(model, data, split="test")
    else:
        import torch

        from .models import GRU4Rec, SASRec, evaluate, train_model
        if args.model == "gru4rec":
            model = GRU4Rec(data.n_items)
        else:
            model = SASRec(data.n_items, max_len=args.max_len)
        model = train_model(model, data, max_len=args.max_len, epochs=args.epochs)
        metrics = evaluate(model, data, split="test", max_len=args.max_len)
        if args.save:
            torch.save({"state_dict": model.state_dict(),
                        "n_items": data.n_items,
                        "model": args.model,
                        "max_len": args.max_len,
                        "dataset": args.dataset,
                        "item_ids": data.item_ids}, args.save)
            print(f"saved -> {args.save}")

    print(f"\n=== {args.model} (test) ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
