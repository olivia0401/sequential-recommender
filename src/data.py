import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

PAD = 0  # item id 0 is reserved for padding


@dataclass
class Dataset:
    train: dict          # user -> sequence for training (all but last two)
    val: dict            # user -> validation target (second to last)
    test: dict           # user -> test target (last)
    n_items: int         # distinct items + 1 (for PAD)
    item_pop: np.ndarray # training frequency per item id
    test_reward: dict = None  # user -> 1 if the test interaction converted
                              # (addtocart / transaction), else 0. RetailRocket
                              # only; None for MovieLens (no event types).


def load_movielens_1m(path="data/ml-1m/ratings.dat", min_len=5):
    """Read ratings.dat, order each user's history by time, split leave-last-out.

    Users with fewer than min_len interactions are dropped.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for the download link."
        )

    df = pd.read_csv(path, sep="::", engine="python",
                     names=["user", "item", "rating", "ts"])
    df = df.sort_values(["user", "ts"], kind="stable")

    # relabel item ids to 1..N, keeping 0 free for padding
    unique_items = df["item"].unique()
    item2idx = {it: i + 1 for i, it in enumerate(unique_items)}
    df["item"] = df["item"].map(item2idx)
    n_items = len(unique_items) + 1

    train, val, test = {}, {}, {}
    item_pop = np.zeros(n_items, dtype=np.int64)

    for user, grp in df.groupby("user", sort=False):
        seq = grp["item"].tolist()
        if len(seq) < min_len:
            continue
        train[user] = seq[:-2]
        val[user] = seq[-2]
        test[user] = seq[-1]
        for it in seq[:-2]:
            item_pop[it] += 1

    return Dataset(train=train, val=val, test=test, n_items=n_items, item_pop=item_pop)


def load_retailrocket(path="data/events.csv", min_len=5, min_item_count=50):
    """Per-visitor event sequences from the RetailRocket e-commerce dataset.

    events.csv columns: timestamp, visitorid, event (view/addtocart/transaction),
    itemid, transactionid. Each visitor's events in time order are one sequence,
    split leave-last-out. Items seen fewer than min_item_count times are dropped
    (RetailRocket has a long tail of items viewed once or twice).

    Also records test_reward[visitor] = 1 when the final (test) interaction was an
    addtocart or transaction rather than a plain view. That conversion signal is
    what the off-policy evaluation in src/offpolicy.py optimises for.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for the download link."
        )

    df = pd.read_csv(path, usecols=["timestamp", "visitorid", "event", "itemid"])
    counts = df["itemid"].value_counts()
    df = df[df["itemid"].isin(counts[counts >= min_item_count].index)]
    df = df.sort_values(["visitorid", "timestamp"], kind="stable")

    unique_items = df["itemid"].unique()
    item2idx = {it: i + 1 for i, it in enumerate(unique_items)}
    df["itemid"] = df["itemid"].map(item2idx)
    n_items = len(unique_items) + 1

    converting = {"addtocart", "transaction"}
    train, val, test, test_reward = {}, {}, {}, {}
    item_pop = np.zeros(n_items, dtype=np.int64)

    for visitor, grp in df.groupby("visitorid", sort=False):
        items = grp["itemid"].tolist()
        events = grp["event"].tolist()
        # collapse immediate repeats (viewing the same item twice in a row),
        # keeping the event of the last row for each kept position
        seq, seq_events = [], []
        for j, it in enumerate(items):
            if j == 0 or it != items[j - 1]:
                seq.append(it)
                seq_events.append(events[j])
            else:
                seq_events[-1] = events[j]
        if len(seq) < min_len:
            continue
        train[visitor] = seq[:-2]
        val[visitor] = seq[-2]
        test[visitor] = seq[-1]
        test_reward[visitor] = int(seq_events[-1] in converting)
        for it in seq[:-2]:
            item_pop[it] += 1

    return Dataset(train=train, val=val, test=test, n_items=n_items,
                   item_pop=item_pop, test_reward=test_reward)


def sample_negatives(true_item, n_items, seen, rng, k=100):
    """k random item ids that aren't the true item and weren't seen by the user."""
    negs = []
    while len(negs) < k:
        cand = int(rng.integers(1, n_items))  # 1.. because 0 is PAD
        if cand != true_item and cand not in seen:
            negs.append(cand)
    return negs
