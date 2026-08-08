import numpy as np

from .data import Dataset


class PopularityModel:
    # score every item by its training frequency, ignore the user's history
    def __init__(self, data: Dataset):
        self.scores = data.item_pop.astype(np.float64)

    def score(self, history):
        return self.scores


class MarkovModel:
    # first-order chain: score(next) = count(last_item -> next).
    # sparse transitions (a dense matrix is huge once the catalogue is big).
    # if the last item was never seen in training, fall back to popularity.
    def __init__(self, data: Dataset):
        self.trans = {}
        for seq in data.train.values():
            for a, b in zip(seq[:-1], seq[1:]):
                row = self.trans.setdefault(a, {})
                row[b] = row.get(b, 0.0) + 1.0
        self.pop = data.item_pop.astype(np.float64)
        self.n_items = data.n_items

    def score(self, history):
        if not history:
            return self.pop
        row = self.trans.get(history[-1])
        if not row:
            return self.pop
        out = np.zeros(self.n_items, dtype=np.float64)
        for it, c in row.items():
            out[it] = c
        return out
