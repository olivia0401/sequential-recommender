"""Off-policy / uplift evaluation for the recommenders.

The ranking metrics (Recall/NDCG/MRR) answer "is the model good at guessing the
next item?". They don't answer the question a business actually has: "if we had
shown this recommender's picks, would people have converted more?" You can't read
that off the logs directly, because the logs only tell you what happened under the
system that was already running, not what would have happened under a new one.

That's a causal question, and this module estimates it off-policy — from logged
data alone, without deploying anything — with an inverse-propensity-scoring (IPS)
estimator, self-normalised (SNIPS) to keep the variance down.

Setup (a contextual bandit over the RetailRocket logs):
  context x  = the visitor's history so far
  action  a  = the item they actually interacted with next (the test target)
  reward  r  = 1 if that interaction was an addtocart / transaction, else 0
               (i.e. did it convert, not just get viewed)

  logging policy  pi0(a)   -- the system that generated the logs. We don't have
                              its true propensities (RetailRocket is organic
                              traffic, not a logged bandit), so we MODEL it as
                              "popularity": pi0(a) proportional to how often a was
                              seen in training. Popular items get shown/seen more.
                              Add-one smoothing gives pi0 full support, which IPS
                              needs.
  target policy   pi(a|x)  -- the recommender we're evaluating, turned into a
                              distribution with a softmax over its scores.

  SNIPS value:  V(pi) = sum_i w_i r_i / sum_i w_i,   w_i = pi(a_i|x_i) / pi0(a_i)
  uplift:       V(pi) - V(pi0),   where V(pi0) = mean(r_i) is the logged
                                  conversion rate (the logging policy's own value)

Honest caveats (the numbers are only as good as these):
  * pi0 is estimated, not logged. Real off-policy evaluation uses the deployed
    system's true propensities; treat the popularity model as a proxy.
  * IPS is unbiased only if pi0 has full support and is roughly right, and the
    variance blows up when pi and pi0 disagree. We clip the weights and report
    the effective sample size (ESS) and the clipped fraction so that variance is
    visible rather than hidden.
  * Popularity-as-target is a sanity row: it equals pi0, so its uplift must come
    out ~0. If it doesn't, the estimator is wired wrong.
"""
import argparse
import json

import numpy as np

from .data import PAD, load_retailrocket


def snips(pi_a, pi0_a, r, clip=20.0):
    """Self-normalised IPS value estimate plus variance diagnostics.

    pi_a, pi0_a, r are per-logged-sample arrays: the target and logging policy
    probabilities of the logged action, and its reward.
    """
    pi_a = np.asarray(pi_a, dtype=np.float64)
    pi0_a = np.asarray(pi0_a, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    w = pi_a / pi0_a
    clipped_frac = float(np.mean(w > clip))
    w = np.minimum(w, clip)
    w_sum = w.sum()

    value = float((w * r).sum() / (w_sum + 1e-12))          # SNIPS
    value_ips = float(np.mean(w * r))                        # plain IPS
    ess = float(w_sum ** 2 / (np.square(w).sum() + 1e-12))   # effective samples
    return {
        "value": value,
        "value_ips": value_ips,
        "ess": ess,
        "ess_frac": ess / len(r),
        "clipped_frac": clipped_frac,
        "n": int(len(r)),
    }


class PopularityPolicy:
    """pi(a) proportional to training frequency, with add-one smoothing so every
    item has non-zero probability. Used as the logging policy pi0."""

    def __init__(self, data):
        p = data.item_pop.astype(np.float64).copy()
        p[PAD] = 0.0
        p = p + 1.0            # add-one smoothing -> full support (an IPS requirement)
        p[PAD] = 0.0
        self.p = p / p.sum()

    def action_probs(self, histories, actions):
        return self.p[np.asarray(actions)]


class MarkovPolicy:
    """First-order transition distribution, smoothed and mixed with popularity so
    it stays a proper distribution even for unseen last-items."""

    def __init__(self, data, temp=1.0, alpha=0.1):
        self.n_items = data.n_items
        self.temp = temp
        self.alpha = alpha
        pop = data.item_pop.astype(np.float64).copy()
        pop[PAD] = 0.0
        self.pop = pop / pop.sum()
        self.trans = {}
        for seq in data.train.values():
            for a, b in zip(seq[:-1], seq[1:]):
                row = self.trans.setdefault(a, {})
                row[b] = row.get(b, 0.0) + 1.0

    def _dist(self, last):
        row = self.trans.get(last)
        if not row:
            return self.pop
        d = np.zeros(self.n_items, dtype=np.float64)
        for it, c in row.items():
            d[it] = c ** (1.0 / self.temp)
        d = d / d.sum()
        # back off toward popularity so every action keeps some support
        return (1.0 - self.alpha) * d + self.alpha * self.pop

    def action_probs(self, histories, actions):
        out = np.empty(len(actions), dtype=np.float64)
        for i, (h, a) in enumerate(zip(histories, actions)):
            out[i] = self._dist(h[-1] if h else PAD)[a]
        return out


class NeuralPolicy:
    """A trained GRU4Rec / SASRec turned into a distribution by softmax over the
    catalogue at the last step, with temperature."""

    def __init__(self, model, max_len=50, temp=1.0, device=None, batch_size=256):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.max_len = max_len
        self.temp = temp
        self.batch_size = batch_size

    def action_probs(self, histories, actions):
        torch = self.torch
        out = np.empty(len(actions), dtype=np.float64)
        for i in range(0, len(histories), self.batch_size):
            chunk = histories[i:i + self.batch_size]
            seqs = []
            for h in chunk:
                s = list(h[-self.max_len:])
                seqs.append([PAD] * (self.max_len - len(s)) + s)
            x = torch.tensor(seqs, device=self.device)
            with torch.no_grad():
                logits = self.model(x)[:, -1, :] / self.temp
                logits[:, PAD] = float("-inf")
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            for j, a in enumerate(actions[i:i + self.batch_size]):
                out[i + j] = probs[j, a]
        return out


def collect(data):
    """Logged (history, action, reward) triples from the test split.

    The test context includes the validation item, matching evaluate() in models.py.
    """
    if not data.test_reward:
        raise ValueError(
            "This dataset has no conversion signal (test_reward). Off-policy "
            "evaluation needs RetailRocket event types; MovieLens has none."
        )
    histories, actions, rewards = [], [], []
    for user, action in data.test.items():
        history = list(data.train[user]) + [data.val[user]]
        if not history:
            continue
        histories.append(history)
        actions.append(action)
        rewards.append(data.test_reward[user])
    return histories, actions, np.asarray(rewards, dtype=np.float64)


def evaluate_policies(data, policies, clip=20.0):
    """Run every named policy through SNIPS and return a results dict.

    `policies` maps name -> policy object exposing action_probs(histories, actions).
    The logging policy pi0 is always popularity.
    """
    histories, actions, rewards = collect(data)
    pi0 = PopularityPolicy(data)
    pi0_a = pi0.action_probs(histories, actions)
    v_logging = float(rewards.mean())

    results = {"logging_conversion_rate": v_logging, "policies": {}}
    for name, policy in policies.items():
        pi_a = policy.action_probs(histories, actions)
        est = snips(pi_a, pi0_a, rewards, clip=clip)
        est["uplift"] = est["value"] - v_logging
        results["policies"][name] = est
    return results


def _print_table(results):
    v0 = results["logging_conversion_rate"]
    print(f"\nlogging (popularity) conversion rate  V(pi0) = {v0:.4f}\n")
    header = f"{'policy':<14}{'V(pi)':>9}{'uplift':>10}{'ESS%':>8}{'clip%':>8}"
    print(header)
    print("-" * len(header))
    for name, e in results["policies"].items():
        print(f"{name:<14}{e['value']:>9.4f}{e['uplift']:>+10.4f}"
              f"{100 * e['ess_frac']:>7.1f}%{100 * e['clipped_frac']:>7.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/events.csv")
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--temp", type=float, default=1.0,
                    help="softmax temperature for the target policies")
    ap.add_argument("--clip", type=float, default=20.0,
                    help="IPS weight clip (variance control)")
    ap.add_argument("--gru4rec", default=None, help="path to a saved GRU4Rec checkpoint")
    ap.add_argument("--sasrec", default=None, help="path to a saved SASRec checkpoint")
    ap.add_argument("--json", default=None, help="write full results to this JSON file")
    args = ap.parse_args()

    print(f"loading retailrocket from {args.data} ...")
    data = load_retailrocket(args.data)
    print(f"users={len(data.test)}  items={data.n_items - 1}  "
          f"converters={int(sum(data.test_reward.values()))}")

    policies = {
        "popularity": PopularityPolicy(data),   # sanity: equals pi0, uplift ~ 0
        "markov": MarkovPolicy(data, temp=args.temp),
    }
    if args.gru4rec or args.sasrec:
        import torch

        from .models import GRU4Rec, SASRec
        if args.gru4rec:
            ckpt = torch.load(args.gru4rec, map_location="cpu")
            m = GRU4Rec(ckpt["n_items"])
            m.load_state_dict(ckpt["state_dict"])
            policies["gru4rec"] = NeuralPolicy(m, args.max_len, args.temp)
        if args.sasrec:
            ckpt = torch.load(args.sasrec, map_location="cpu")
            m = SASRec(ckpt["n_items"], max_len=args.max_len)
            m.load_state_dict(ckpt["state_dict"])
            policies["sasrec"] = NeuralPolicy(m, args.max_len, args.temp)

    results = evaluate_policies(data, policies, clip=args.clip)
    _print_table(results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved -> {args.json}")


if __name__ == "__main__":
    main()
