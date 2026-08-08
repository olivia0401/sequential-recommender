"""Correctness checks for the off-policy estimator and the policy plumbing.

These use tiny hand-built inputs so the expected numbers can be worked out by
hand — no dataset download or model training needed.
"""
import numpy as np

from src.data import Dataset
from src.offpolicy import (
    MarkovPolicy,
    PopularityPolicy,
    evaluate_policies,
    snips,
)


def test_snips_equals_mean_reward_when_policies_match():
    # pi == pi0 -> every weight is 1 -> SNIPS is just the average reward.
    rng = np.random.default_rng(0)
    p = rng.random(50)
    r = (rng.random(50) < 0.3).astype(float)
    out = snips(p, p, r)
    assert np.isclose(out["value"], r.mean())
    assert np.isclose(out["ess"], len(r))          # all weights 1 -> ESS == n


def test_snips_reweights_toward_high_reward_actions():
    # Target policy favours the converting actions relative to logging -> the
    # estimated value should exceed the raw logged conversion rate.
    r = np.array([1.0, 1.0, 0.0, 0.0])
    pi0 = np.array([0.25, 0.25, 0.25, 0.25])
    pi = np.array([0.45, 0.45, 0.05, 0.05])        # up-weights the r=1 rows
    out = snips(pi, pi0, r)
    assert out["value"] > r.mean()


def test_clipping_flags_heavy_weights():
    r = np.ones(3)
    pi0 = np.array([0.5, 0.5, 0.001])              # third row -> weight ~500
    pi = np.array([0.5, 0.5, 0.5])
    out = snips(pi, pi0, r, clip=10.0)
    assert out["clipped_frac"] > 0.0


def _toy_dataset():
    # three "users", catalogue of items 1..4 (0 is PAD). item_pop drives the
    # popularity policy; test_reward marks who converted.
    train = {1: [1, 2], 2: [2, 3], 3: [3, 1]}
    val = {1: 3, 2: 1, 3: 2}
    test = {1: 4, 2: 4, 3: 1}
    item_pop = np.array([0, 3, 2, 2, 0], dtype=np.int64)   # index 0 = PAD
    test_reward = {1: 1, 2: 0, 3: 1}
    return Dataset(train=train, val=val, test=test, n_items=5,
                   item_pop=item_pop, test_reward=test_reward)


def test_popularity_target_has_zero_uplift():
    # The popularity target policy IS the logging policy, so however the reward
    # falls, its uplift must be ~0. This is the estimator's self-check.
    data = _toy_dataset()
    res = evaluate_policies(data, {"popularity": PopularityPolicy(data)})
    assert abs(res["policies"]["popularity"]["uplift"]) < 1e-9


def test_policies_return_probabilities_in_range():
    data = _toy_dataset()
    res = evaluate_policies(
        data, {"popularity": PopularityPolicy(data), "markov": MarkovPolicy(data)}
    )
    v0 = res["policies"]["popularity"]["value"]
    assert 0.0 <= v0 <= 1.0
    assert res["policies"]["markov"]["n"] == 3
