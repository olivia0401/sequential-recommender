"""Cross-check our hand-written estimator against Open Bandit Pipeline (OBP).

OBP (github.com/st-tech/zr-obp) is the reference library for off-policy
evaluation of bandit / recommender policies. If our `snips` reproduces OBP's
InverseProbabilityWeighting and SelfNormalizedInverseProbabilityWeighting on the
same synthetic bandit feedback, the implementation in src/offpolicy.py is doing
exactly what the standard tooling does (just memory-cheaply, without
materialising a dense action distribution over an 11k-item catalogue).

Skipped automatically if OBP isn't installed.
"""
import numpy as np
import pytest

obp_ope = pytest.importorskip("obp.ope")
from obp.ope import (  # noqa: E402
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
)

from src.offpolicy import snips  # noqa: E402


def _synthetic_bandit(seed=0, n_rounds=300, n_actions=6):
    rng = np.random.default_rng(seed)

    def policy():
        z = rng.random((n_rounds, n_actions))
        return z / z.sum(axis=1, keepdims=True)   # each row a valid distribution

    target = policy()
    behavior = policy()
    action = rng.integers(0, n_actions, size=n_rounds)
    reward = (rng.random(n_rounds) < 0.3).astype(float)
    position = np.zeros(n_rounds, dtype=int)
    pscore = behavior[np.arange(n_rounds), action]      # logging prob of taken action
    action_dist = target[:, :, None]                    # (n_rounds, n_actions, 1)
    pi_a = target[np.arange(n_rounds), action]          # target prob of taken action
    return action, reward, position, pscore, action_dist, pi_a


def test_snips_matches_open_bandit_pipeline():
    action, reward, position, pscore, action_dist, pi_a = _synthetic_bandit()

    # clip disabled so the comparison is exact (OBP's basic estimators don't clip)
    mine = snips(pi_a, pscore, reward, clip=1e12)

    obp_ipw = InverseProbabilityWeighting().estimate_policy_value(
        reward=reward, action=action, pscore=pscore,
        action_dist=action_dist, position=position)
    obp_snipw = SelfNormalizedInverseProbabilityWeighting().estimate_policy_value(
        reward=reward, action=action, pscore=pscore,
        action_dist=action_dist, position=position)

    assert np.isclose(mine["value_ips"], obp_ipw, rtol=1e-6, atol=1e-9)
    assert np.isclose(mine["value"], obp_snipw, rtol=1e-6, atol=1e-9)
