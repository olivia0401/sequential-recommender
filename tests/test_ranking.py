import numpy as np
import pytest

from src.data import sample_negatives
from src.metrics import aggregate, rank_of_true


def test_ties_rank_the_target_last():
    rank = rank_of_true(np.array([0.2, 0.2, 0.5, 0.1]), 0)
    assert rank == 3
    assert aggregate([rank], ks=(2,))["Recall@2"] == 0


def test_negatives_are_unique_and_exclude_seen_and_target():
    result = sample_negatives(2, 7, {0, 1, 3}, np.random.default_rng(42), k=100)
    assert sorted(result) == [4, 5, 6]


def test_exhausted_catalogue_returns_no_negatives():
    assert sample_negatives(2, 4, {1, 2, 3}, np.random.default_rng(42)) == []


def test_negative_sample_size_is_rejected():
    with pytest.raises(ValueError):
        sample_negatives(1, 4, set(), np.random.default_rng(42), k=-1)


@pytest.mark.parametrize('scores', [[float('nan'), 0], [0, float('inf')]])
def test_invalid_model_scores_cannot_produce_a_successful_rank(scores):
    with pytest.raises(ValueError):
        rank_of_true(scores, 0)
