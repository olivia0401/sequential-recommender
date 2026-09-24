"""API tests against a tiny, untrained SASRec checkpoint written to tmp_path."""
import pytest
import torch

pytest.importorskip("httpx")  # required by fastapi.testclient
from fastapi.testclient import TestClient  # noqa: E402

import api.app as app_module  # noqa: E402
from src.models import SASRec  # noqa: E402

N_ITEMS = 20  # internal indices 1..19, 0 is PAD
RAW_IDS = [1000 + i for i in range(1, N_ITEMS)]


def _client(tmp_path, with_mapping=True):
    torch.manual_seed(0)
    ckpt = {"state_dict": SASRec(N_ITEMS, max_len=8).state_dict(), "n_items": N_ITEMS}
    if with_mapping:
        ckpt.update({"max_len": 8, "item_ids": RAW_IDS})
    path = tmp_path / "model.pt"
    torch.save(ckpt, path)
    app_module._state = app_module.load_state(str(path))
    return TestClient(app_module.app)


def test_recommend_returns_raw_ids_and_excludes_history(tmp_path):
    c = _client(tmp_path)
    r = c.post("/recommend", json={"history": [1001, 1002, 1003], "k": 5})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 5
    assert set(items) <= set(RAW_IDS)
    assert not set(items) & {1001, 1002, 1003}


def test_unknown_raw_id_is_422(tmp_path):
    c = _client(tmp_path)
    assert c.post("/recommend", json={"history": [5], "k": 5}).status_code == 422


@pytest.mark.parametrize("k", [0, -1, 101])
def test_k_out_of_range_is_422(tmp_path, k):
    c = _client(tmp_path)
    assert c.post("/recommend", json={"history": [1001], "k": k}).status_code == 422


def test_k_larger_than_catalogue_is_capped(tmp_path):
    c = _client(tmp_path)
    r = c.post("/recommend", json={"history": [1001], "k": 100})
    assert r.status_code == 200
    assert len(r.json()["items"]) == N_ITEMS - 2  # minus PAD and the seen item


def test_empty_history_ok(tmp_path):
    c = _client(tmp_path)
    assert c.post("/recommend", json={"history": [], "k": 3}).status_code == 200


def test_old_checkpoint_uses_internal_ids_with_bounds(tmp_path):
    c = _client(tmp_path, with_mapping=False)
    assert app_module._state["max_len"] == 8  # recovered from pos_emb
    assert c.post("/recommend", json={"history": [1, 2], "k": 3}).status_code == 200
    for bad in ([0], [-1], [N_ITEMS]):
        assert c.post("/recommend", json={"history": bad, "k": 3}).status_code == 422
