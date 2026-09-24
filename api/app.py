import os

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import PAD
from src.models import SASRec

# Train and save a model first, e.g.:
#   python -m src.train --model sasrec --dataset retailrocket --save model_retailrocket.pt
MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")
MAX_K = 100

app = FastAPI(title="Sequential Recommender")
_state = None


class Request(BaseModel):
    # Item ids, oldest first. Raw dataset ids when the checkpoint carries the id
    # mapping (checkpoints saved by the current src.train); internal indices
    # (1..n_items-1) for older checkpoints.
    history: list[int] = Field(default_factory=list, max_length=10_000)
    k: int = Field(10, ge=1, le=MAX_K)
    exclude_seen: bool = True  # drop items already in the history from the results


def load_state(path=None):
    """Load the checkpoint once. Returns a dict with the model and id mapping."""
    ckpt = torch.load(path or MODEL_PATH, map_location="cpu")
    # Older checkpoints did not store max_len; recover it from the position table.
    max_len = ckpt.get("max_len") or ckpt["state_dict"]["pos_emb.weight"].shape[0]
    model = SASRec(ckpt["n_items"], max_len=max_len)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    item_ids = ckpt.get("item_ids")
    return {
        "model": model,
        "n_items": ckpt["n_items"],
        "max_len": max_len,
        "item_ids": item_ids,
        "raw_to_idx": {raw: i + 1 for i, raw in enumerate(item_ids)} if item_ids else None,
    }


def get_state():
    global _state
    if _state is None:
        _state = load_state()
    return _state


def to_internal(history, state):
    """Map request ids to internal indices, rejecting anything the model can't score."""
    if state["raw_to_idx"] is not None:
        unknown = [i for i in history if i not in state["raw_to_idx"]]
        if unknown:
            raise HTTPException(422, f"unknown item ids: {unknown[:10]}")
        return [state["raw_to_idx"][i] for i in history]
    bad = [i for i in history if not 1 <= i < state["n_items"]]
    if bad:
        raise HTTPException(422, f"item ids must be in 1..{state['n_items'] - 1}; got {bad[:10]}")
    return list(history)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
@torch.no_grad()
def recommend(req: Request):
    state = get_state()
    model, max_len = state["model"], state["max_len"]
    hist = to_internal(req.history, state)

    seq = hist[-max_len:]
    seq = [PAD] * (max_len - len(seq)) + seq
    logits = model(torch.tensor(seq).unsqueeze(0))[0, -1]
    logits[PAD] = float("-inf")
    if req.exclude_seen:
        for it in hist:
            logits[it] = float("-inf")
    k = min(req.k, int(torch.isfinite(logits).sum()))
    top = torch.topk(logits, k).indices.tolist()
    if state["item_ids"] is not None:
        top = [state["item_ids"][i - 1] for i in top]
    return {"items": top}
