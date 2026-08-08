import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from src.data import PAD
from src.models import SASRec

# train a model first:  python -m src.train --model sasrec --epochs 20 --save model.pt
MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")
MAX_LEN = int(os.getenv("MAX_LEN", "50"))

app = FastAPI(title="Sequential Recommender")
_model = None


class Request(BaseModel):
    history: list[int]      # item ids, oldest first
    k: int = 10


def get_model():
    global _model
    if _model is None:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        _model = SASRec(ckpt["n_items"], max_len=MAX_LEN)
        _model.load_state_dict(ckpt["state_dict"])
        _model.eval()
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
@torch.no_grad()
def recommend(req: Request):
    model = get_model()
    seq = req.history[-MAX_LEN:]
    seq = [PAD] * (MAX_LEN - len(seq)) + seq
    x = torch.tensor(seq).unsqueeze(0)
    logits = model(x)[0, -1]
    logits[PAD] = float("-inf")
    for it in req.history:            # don't recommend something already seen
        logits[it] = float("-inf")
    top = torch.topk(logits, req.k).indices.tolist()
    return {"items": top}
