# Sequential recommendation

Predict the next item a user will interact with from their past interactions.
I train four models on the same split and compare them on two datasets: MovieLens-1M
(movie ratings) and RetailRocket (real e-commerce browsing sessions).

Models: two baselines (popularity, first-order Markov) and two sequence models
(GRU4Rec, SASRec).

## Setup

```
pip install -r requirements.txt
```

Get a dataset (links in `data/README.md`):
- MovieLens-1M unzips to `data/ml-1m/ratings.dat`
- RetailRocket's `events.csv` goes at `data/events.csv`

## Running

Pick the dataset with `--dataset` (default `movielens`). Baselines first, they run
on CPU in seconds and check the data pipeline:

```
python -m src.train --model popularity --dataset retailrocket
python -m src.train --model markov --dataset retailrocket
```

Sequence models:

```
python -m src.train --model gru4rec --dataset retailrocket --epochs 15
python -m src.train --model sasrec --dataset retailrocket --epochs 15 --save model_retailrocket.pt
```

Serve top-N from a saved SASRec model (on Windows call uvicorn through `python -m`):

```
set MODEL_PATH=model_retailrocket.pt
python -m uvicorn api.app:app --reload
curl -X POST localhost:8000/recommend -H "content-type: application/json" -d "{\"history\": [12, 45, 7], \"k\": 10}"
```

## Evaluation

Leave-last-out. For each user the last interaction is the test target, the one
before it is validation, and everything earlier is the history the model sees. The
target is ranked against 100 sampled negatives. I report Recall@K, NDCG@K and MRR.

## Results

### MovieLens-1M (6040 users, 3706 items)

| Model | Recall@10 | NDCG@10 | MRR |
|-------|-----------|---------|-----|
| Popularity | 0.453 | 0.250 | 0.210 |
| Markov (first-order) | 0.916 | 0.621 | 0.535 |
| GRU4Rec | 0.746 | 0.531 | 0.475 |
| SASRec | 0.769 | 0.611 | 0.573 |

SASRec beats GRU4Rec on every metric and has the best MRR. The Markov chain is a
strong baseline: on MovieLens the previous item already predicts the next one well.

### RetailRocket e-commerce (22.9k visitors, 11.2k items, items with <50 events dropped)

| Model | Recall@10 | NDCG@10 | MRR |
|-------|-----------|---------|-----|
| Popularity | 0.342 | 0.186 | 0.163 |
| Markov (first-order) | 0.997 | 0.958 | 0.944 |
| GRU4Rec | 0.810 | 0.704 | 0.677 |
| SASRec | 0.994 | 0.992 | 0.992 |

The e-commerce numbers are much higher across the board. Browsing sessions are short
and repetitive (visitors move between a handful of items), so next-item prediction
against sampled negatives is far easier than on MovieLens: both the Markov baseline
and SASRec sit near the ceiling. The interesting gap here is GRU4Rec vs SASRec:
self-attention captures the session structure that the GRU misses.

## Layout

```
data/    datasets, not committed
src/     data.py, metrics.py, baselines.py, models.py, train.py
api/     FastAPI service
```

## Next steps

- Use addtocart / transaction events as a stronger signal than views.
- Add an off-policy / uplift evaluation to ask which recommendation actually
  changes behaviour, not just which item is likely next.
- Expose latency and request metrics on the API.
