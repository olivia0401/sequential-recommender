# Sequential Recommendation

Compare popularity and first-order Markov baselines with GRU4Rec and SASRec on MovieLens-1M and RetailRocket. Ranking metrics measure next-item prediction; the SNIPS off-policy estimate is a separate, assumption-dependent estimate from logged data, not evidence of live business impact.

## Setup and data

```bash
pip install -r requirements.txt
```

Download MovieLens-1M or RetailRocket and place files as described in [`data/README.md`](data/README.md).

## Train and serve

```bash
python -m src.train --model sasrec --dataset movielens --epochs 15 --save model.pt
```

Set `MODEL_PATH` to the saved checkpoint and run:

```bash
# macOS/Linux
MODEL_PATH=model.pt python -m uvicorn api.app
```

Evaluation commands and metric details are in `README`'s evaluation scripts and `src/offpolicy.py`. Published benchmark scores may need rerunning after the documented evaluation corrections; check the results notes before comparing them.
