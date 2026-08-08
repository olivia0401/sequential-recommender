# Data

Two datasets are supported. Pick one with `--dataset` when training.

## MovieLens-1M (`--dataset movielens`, default)

Download and unzip here so you end up with `data/ml-1m/ratings.dat`:

https://files.grouplens.org/datasets/movielens/ml-1m.zip

`ratings.dat` is `::`-separated: `userId::movieId::rating::timestamp`. Only the order
of each user's ratings is used (implicit feedback), not the rating value.

## RetailRocket (`--dataset retailrocket`, e-commerce)

https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

Download `events.csv` from that page and put it at `data/events.csv` (a Kaggle
login is required). Columns: `timestamp, visitorid, event, itemid, transactionid`,
where `event` is view / addtocart / transaction. Each visitor's events in time
order are treated as one interaction sequence.
