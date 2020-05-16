## Collaborative Filtering
This work does experiments with various collaborative filtering approaches. Specifically, comparing

- User based filtering

- Item based filtering

- One other model based filtering approach #TODO

Their performance is compared in two ways:

- Effectiveness of the recommendation on a supplied training set.

- Efficiency of the recommendation (i.e. runtime).

### Installation

### Running

Custom hyperparameters in a textfile i.e. _"./configs/config.txt"_.


A _results_ folder will contain a timestamp directory with the latest results.

### Datasets
Store unzipped under directory ```./data```
* MovieLens (smallest version available, with 100k ratings)(https://grouplens.org/datasets/movielens/100k/)

### Splitting
80:20 training:test set, after shuffling the data. 

### Techniques
* TODO
* TODO
* TODO

### Metrics
For evaluation of effectiveness, MSE.

### Report
Collaborative-Filtering-Alonso.pdf