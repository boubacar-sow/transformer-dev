# Statistics on snapshots

## Why do we need statistics?

The current Transformer model uses statistics on the training data in various stages. Means and standard deviations (std) are used to center and normalize the model inputs, e.g. normalize the delay of a train at its last observation with respect to its average delay (over the training set) at this station.

The statistics are represented in a dataclass `AllStats` at `transformer/inputs/snapshots/statistics.py`. Since computing them is costly, the computing and loading are done in separate functions, and computations from previous runs can be re-used if they match the current configuration. The two important functions are:

```python
from transformer.inputs.snapshots.statistics import compute_statistics, load_statistics

compute_statistics() # may take a long time

all_stats = load_statistics() # no extra cost
```

The `all_stats` object contains integer counts and `MeanStd` objects, which are a small dataclass containing the mean and standard deviation of a quantity. For instance, `all_stats.delays.mean` gives the average value of the delays across the training data.

## Storage system

Since computing the statistics takes time, we don't want to do it every time we train the model. For this reason, the statistics computation result is stored after it is done, in a folder whose name is a hash of the `cfg.data` parameters (see [Project configuration](1_project_configuration.md)). Calling `compute_statistics()` will only perform the computation if this folder is not found, or if `cfg.training.use_precomputed.statistics = False`.

## How statistics are computed

The general approach to compute each statistics is always the same : first, we loop over the data set of snapshots and add each value to a `StatsCounter`, which stores the distribution of the data that is fed to it (up to a precision `stats_counter.bin_size`, which allows not remembering every single input data). Then, the distribution represented by the `StatsCounter` can be summarized into its mean, standard deviation, min/max, etc.

### How to add new statistics

All the required steps happen in `transformer/inputs/snapshots/statistics.py`.
- Compute the statistics in `StatisticsFactory`:
  - In `__init__`: define a StatsCounter object to aggregate the data of interest
  - In `update`: update the StatsCounter with the data from each `SillonChunk` object
  - In `save_summary`:
    - Compute the summary statistics of the StatsCounter, e.g. with `counter.mean_std()` or `counter.median()`
    - Save it in the appropriate pickle file (`inter_pr` if the statistics is over pairs of PRs, global otherwise)
- Add it to the `AllStats` dataclass (or to `InterPrStats`, see above)
- **Update this documentation!**

### Filters and tags

**NB:** See [Filters and tags](4_filters_and_tags.md) for more details.

As in the actual training step, the input data is filtered according to the settings in `cfg.data.filters`. Several statistics (e.g., inter-PR statistics) are aggregated according to their tag. For instance, the inter-PR travel times may be grouped by train type, which prevents from averaging TGV and TER travel times.

## List of all computed statistics

In this table, `sc` denotes a `SillonChunk` object from the training set.

NB: `NDMeanStd` stands for N-Dimensional MeanStd, and contains numpy arrays in its mean and std.

### Global statistics

| Name                            | type                 | averages or counts the quantity                                |
| ------------------------------- | -------------------- | -------------------------------------------------------------- |
| `delays`                        | `MeanStd`            | `sc.prev_delays` and `sc.foll_delays`                          |
| `prev_theo_times`               | `MeanStd`            | `sc.prev_theo_times`                                           |
| `foll_theo_times`               | `MeanStd`            | `sc.foll_theo_times`                                           |
| `pk`                            | `MeanStd`            | Points Kilométriques of PRs                                    |
| `nbr_trains`                    | `MeanStd`            | Number of trains in the snapshot                               |
| `week_day`                      | `MeanStd`            | Day of the week                                                |
| `current_time`                  | `MeanStd`            | Current time in the day, in minutes                            |
| `train_type`                    | `NDMeanStd`          | One-hot occurrences of each train type                         |
| `train_num`                     | `int`                | Occurrences of each train number                               |
| `pr_nbr_passage`                | `int`                | Number of times each PR was seen in the data                   |
| `avg_inter_pr_diff_late`        | `(MeanStd, MeanStd)` | Average delay trains accumulate between two PRs                |
| `avg_inter_pr_middle_diff_late` | `(MeanStd, MeanStd)` | Same as above but middle mean (see `StatsCounter.middle_mean`) |

### Statistics over inter-PR travels

A dedicated `InterPrStats` dataclass is defined to store the inter-pr data. Its elements are dictionaries who take a pair of connected PRs, as well as possibly tags, depending on the code configuration (see [Filters](#filters-and-tags) section above). The pairs of PRs are those which are appear consecutively in at least one sillon.

| Name               | type      | averages or counts the quantity                                |
| ------------------ | --------- | -------------------------------------------------------------- |
| `counts`           | `int`     | occurrences of PR transitions                                  |
| `clearing_times`   | `MeanStd` | inter-PR travel times                                          |
| `diff_late`        | `MeanStd` | delay loss or gain over the inter-PR travel                    |
| `middle_diff_late` | `MeanStd` | Same as above but middle mean (see `StatsCounter.middle_mean`) |