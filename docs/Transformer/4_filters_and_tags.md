# Filters and tags

## Motivation

Depending on the application and the place in the code, we may want to automatically access a subset of the trains in each snapshot, and have an automated way to do this. The main interests are:
- Filtering trains in the data set to remove non-relevant ones (trains de service, ...)
- Group the data by categories to make statistics (and, at training time, determine the group of each data point to use the relevant statistics)
- Evaluating the model on distinct subsets of the data set (e.g., measure the model's error only on TGV, TER, etc.)

Several small classes in `transformer.inputs.utils` are here to allow this.

## Filters

Filters are callable objects that return `True` or `False` depending on whether the input matches their condition. There are several classes on filters, the most used one being `TrainNumFilter` and its children classes (in `transformer.inputs.utils.train_num_filters`): a `TrainNumFilter` object takes a train number (`SillonChunk.train_num`) and returns a boolean. The filter class system is meant to be generic and composable, but feel free to modify it according to your needs (it might be *too* generic and composable).

Each filter file provides a dictionary of filters, e.g., the variable `TRAIN_NUM_FILTERS` in `transformer.inputs.utils.train_num_filters`. When running the code, we look at the settings in `cfg.data.filters` (see [Project configuration](1_project_configuration.md) for details on the configuration) to determine which filter to use. Additionally, `cfg.training.test_step.train_num_filters` also contains a list of filters which are used separately to compute test metrics.

## Tags

Tags are callable objects that return a tuple of strings that give some information about the input sillon, e.g., a train type given a train number. Similarly to filters, taggers inherit from a common abstract class and can be added according to the needs. The main use of tags is to group the data by categories when computing the statistics. For instance, instead of computing the average travel time between PRs, we could compute the average travel time between consecutive PRs *per train type*. The code could look like:

```python
tag = some_sillon_tagger(sillon) # e.g., tag = ("TGV",)
for i in range(len(prs)):
    key = (prs[i], prs[i+1]) + tag
    inter_pr_statistics[key] = ...
```