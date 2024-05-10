# File storage

ℹ️ Please read the "Project configuration" page first.

The code needs to read various files (training data, pre-computed embeddings and statistics, information on the railway network, etc) and write to various files (model, embeddings and statistics, figures, etc). The code uses multiple storage locations:
- a local folder on the machine where the code is running (name given in `cfg.data.local_folder`)
- several AWS S3 buckets
  - one where the snapshots (training data) are created by the [augias](https://gitlab.com/osrdata/services/augias/) replay DAG
  - one for storing every other data required by the code
  - one for MLFlow artifacts (logs, figures, models on previous runs), which is never directly accessed (only used through the MLFlow API)

## Local file storage

All the interactions between the code and the files are done through the `transformer.utils.file_system.FileSystem` class, which provides functions to read, write, remove or list files. The main reasons for this abstraction layer are:
- the use of zstandard compression at several places in the code ; the `FileSystem` functions handle both compressed and uncompressed files in read and write modes, by only changing the `zstd_format` boolean argument,
- as it is written, this class is highly extendible to handle direct read / write / list / remove operations on an S3 bucket through the `S3Hook` tool (or directly through its dependence `boto3`).

In practice though, the `FileSystem` class is not used directly, as all the reqiured read and write functions are provided in the code in `transformer.utils.loaders`. For instance:

```python
from transformer.utils.loaders import load_csv, write_csv, load_pickle, write_pickle

df = load_csv("my_dataframe.csv", sep=";")
# do some stuff
save_csv("my_new_dataframe.csv", df)

large_object = load_pickle("large_object.pickle", zstd_format=True) # read zstandard compressed files
# do some stuff
save_pickle(new_large_object, "large_object_v2.pickle", zstd_format=True) # write with zstandard
```

Generally speaking, each function in `loaders` is a wrapper around a another function (from pandas, numpy, torch, matplotlib, etc) ; additional keyword arguments are passed to the auxiliary function, like the `sep=";"` argument in the example above. 

A non-exhaustive list of handled files include:
- text
- CSV
- JSON
- pickle files
- numpy storage (`np.save`)
- torch storage (`torch.save`)
- matplotlib figures

## Remote file storage

The remote files are stored in AWS S3 buckets. Reading and writing to these is done in the module `transformer.utils.s3`. It contains a few basic functions which allow downloading from and uploading to the buckets. In particular, the function `download_training_data_from_s3` downloads all the data necessary to the training procedure (train/test dataset, pre-computed embeddings or statistics, etc).