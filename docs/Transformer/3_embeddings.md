# Embeddings

ℹ️ For an explanation of the Transformer embeddings and their use in the model, see this page: [Résilience de l'Exploitation / Transformer / Embeddings des inputs](https://pdi-dgexsolutions-sncf.atlassian.net/wiki/spaces/RE/pages/508297247/Embeddings+des+inputs).


## Computation and storage

The module `transformer.inputs.embeddings.embeddings` contains the two main functions used by the rest of the code to manipulate the embeddings:

```python
from transformer.inputs.embeddings.embeddings import compute_embeddings, load_embeddings

compute_embeddings() # may take a long time

embeddings = load_embeddings() # no extra cost
```

### Computation

The function `compute_embeddings` looks at which embeddings are required as inputs of the model (this is written in the dictionary `cfg.model.use_inputs`). It only computes the required embeddings, if 1) they have not already been computed or 2) `cfg.training.use_precomputed.embeddings` is `False`.

Once an embedding is computed, it is stored in a folder whose name is a hash of its configuration. For instance, the Laplacian embedding will be stored in the folder `path_to_embeddings_directory/laplacian/hash_code`, with `hash_code` obtained by using the `transformer.utils.config.hash_config` function on `cfg.embeddings.laplacian_embedding`. This ensures that only the embeddings with new settings are computed.

### Loading and storage

The `embeddings` variable obtained from `load_embeddings` is a `dict[str, Embedding]`: for instance, `embeddings["laplacian_pr"]` contains the Laplacian embedding of PRs. The `Embedding` class is a wrapper class around `torch.nn.Embedding`. It takes as input the name of the directory where the embedding is stored, e.g. `laplacian/97aua878aojlhv986`.

The embeddings are stored under a common format:
- `embedding.pickle`: the state dictionary of the `torch.nn.Embedding` object
- `key_to_emb.json`: a dictionary mapping the key of each embedding (typically, the ID Gaia of Points Remarquables)
- `config.json`: the settings of the embedding (its dimension, algorithm hyperparameters, etc.)

ℹ️ In practice, these paths are accessed through the `paths` variable, through the keys `embedding_weights_f`, `embedding_config_f`, `embedding_key_to_emb_f` (see [Project configuration](1_project_configuration.md)).

In practice, for instance, in order to access the embedding of a list of PRs, we would do the following.

```python
from transformer.inputs.embeddings.embeddings import load_embeddings

embeddings = load_embeddings()

laplacian_embedding = embeddings["laplacian_pr"]

pr_list = [...] # list of ID Gaia of PRs

emb_list = laplacian_embedding[pr_list] # matrix with shape (len(pr_list), embedding dimension)
```

ℹ️ By default, the embeddings are centered and normalized. This behavior can be controlled with the `normalize_embedding` argument of the `Embedding` class.

Additionally to the method shown above, the embeddings can be accessed in two steps, as shown below:
```python
pr_list = [...]

indices = laplacian_embedding.get_indices(pr_list) # torch.Tensor filled with integer indices

emb_list = laplacian_embedding.get_values(indices) # embeddings tensor
```

This alternate method is the one used in the data loading process:
- in the torch DataLoader running in the background, we retrieve, for each train, the list of embeddings indices it requires for each type of embedding
- in the main thread, when the indices are aggregated and converted to embeddings values.

This two-steps process allows backpropagating on the embeddings within the training step. If we used the simple loading method based on `laplacian_embedding[pr_list]` (or, equivalently, if we successively used `get_indices` and `get_values` in the data loader), the first steps of the computational graph would happen in the data loading child processes, whereas the remainder (calling the neural network and postprocessing the output) happen in the main process. However, Pytorch is currently not able to simply track gradient operations across processes without delving into some [more sophisticated machinery](https://discuss.pytorch.org/t/multiprocessing-with-tensors-requires-grad/87475) based on `torch.distributed`.

## Embeddings of Points Remarquables (PRs)

We have 5 such embeddings: `laplacian`, `node2vec`, `random`, `geographic` and `nextdesserte_pr`. PR embeddings are flagged as such in the Embeddings class through the boolean argument `is_pr_embedding`. This flag is used to determine whether or not to interpolate the embeddings of missing PRs with the embeddings of their closest non-missing neighbors in the itinerary.

### PRs Graph

Several PR embeddings (Laplacian and Node2Vec) require to represent the railway network as a graph. The folder `PRs_graph` contains scripts to retrieve data from Cassini, which is then used to build an adjacency matrix of the PRs. This matrix is used to compute PRs embeddings, like the Laplacian embedding.

### Missing PRs embedding interpolation

The interpolation is done in three steps:
- First, **during the statistics computation step** (see [Statistics](2_statistics.md)),for each each PR whose "ID Gaia" is not among the nodes of the PRs graph:
  - The first N times we see it appear in a sillon, we record its two nearest neighbors in the sillon which are registered in the graph. We also note the theoretical times of these two PRs as well as the theoretical time of the missing PR.
- Then, **periodically during the training epochs**, we create interpolated embeddings for each missing PR by averaging the embeddings of its neighbors. The neighbors remain the same throughout the training, but the value of their embedding may evolve if the embeddings are learned.
- Finally, **when requesting embeddings values**, we provide the true embeddings for the non-missing PRs, and the interpolated embeddings for the missing PRs.

ℹ️ For missing PRs, the embeddings are stored in a dedicated embedding matrix `embedding.emb_missing_prs`, distinct from the matrix `embedding.emb` of non-missing PRs. The function `embedding.get_indices` returns values between `embedding.emb.num_embeddings` and `embedding.emb.num_embeddings + embedding.emb_missing_prs.num_embeddings` ; then, when calling `get_indices` on an index `i`:
- if `i < embedding.emb.num_embeddings`, we return the `i`-th non-missing embedding
- else, we return the `i - embedding.emb.num_embeddings`-th missing embedding

## NextDesserte_train and LignePk embeddings (legacy)

These two embeddings don't have a specific interpolation strategy. The `nextdesserte_train` embedding takes as input train numbers (numéros de circulation), and the `lignepk` embedding takes as input the "code ligne" of the trains ar each PR of their itinerary (so, technically this is a PR embedding, but the embedding keys are "code lignes", not PR IDs). The train numbers are stored in the sillons, and the "code lignes" are obtained from the PR IDs through the `transformer.resources.pr_information` module.

ℹ️ The `lignepk` embedding is completed by adding, additionally to the ligne embedding, the Point Kilométrique in this ligne (the combination ligne + PK is supposed to determine the position of a point in the railway network).

⚠️ The `nextdesserte_embedding` module produces **two** embeddings, `nextdesserte_pr` and `nextdesserte_train`.