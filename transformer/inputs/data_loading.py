"""
This file contains the function data_loader, and auxiliary functions:
- build_input_embeddings concatenates the embeddings
- build_input_tensor concatenates the embeddings with the other input data
- data_loader returns a Snapshot object, containing the input tensor, the output
    tensor, and additional data (e.g., time)
"""

import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, TypeVar

import torch
from torch import nn

from transformer.inputs.data_structures import MeanStd, SillonChunk, Snapshot, SnapshotBatch
from transformer.inputs.embeddings.embeddings import Embedding
from transformer.inputs.resources.train_type import get_train_type_eigenvect, get_train_type_str
from transformer.inputs.snapshots.snapshots_files import load_raw_snapshot
from transformer.inputs.snapshots.statistics import AllStats
from transformer.inputs.snapshots.time_transform import TimeTransform
from transformer.inputs.utils.train_num_filters import TrainNumFilter
from transformer.utils.config import get_config
from transformer.utils.logger import logger
from transformer.utils.misc import torch_device
from transformer.utils.times import TIMEZONE_FRANCE, day_to_sine_encoding

cfg, paths = get_config()


T = TypeVar("T")


def _impute_none(x: Optional[T], default: T) -> T:
    return default if x is None else x


def build_embeddings_indices(
    raw_snapshot: list[SillonChunk], embeddings: dict[str, Embedding]
) -> dict[str, torch.Tensor]:
    use_inputs = cfg.model.use_inputs

    result = {}

    pr_embeddings = {
        "nextdesserte_pr": use_inputs.pr_nextdesserte_embedding,
        "laplacian_pr": use_inputs.laplacian_embedding,
        "random_pr": use_inputs.random_embedding,
        "geographic_pr": use_inputs.geographic_embedding,
        "node2vec_pr": use_inputs.node2vec_embedding,
    }
    pr_embeddings_list = [k for k, v in pr_embeddings.items() if v]

    if use_inputs.train_nextdesserte_embedding:
        result["nextdesserte_train"] = (
            embeddings["nextdesserte_train"]
            .get_indices([t.train_num for t in raw_snapshot])
            .view(len(raw_snapshot), -1)
        )

    if use_inputs.lignepk_embedding:
        result["lignepk"] = torch.stack(
            [embeddings["lignepk"].get_indices(t.prev_codes_ligne + t.foll_codes_ligne) for t in raw_snapshot]
        ).view(len(raw_snapshot), -1)

    values = defaultdict(list)
    for t in raw_snapshot:
        prs = t.prev_prs + t.foll_prs
        for embedding_name in pr_embeddings_list:
            values[embedding_name].append(embeddings[embedding_name].get_indices(prs))

    for embedding_name in pr_embeddings_list:
        result[embedding_name] = torch.stack(values[embedding_name]).view(len(raw_snapshot), -1)

    return result


def build_input_tensor(  # noqa: C901
    raw_snapshot: list[SillonChunk],
    prev_theo_times: torch.Tensor,
    prev_delays: torch.Tensor,
    foll_times: torch.Tensor,
    train_type: Optional[torch.Tensor],
    day: str,
    snapshot_time: int,
    all_stats: AllStats,
    time_transform: TimeTransform,
) -> torch.Tensor:
    """
    Concatenates the various elements forming the input tensor to the model.
    Sub-function used in data_loader()
    """

    use_inputs = cfg.model.use_inputs
    device = cfg.data.data_loader_device

    normalize_all = cfg.data.preprocessing.normalize_all_inputs
    normalize_non_eigen = normalize_all or cfg.data.preprocessing.normalize_non_eigen_inputs

    if use_inputs.nbr_train_now:
        if normalize_non_eigen:
            nbr_train_now_mean, nbr_train_now_std = all_stats.nbr_train.mean, all_stats.nbr_train.std
            # nbr_trains = torch.tensor(
            #     [(t['prev_types'][-1] != 'T') and (t['foll_types'][0] != 'O') for t in raw_data],
            #     dtype=torch.float, device=device).sum()
            nbr_trains = torch.tensor(
                [(t.prev_prs[-1] != "preDeparture") and (t.foll_prs[0] != "postArrival") for t in raw_snapshot],
                dtype=torch.float,
                device=device,
            ).sum()
            nbr_trains -= nbr_train_now_mean
            nbr_trains /= nbr_train_now_std
        else:
            # nbr_trains = (torch.tensor(
            #     [(t['prev_types'][-1] != 'T') and (t['foll_types'][0] != 'O') for t in raw_data],
            #     dtype=torch.float, device=device).sum() - 1000)/1000. # normalisation a la louche
            nbr_trains = (
                torch.tensor(
                    [(t.prev_prs[-1] != "preDeparture") and (t.foll_prs[0] != "postArrival") for t in raw_snapshot],
                    dtype=torch.float,
                    device=device,
                ).sum()
                - 1000
            ) / 1000.0  # normalisation a la louche
        nbr_trains = torch.full((len(raw_snapshot), 1), nbr_trains, dtype=torch.float, device=device)

    if use_inputs.year_day:
        year_day = torch.from_numpy(day_to_sine_encoding(day))
        year_day = year_day[None, :].repeat(len(raw_snapshot), 1)

    if use_inputs.week_day:
        week_day_: float
        if normalize_non_eigen:
            week_day_mean, week_day_std = all_stats.week_day.mean, all_stats.week_day.std

            # week_day = time.strptime(filename.rsplit("/", 2)[-2], "%Y-%m-%d").tm_wday
            week_day_ = time.strptime(day, "%Y-%m-%d").tm_wday
            week_day_ -= week_day_mean
            week_day_ /= week_day_std
        else:
            # week_day = (time.strptime(filename.rsplit("/", 2)[-2], "%Y-%m-%d").tm_wday - 3)/2.
            week_day_ = (time.strptime(day, "%Y-%m-%d").tm_wday - 3) / 2.0
        week_day = torch.full((len(raw_snapshot), 1), week_day_, dtype=torch.float, device=device)

    if use_inputs.current_time:
        current_time_: float
        if normalize_non_eigen:
            time_mean, time_std = all_stats.current_time.mean, all_stats.current_time.std

            # hour = int(filename.rsplit("/", 1)[-1][:-11])
            current_time_ = int(snapshot_time)
            current_time_ -= time_mean
            current_time_ /= time_std
        else:
            # hour = (int(filename.rsplit("/", 1)[-1][:-11]) - 800)/500. # normalisation a la louche
            current_time_ = (int(snapshot_time) - 800) / 500.0  # normalisation a la louche
        current_time = torch.full((len(raw_snapshot), 1), current_time_, dtype=torch.float, device=device)

    if use_inputs.diff_delay:
        _dlm, _dls = all_stats.avg_inter_pr_diff_delay
        diff_delay_mean_mean, diff_delay_mean_std = _dlm.mean, _dlm.std
        diff_delay_std_mean, diff_delay_std_std = _dls.mean, _dls.std
        default_mean_std = MeanStd(diff_delay_mean_mean, diff_delay_std_mean)

        diff_delay_mean_: list[list[float]] = []
        diff_delay_std_: list[list[float]] = []
        for t in raw_snapshot:
            prs = t.prev_prs + t.foll_prs
            type_train = get_train_type_str(t.train_num)
            inter_prs_keys = [(pr1, pr2, type_train) for pr1, pr2 in zip(prs[:-1], prs[1:])]
            diff_delay_mean_.append(
                [
                    _impute_none(all_stats.inter_pr.diff_delay.get(k, None), default_mean_std).mean
                    for k in inter_prs_keys
                ]
            )
            diff_delay_std_.append(
                [_impute_none(all_stats.inter_pr.diff_delay.get(k, None), default_mean_std).std for k in inter_prs_keys]
            )
        diff_delay_mean = torch.tensor(diff_delay_mean_, dtype=torch.float32)
        diff_delay_mean -= diff_delay_mean_mean  # type: ignore
        diff_delay_mean /= diff_delay_mean_std
        diff_delay_std = torch.tensor(diff_delay_std_, dtype=torch.float32)
        diff_delay_std -= diff_delay_std_mean  # type: ignore
        diff_delay_std /= diff_delay_std_std

    if use_inputs.middle_diff_delay:
        _mdlm, _mdls = all_stats.avg_inter_pr_middle_diff_delay
        middle_diff_delay_mean_mean, middle_diff_delay_mean_std = _mdlm.mean, _mdlm.std
        middle_diff_delay_std_mean, middle_diff_delay_std_std = _mdls.mean, _mdls.std
        default_middle_mean_std = MeanStd(middle_diff_delay_mean_mean, middle_diff_delay_std_mean)

        middle_diff_delay_mean_: list[list[float]] = []
        middle_diff_delay_std_: list[list[float]] = []
        for t in raw_snapshot:
            prs = t.prev_prs + t.foll_prs
            type_train = get_train_type_str(t.train_num)
            inter_prs_keys = [(pr1, pr2, type_train) for pr1, pr2 in zip(prs[:-1], prs[1:])]
            middle_diff_delay_mean_.append(
                [
                    _impute_none(all_stats.inter_pr.middle_diff_delay.get(k, None), default_middle_mean_std).mean
                    for k in inter_prs_keys
                ]
            )
            middle_diff_delay_std_.append(
                [
                    _impute_none(all_stats.inter_pr.middle_diff_delay.get(k, None), default_middle_mean_std).std
                    for k in inter_prs_keys
                ]
            )
        middle_diff_delay_mean = torch.tensor(middle_diff_delay_mean_, dtype=torch.float32)
        middle_diff_delay_mean -= middle_diff_delay_mean_mean  # type: ignore
        middle_diff_delay_mean /= middle_diff_delay_mean_std
        middle_diff_delay_std = torch.tensor(middle_diff_delay_std_, dtype=torch.float32)
        middle_diff_delay_std -= middle_diff_delay_std_mean  # type: ignore
        middle_diff_delay_std /= middle_diff_delay_std_std

    prev_theo_times = time_transform.times_to_nn(prev_theo_times, prev_foll="prev")
    foll_times = time_transform.times_to_nn(foll_times, prev_foll="foll")
    prev_delays = time_transform.delay_to_nn(prev_delays)

    if use_inputs.lignepk_embedding:
        pk_mean, pk_std = all_stats.pk.mean, all_stats.pk.std

        pks = torch.tensor(
            [t.prev_pks + t.foll_pks for t in raw_snapshot],
            dtype=torch.float32,
            device=device,
        )
        pks -= pk_mean  # type: ignore
        pks /= pk_std

    times = torch.cat((prev_theo_times, foll_times), dim=-1)

    # The total dim if the input vector is equal to INPUT_DIM
    input_vector = torch.cat(
        ()
        + (prev_delays, times)
        + ((train_type,) if use_inputs.train_type else ())
        + ((nbr_trains,) if use_inputs.nbr_train_now else ())
        + ((year_day,) if use_inputs.year_day else ())
        + ((week_day,) if use_inputs.week_day else ())
        + ((current_time,) if use_inputs.current_time else ())
        + ((diff_delay_mean, diff_delay_std) if use_inputs.diff_delay else ())
        + ((middle_diff_delay_mean, middle_diff_delay_std) if use_inputs.middle_diff_delay else ())
        + ((pks,) if use_inputs.lignepk_embedding else ()),
        dim=1,
    )

    return input_vector


def load_snapshot(
    filename: str,
    train_num_filter: TrainNumFilter,
    embeddings: dict[str, Embedding],
    all_stats: AllStats,
    time_transform: TimeTransform,
    return_translation: bool = False,
    return_foll_theo_times: bool = False,
    return_foll_delays: bool = False,
    return_train_type: bool = False,
    return_train_num: bool = False,
    return_pr_cich: bool = False,
) -> Snapshot:
    """
    Loads each training file (i.e. a snapshot), transforms it into tensors
    Also adds other inputs, such as embeddings, means of delay for a pr, etc
    This is called by a torch Dataset, therefore in parallel inside multiple threads and on cpu

    The return object is a dict with the following fields:
    - inVect : the embeddings of the (train, PR) pairs for each train in the snapshot
               These vectors are the direct inputs of the transformer
    - outVect : the true delay of the 40(=NFOLL) next PR crossed by each train
                this is the ground truth for the output of the transformer
    - mask : boolean array saying, for each train and each of the NFOLL next stops, which
             one is before the arrival (True) and after (False). Typically, in practice, if there
             is only one PR remaining in the itinerary, the mask for this train will look like:
             [True, False, ..., False]
    - is_td (optional) : for each train, boolean True if the train state is T (terminus) or D (departure)
    - translationLate (optional) : delay predicted by translation
    """

    cfg_data = cfg.data
    # which variables are used as input features of each train
    device = cfg_data.data_loader_device

    raw_snapshot = load_raw_snapshot(filename)

    snapshot_datetime = datetime.fromisoformat(raw_snapshot[0].datetime).astimezone(TIMEZONE_FRANCE)
    snapshot_day = snapshot_datetime.strftime("%Y-%m-%d")
    snapshot_time = snapshot_datetime.hour * 60 + snapshot_datetime.minute

    first_sillon = raw_snapshot[:1]
    raw_snapshot = [t for t in raw_snapshot if train_num_filter(t.train_num)]
    if len(raw_snapshot) == 0:
        raw_snapshot = first_sillon
    del first_sillon

    logger.debug([raw_snapshot[0].train_num, raw_snapshot[0].circulation_id])

    # compute mask
    mask = torch.tensor(
        [
            [(not math.isnan(x)) and pr != "postArrival" for pr, x in zip(t.foll_prs, t.foll_obs_times)]
            for t in raw_snapshot
        ],
        dtype=torch.bool,
        device=device,
    )

    prev_theo_times = torch.tensor([t.prev_theo_times for t in raw_snapshot], dtype=torch.float, device=device)
    foll_theo_times = torch.tensor([t.foll_theo_times for t in raw_snapshot], dtype=torch.float, device=device)
    prev_delays = torch.tensor([t.prev_delays for t in raw_snapshot], dtype=torch.float, device=device)
    foll_delays = torch.tensor([t.foll_delays for t in raw_snapshot], dtype=torch.float, device=device)

    try:
        # Only define the translation if it is used
        if return_translation or cfg_data.preprocessing.remove_translation_in_outputs:
            translation_delays = torch.zeros_like(foll_delays, device=device).T
            # translation = the projected delay is the last observed delay (prev_delays = past delays)
            translation_delays += prev_delays[:, -1]
            translation_delays = translation_delays.T
        else:
            translation_delays = None
    except IndexError:
        logger.error("IndexError in data_loader")
        logger.error(foll_delays)
        logger.error(f"shape: {foll_delays.shape}")
        logger.error(prev_delays)
        logger.error(f"shape: {prev_delays.shape}")
        logger.error(len(raw_snapshot))
        raise

    raw_foll_theo_times = foll_theo_times.clone() if return_foll_theo_times else None
    raw_foll_delays = foll_delays.clone() if return_foll_delays else None
    train_num = [t.train_num for t in raw_snapshot] if return_train_num else None

    if cfg_data.preprocessing.remove_translation_in_outputs:
        foll_delays -= translation_delays

    if cfg.model.use_inputs.train_type or return_train_type:
        train_type = torch.vstack([get_train_type_eigenvect(t.train_num) for t in raw_snapshot])

        if cfg_data.preprocessing.normalize_all_inputs:
            train_type_mean = torch.tensor(all_stats.train_type.mean).unsqueeze(0)
            train_type_std = torch.tensor(all_stats.train_type.std).unsqueeze(0)

            train_type -= train_type_mean
            train_type /= train_type_std
    else:
        train_type = None

    input_tensor = build_input_tensor(
        raw_snapshot,
        prev_theo_times,
        prev_delays,
        foll_theo_times,
        train_type,
        snapshot_day,
        snapshot_time,
        all_stats,
        time_transform,
    )

    embeddings_indices = build_embeddings_indices(raw_snapshot, embeddings)

    foll_delays = time_transform.delay_to_nn(foll_delays) * mask
    output_tensor = foll_delays

    if return_pr_cich:
        prev_prs_cich = [t.prev_prs for t in raw_snapshot]
        foll_prs_cich = [t.foll_prs for t in raw_snapshot]
    else:
        prev_prs_cich = None
        foll_prs_cich = None

    loaded_data = Snapshot(
        x=input_tensor,
        y=output_tensor,
        embeddings_indices=embeddings_indices,
        mask=mask,
        foll_theo_times=raw_foll_theo_times,
        foll_delays=raw_foll_delays,
        translation_delays=translation_delays,
        train_type=train_type,
        train_num=train_num,
        prev_prs_cich=prev_prs_cich,
        foll_prs_cich=foll_prs_cich,
        day=snapshot_day,
        time=snapshot_time,
    )

    return loaded_data


def _aggregate_key(data_batch: list[Snapshot], key: str, pad_sequence: bool = False) -> Any:
    if data_batch[0].__getattribute__(key) is not None:
        sequence = [e.__getattribute__(key) for e in data_batch]
        if pad_sequence:
            return nn.utils.rnn.pad_sequence(sequence)
        else:
            return sequence
    else:
        return None


def aggregate_data_batch(data_batch: list[Snapshot], embeddings: dict[str, Embedding]) -> SnapshotBatch:
    """
    Pads inputs and outputs accross batches before being fed to the model
    Called inside train and test phases

    data_batch: list of snapshots, as produced by data_loader. Each snapshot contains a different number of trains.
    """

    x0 = [e.x for e in data_batch]
    x1 = nn.utils.rnn.pack_sequence(x0, enforce_sorted=False)
    x, lengths = nn.utils.rnn.pad_packed_sequence(x1)
    # x has shape (l, n, d) where l is max length (max number of trains across the snapshots in the batch),
    # n is batch size and d is embedding dimension
    # x[i,j] = embedding vector for train i in snapshot j

    mask0 = [e.mask for e in data_batch]
    mask = nn.utils.rnn.pad_sequence(mask0, padding_value=False)
    # mask has shape (l, n, NFOLL)
    # mask[i,j,k] = True iff train i in snapshot j has still at least k PRs to go before arrival

    key_mask = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.long)
    key_mask += torch.arange(x.shape[0])
    key_mask = key_mask.T >= lengths
    key_mask = key_mask.T.to(torch_device)
    # key_mask has shape (n, l)
    # key_mask[i,j] = True iff snapshot i contains less that j trains

    y = _aggregate_key(data_batch, key="y", pad_sequence=True)
    y *= mask
    # y has shape (l, n, NFOLL)
    # y[i,j,k] = delay to predict for train j at PR k in snapshot i

    day = [e.day for e in data_batch]
    time = [e.time for e in data_batch]

    # foll_theo_times, foll_delays and translation_delay have the same shape as y
    foll_theo_times = _aggregate_key(data_batch, key="foll_theo_times", pad_sequence=True)
    foll_delays = _aggregate_key(data_batch, key="foll_delays", pad_sequence=True)
    translation_delays = _aggregate_key(data_batch, key="translation_delays", pad_sequence=True)
    if translation_delays is not None:
        translation_delays *= mask
    train_type = _aggregate_key(data_batch, key="train_type", pad_sequence=True)
    train_num = _aggregate_key(data_batch, key="train_num", pad_sequence=False)
    prev_prs_cich = _aggregate_key(data_batch, key="prev_prs_cich", pad_sequence=False)
    foll_prs_cich = _aggregate_key(data_batch, key="foll_prs_cich", pad_sequence=False)

    embeddings_values = []
    for key in data_batch[0].embeddings_indices.keys():
        embedding_indices = nn.utils.rnn.pad_sequence([e.embeddings_indices[key] for e in data_batch]).to(
            dtype=int, device=torch_device
        )
        embeddings_values.append(embeddings[key].get_values(embedding_indices).view(x.shape[0], x.shape[1], -1))

    x = torch.cat((x, *embeddings_values), dim=-1)

    return SnapshotBatch(
        x=x,
        y=y,
        key_mask=key_mask,
        obs_mask=mask,
        day=day,
        time=time,
        foll_theo_times=foll_theo_times,
        foll_delays=foll_delays,
        translation_delays=translation_delays,
        train_type=train_type,
        train_num=train_num,
        prev_prs_cich=prev_prs_cich,
        foll_prs_cich=foll_prs_cich,
    )
