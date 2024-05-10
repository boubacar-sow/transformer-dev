from pathlib import Path

import torch

from transformer.inputs.data_loading import aggregate_data_batch
from transformer.inputs.data_module import SnapshotDataModule
from transformer.inputs.data_structures import Snapshot, SnapshotBatch
from transformer.inputs.embeddings.embeddings import load_embeddings
from transformer.tests.fixtures.mock_files import mock_all_files
from transformer.tests.fixtures.mock_stats import MockAllStats
from transformer.training.modules import TransformerLightningModule, TransformerModel
from transformer.utils.config import get_config
from transformer.utils.misc import torch_device

cfg, paths = get_config()


def snapshot_to_device(snapshot: Snapshot, device: torch.device) -> None:
    """
    Puts all the tensors in a Snapshot on the given device (used when running the tests on GPU)
    """

    snapshot.x = snapshot.x.to(device)
    snapshot.y = snapshot.y.to(device)
    snapshot.mask = snapshot.mask.to(device)
    if snapshot.foll_theo_times is not None:
        snapshot.foll_theo_times = snapshot.foll_theo_times.to(device)
    if snapshot.foll_delays is not None:
        snapshot.foll_delays = snapshot.foll_delays.to(device)
    if snapshot.translation_delays is not None:
        snapshot.translation_delays = snapshot.translation_delays.to(device)
    if snapshot.train_type is not None:
        snapshot.train_type = snapshot.train_type.to(device)


def test_data_module(tmp_path: Path) -> None:
    mock_all_files(tmp_path)

    all_stats = MockAllStats()
    embeddings = load_embeddings()

    model = TransformerModel(all_stats=all_stats, embeddings=embeddings)
    data_module = SnapshotDataModule(model=model)
    train_loader = data_module.train_dataloader()
    test_loader = data_module.train_dataloader()

    snapshot_list = next(iter(train_loader))
    assert isinstance(snapshot_list[0], Snapshot)

    snapshot_list = next(iter(test_loader))
    assert isinstance(snapshot_list[0], Snapshot)

    # This is normally done by pytorch lightning
    for snapshot in snapshot_list:
        snapshot_to_device(snapshot, torch_device)

    snapshot_batch = aggregate_data_batch(snapshot_list, embeddings)
    assert isinstance(snapshot_batch, SnapshotBatch)

    lightning_module = TransformerLightningModule(model=model)
    lightning_module = lightning_module.to(torch_device)
    out = lightning_module.model(snapshot_batch.x.to(torch_device), snapshot_batch.key_mask.to(torch_device))
    assert snapshot_batch.y.shape == out.shape
