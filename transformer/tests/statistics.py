from pathlib import Path

from transformer.inputs.snapshots.statistics import compute_statistics, load_statistics
from transformer.tests.fixtures.mock_files import mock_all_files


def test_compute_statistics(tmp_path: Path) -> None:
    """
    Statistics computing pipeline.

    Caution: This test does not validate the correctness of the statistics computation.
    """
    mock_all_files(tmp_path)

    compute_statistics()
    load_statistics()
