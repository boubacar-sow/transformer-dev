"""
Manual script to generate small snapshots for tests using real snapshots
"""

from datetime import datetime, timedelta

from transformer.utils.config import get_config
from transformer.utils.loaders import load_json, save_json

cfg, paths = get_config()

if __name__ == "__main__":
    snapshot = load_json(paths.snapshot_f.format("2022-01-01", "2022-01-01T11:05:00+00:00"), zstd_format=True)
    dt = datetime.fromisoformat(snapshot[0]["datetime"])
    time_deltas = [timedelta(hours=i) for i in range(4)]
    days = ["2023-11-17", "2023-11-17", "2023-11-18", "2023-11-18"]
    names = list("abcd")

    for i in range(4):
        new_time = (dt + time_deltas[i]).isoformat()
        print(names[i], new_time)
        new_snapshot = snapshot[i * 10 : (i + 1) * 10]
        for j in range(len(new_snapshot)):
            new_snapshot[i]["datetime"] = new_time
        save_json(f"transformer/tests/fixtures/data/snapshots/{names[i]}.json.zstd", new_snapshot, zstd_format=True)
