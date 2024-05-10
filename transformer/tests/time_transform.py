import numpy as np

from transformer.inputs.snapshots.time_transform import TimeTransform
from transformer.tests.fixtures.mock_stats import MockAllStats
from transformer.utils.config import get_config

cfg, paths = get_config()


def test_time_transform() -> None:  # noqa: C901
    all_stats = MockAllStats()

    time_transform = TimeTransform(all_stats)

    for key in ["prev", "foll"]:
        a = np.pi
        b = time_transform.times_to_nn(a, prev_foll=key, normalize=False)
        if cfg.data.preprocessing.transform_times_sqrt in ["sqrt", True]:
            c = np.sqrt(a)
        elif cfg.data.preprocessing.transform_times_sqrt == "log":
            c = np.log(a)
        else:
            c = a
        assert b == c

        a = np.pi
        b = time_transform.times_to_nn(a, prev_foll=key, normalize=True)
        ms = time_transform.prev_foll_theo_times[key]
        if cfg.data.preprocessing.transform_times_sqrt in ["sqrt", True]:
            c = (np.sqrt(a) - ms.mean) / ms.std
        elif cfg.data.preprocessing.transform_times_sqrt == "log":
            c = (np.log(a) - ms.mean) / ms.std
        else:
            c = (a - ms.mean) / ms.std
        assert b == c

        a = np.pi
        b = time_transform.nn_to_times(a, prev_foll=key, normalize=False)
        if cfg.data.preprocessing.transform_times_sqrt in ["sqrt", True]:
            c = a**2
        elif cfg.data.preprocessing.transform_times_sqrt == "log":
            c = np.exp(a)
        else:
            c = a
        assert b == c

        a = np.pi
        b = time_transform.nn_to_times(a, prev_foll=key, normalize=True)
        ms = time_transform.prev_foll_theo_times[key]
        if cfg.data.preprocessing.transform_times_sqrt in ["sqrt", True]:
            c = (a * ms.std + ms.mean) ** 2
        elif cfg.data.preprocessing.transform_times_sqrt == "log":
            c = np.exp((a - ms.mean) / ms.std)
        else:
            c = a * ms.std + ms.mean
        assert b == c

    a = np.pi
    b = time_transform.delay_to_nn(a, normalize=False)
    if cfg.data.preprocessing.transform_delay_sqrt in ["sqrt", True]:
        c = np.sqrt(a)
    elif cfg.data.preprocessing.transform_delay_sqrt == "log":
        c = np.log(a)
    else:
        c = a
    assert b == c

    a = np.pi
    b = time_transform.delay_to_nn(a, normalize=True)
    ms = time_transform.prev_foll_theo_times[key]
    if cfg.data.preprocessing.transform_delay_sqrt in ["sqrt", True]:
        c = (np.sqrt(a) - ms.mean) / ms.std
    elif cfg.data.preprocessing.transform_delay_sqrt == "log":
        c = (np.log(a) - ms.mean) / ms.std
    else:
        c = (a - ms.mean) / ms.std
    assert b == c

    a = np.pi
    b = time_transform.nn_to_delay(a, normalize=False)
    if cfg.data.preprocessing.transform_times_sqrt in ["sqrt", True]:
        c = a**2
    elif cfg.data.preprocessing.transform_times_sqrt == "log":
        c = np.exp(a)
    else:
        c = a
    assert b == c

    a = np.pi
    b = time_transform.nn_to_delay(a, normalize=True)
    ms = time_transform.prev_foll_theo_times[key]
    if cfg.data.preprocessing.transform_times_sqrt in ["sqrt", True]:
        c = (a * ms.std + ms.mean) ** 2
    elif cfg.data.preprocessing.transform_times_sqrt == "log":
        c = np.exp((a - ms.mean) / ms.std)
    else:
        c = a * ms.std + ms.mean
    assert b == c
