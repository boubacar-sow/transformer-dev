import numpy as np

from transformer.utils.stats_counter import StatsCounter


def inc(x: float) -> float:
    return x + 2


def dec(x: float) -> float:
    return x - 2


def test_stats_counter() -> None:
    a = [1, 1, 2, 4]
    sc = StatsCounter()
    sc.add(a)
    assert sc.min() == 1
    assert sc.max() == 4

    sc2 = sc.minimum(1.5)
    assert sc2.min() == 1.5
    sc2 = sc.minimum(0.5)
    assert sc2.min() == 1
    sc2 = sc.maximum(3)
    assert sc2.max() == 3
    sc2 = sc.maximum(5)
    assert sc2.max() == 4

    sc2 = sc.filter(min_key=2)
    assert dict(sc2) == {2: 1, 4: 1}
    sc2 = sc.filter(max_key=2)
    assert dict(sc2) == {1: 2}
    sc2 = sc.filter(min_key=2, max_key=4)
    assert dict(sc2) == {2: 1}

    assert sc.mean(transform=dec) == 0
    assert sc.std(transform=dec) == np.std(a)
    assert sc.median() == 1.5

    assert not sc.has_float()
    sc2 = StatsCounter([1.4, -0.2])
    assert sc2.has_float()

    assert sc.to_pos_list() == [0, 2, 1, 0, 1]
    assert sc.to_cumsum_pos_list() == [0, 2, 3, 3, 4]
    assert sc.to_neg_list() == []

    sc2 = StatsCounter([-1, -1, -2, -4])
    assert sc2.to_neg_list() == [1, 0, 1, 2, 0]
    assert sc2.to_cumsum_neg_list() == [1, 1, 2, 4, 4]

    sc2 = StatsCounter([-1, -1, -2, -4])
    sc2.add(sc)
    assert sc2.to_list() == [1, 0, 1, 2, 0, 2, 1, 0, 1]
    assert sc2.to_cumsum_list() == [1, 1, 2, 4, 4, 6, 7, 7, 8]

    assert sc2.percent(0) == 50
    assert sc2.nbr() == 8

    assert sc2.middle_nbr(lower_bound=0.25, upper_bound=0.75) == 4

    assert sc2.middle_mean(lower_bound=0.25, upper_bound=0.75, transform=lambda x: x**2) == (1 + 2**2) / 2
    assert sc2.middle_std(lower_bound=0.25, upper_bound=0.75, transform=lambda x: x**2) == np.sqrt(
        ((1 - 2.5) ** 2 + (4 - 2.5) ** 2) / 2
    )
