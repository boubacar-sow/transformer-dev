import calendar
import datetime
import time
from functools import lru_cache
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from pytz import timezone
from zoneinfo import ZoneInfo

from transformer.utils.config import get_config

cfg, _ = get_config()

TIMEZONE_FRANCE = ZoneInfo("Europe/Paris")


def get_local_timezone_diff(date: str | datetime.datetime, fmt: Optional[str] = "%Y-%m-%d %H:%M:%S") -> int:
    utc = timezone("UTC")
    loc = timezone("Europe/Paris")
    if isinstance(date, str) and isinstance(fmt, str):
        date_ = datetime.datetime.strptime(date, fmt)
    elif isinstance(date, datetime.datetime):
        date_ = date
    else:
        raise Exception()
    return (utc.localize(date_) - loc.localize(date_).astimezone(utc)).seconds


def gm_time_to_gm_date(obs_time: float, fmt_out: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    same as strftime(localtime()) but always take the local timezone of Paris wherever the machine is
    """
    return time.strftime(fmt_out, time.gmtime(obs_time))


def gm_date_to_gm_time(obs_date: str, fmt_in: str = "%Y-%m-%d %H:%M:%S") -> int:
    obs_date = obs_date.split(".", 1)[0].split("Z", 1)[0]
    return int(calendar.timegm(time.strptime(obs_date, fmt_in)))


def gm_time_to_local_date(obs_time: float, fmt_out: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    same as strftime(localtime()) but always take the local timezone of Paris wherever the machine is
    """
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(obs_time))
    return time.strftime(fmt_out, time.gmtime(obs_time + get_local_timezone_diff(date)))


def local_date_to_gm_time(obs_date: str, fmt_in: str = "%Y-%m-%d %H:%M:%S") -> int:
    """
    same as mktime(strptime()) but always take the local timezone of Paris wherever the machine is
    """
    obs_date = obs_date.split(".", 1)[0].split("Z", 1)[0]
    return int(calendar.timegm(time.strptime(obs_date, fmt_in)) - get_local_timezone_diff(obs_date, fmt=fmt_in))


def gm_time_to_local_time(obs_time: float) -> float:
    """
    Convert to local time (allows to do obs_time % (24*60*60) to get the hour in [0,24[ range)
    """
    datetime.datetime.fromtimestamp(obs_time)
    date = datetime.datetime.fromtimestamp(obs_time)
    return float(obs_time + get_local_timezone_diff(date, fmt=None))


def current_gm_day(fmt: str = "%Y-%m-%d") -> str:
    return gm_time_to_gm_date(time.time(), fmt_out=fmt)


def current_local_day(fmt: str = "%Y-%m-%d") -> str:
    return gm_time_to_local_date(time.time(), fmt_out=fmt)


def next_day(
    day: str | list[str], offset: int = 1, fmt_in: str = "%Y-%m-%d", fmt_out: str = "%Y-%m-%d"
) -> str | list[str]:
    def aux(day: str) -> str:
        t = time.mktime(time.strptime(day, fmt_in))
        t += int(24 * 60 * 60 * offset)
        return time.strftime(fmt_out, time.localtime(t))

    if not isinstance(day, str):
        return sorted({aux(d) for d in day})
    else:
        return aux(day)


def next_days(
    day: str | list[str], offsets: list[int] = [1], fmt_in: str = "%Y-%m-%d", fmt_out: str = "%Y-%m-%d"
) -> list[str] | list[list[str]]:
    def aux(day: str) -> list[str]:
        t = time.mktime(time.strptime(day, fmt_in))
        return [time.strftime(fmt_out, time.localtime(t + int(24 * 60 * 60 * offset))) for offset in offsets]

    if isinstance(day, str):
        return aux(day)
    else:
        return sorted(
            set.union(*[set(next_days(d, offsets=offsets, fmt_in=fmt_in, fmt_out=fmt_out)) for d in day])  # type: ignore
        )


def days_around(
    day: str | list[str],
    offset_min: int = 1,
    offset_max: Optional[int] = None,
    fmt_in: str = "%Y-%m-%d",
    fmt_out: str = "%Y-%m-%d",
) -> list[str]:
    if not isinstance(day, str):
        return sorted(
            set.union(
                *[
                    set(days_around(d, offset_min=offset_min, offset_max=offset_max, fmt_in=fmt_in, fmt_out=fmt_out))
                    for d in day
                ]
            )
        )
    if offset_max is None:
        offset_max = offset_min + 1
    return next_days(day, range(-offset_min, offset_max), fmt_in=fmt_in, fmt_out=fmt_out)  # type: ignore


def days_between(day_min: str, day_max: str, fmt_in: str = "%Y-%m-%d", fmt_out: str = "%Y-%m-%d") -> list[str]:
    t_min = time.mktime(time.strptime(day_min, fmt_in))
    t_max = time.mktime(time.strptime(day_max, fmt_in))
    if t_min > t_max:
        return []
    offset = int((t_max - t_min) // (24 * 60 * 60))
    return next_days(day_min, list(range(0, offset)), fmt_in=fmt_in, fmt_out=fmt_out)  # type: ignore


def day1_minus_day2(day1: str, day2: str, fmt: str = "%Y-%m-%d") -> float:
    t1 = time.mktime(time.strptime(day1, fmt))
    t2 = time.mktime(time.strptime(day2, fmt))
    return (t1 - t2) / (24 * 60 * 60)


def select_days(days: list[str], min_date: str, max_date: str) -> list[str]:
    if isinstance(min_date, str):
        return sorted((d for d in days if (d >= min_date) and (d < max_date)))
    else:

        def f(d):
            return any((d >= mi) and (d < ma) for mi, ma in zip(min_date, max_date))

        return sorted((d for d in days if f(d)))


@lru_cache()
def get_day_to_sine_encoding_table(year_day_dim: int) -> dict[str, ArrayLike]:
    day_to_sine_encoding_table: dict[str, ArrayLike] = {}
    days_bissextile = days_between("2000-01-01", "2001-01-01")
    for i, day in enumerate(days_bissextile):
        day_as_float = 2 * np.pi * i / 366.0
        arange = 2 ** np.arange(year_day_dim)
        day_to_sine_encoding_table[day[5:]] = np.concatenate(
            [
                np.cos(day_as_float * arange),
                np.sin(day_as_float * arange),
            ],
            dtype=np.float32,
        )
    return day_to_sine_encoding_table


def day_to_sine_encoding(day: str) -> ArrayLike:
    """
    Convert a day of the year, e.g. "2018-01-03", to a vector that represent it.
    The idea is to use a representation similar  to positional encoding in the Transformer
    of Vaswani et al. 2017 : https://arxiv.org/pdf/1706.03762.pdf#subsection.3.5
    """
    day_to_sine_encoding_table = get_day_to_sine_encoding_table(cfg.model.use_inputs.year_day_dim)
    return day_to_sine_encoding_table[day[5:]]


def get_train_validation_test_days() -> tuple[list[str], list[str]]:
    cfg_days = cfg.data.days
    days_test = days_between(cfg_days.min_test, cfg_days.max_test)
    days_train_validation = days_between(cfg_days.min_train, cfg_days.max_train)
    days_train_validation = sorted((d for d in days_train_validation if (d not in days_test)))
    return days_train_validation, days_test
