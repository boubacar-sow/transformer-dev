"""
Functions to read and write different types of files.
The functions rely on the FileSystem class, which behind the scene can use
a local file system or an S3 file system. Currently, only a local file system is used,
but this system allows easily switching to something more complex.
"""

import json
import pickle
from io import BytesIO, TextIOWrapper
from typing import Any, Callable, Iterable, Optional

import numpy as np
import numpy.typing
import pandas as pd
import torch
from matplotlib.figure import Figure

from transformer.utils.file_system import FileSystem

fs = FileSystem()

#################################################################################
# .csv file
# Csv


def load_csv(filename: str, zstd_format: bool = False, **kwargs: Any) -> Any:
    """
    Load a .csv file
    Args:
        filename (str) or (file-like): file name to load, can be an open file (mounted or not)
    Returns:
        pandas frame: loaded data
    """
    buffer = fs.read_buffer(file=filename, zstd_format=zstd_format)
    data = pd.read_csv(buffer, **kwargs)
    buffer.close()
    return data


def save_csv(filename: str, data: pd.DataFrame, zstd_format: bool = False, **kwargs: Any) -> None:
    """
    Save pandas frame to csv file
    Args:
        filename (str) or (file-like): file name to save, can be an open file (mounted or not)
        data (pandas frame): data to save
        **kwargs: will be passed on to "data.to_csv"
    """
    buffer = BytesIO()
    if "index" in kwargs:
        data.to_csv(buffer, **kwargs)
    else:
        data.to_csv(buffer, index=False, **kwargs)
    fs.save_buffer(buffer=buffer, file=filename, zstd_format=zstd_format)
    buffer.close()


def load_csv_ll(filename: str, zstd_format: bool = False, **kwargs: Any) -> list[list[Any]]:
    """
    Load a .csv file
    Args:
        filename (str) or (file-like): file name to load, can be an open file (mounted or not)
    Returns:
        (list(list)): loaded data
    """
    return [list(line) for line in load_csv(filename, zstd_format=zstd_format, **kwargs).values]


def save_csv_ll(filename: str, data: list[list[Any]], sep: str = "\t", zstd_format: bool = False) -> None:
    """
    Save list of lists to csv file
    Args:
        filename (str) or (file-like): file name to save, can be an open file (mounted or not)
        data (list(list)): data to save
        **kwargs: will be passed on to "data.to_csv"
    """
    data1 = [sep.join([str(dii) for dii in di]) for di in data]
    data2 = "\n".join(data1)
    save_txt(filename, data2, zstd_format=zstd_format)


#################################################################################
# .json file
# Json


def load_json(filename: str, zstd_format: bool = False, **kwargs: Any) -> Any:
    """
    Load json file
    Args:
        filename (str) or (file-like): file name to load, can be an open file (mounted or not)
        **kwargs: will be passed on to "json.load"
    Returns:
        dict: loaded data
    """
    buffer = fs.read_buffer(file=filename, zstd_format=zstd_format)
    json_dict = json.load(buffer, **kwargs)
    buffer.close()
    return json_dict


def save_json(filename: str, data: Any, zstd_format: bool = False, indent: Optional[int] = None, **kwargs: Any) -> None:
    """
    Save a json
    Args:
        filename (str) or (file-like): file name to save, can be an open file (mounted or not)
        indent (int): if not None, the JSON will be written nicely with the given indent length.
                      If None, the JSON will be written on a single line.
        data (dict*): data to save
        **kwargs: will be passed on to "json.dump"
    """
    json_string = json.dumps(data, indent=indent, **kwargs)
    json_bytes = bytes(json_string, encoding="utf8")
    fs.save_bytes(json_bytes, filename, zstd_format=zstd_format)


#################################################################################
# .npy file
# Numpy array


def load_npy(filename: str, zstd_format: bool = False, **kwargs: Any) -> Any:
    """
    Load numpy array from a .npy file (same as np.load, but with allow_pickle=True by default)
    Args:
        filename (str) or (file-like): file name to load, can be an open file (mounted or not)
        **kwargs: will be passed on to "np.load"
    Returns:
        dict: loaded data
    """
    buffer = fs.read_buffer(file=filename)
    if "allow_pickle" in kwargs:
        data = np.load(buffer, **kwargs)
    else:
        data = np.load(buffer, allow_pickle=True, **kwargs)
    if zstd_format:
        if len(data) == 1:
            data = list(data.values())[0]
    buffer.close()
    return data


def save_npy(filename: str, data: numpy.typing.ArrayLike, zstd_format: bool = False, **kwargs: Any) -> None:
    """
    Save numpy array in a .npy file (same as np.save, but function name is changed for naming consistency)

    Args:
        filename (str) or (file-like): file name to save, can be an open file (mounted or not)
        data (numpy array): data to save
        **kwargs: will be passed on to "np.save"
    """
    buffer = BytesIO()
    if zstd_format:
        np.savez(buffer, data, **kwargs)
    else:
        np.save(buffer, data, **kwargs)
    fs.save_buffer(buffer=buffer, file=filename)
    buffer.close()


#################################################################################
# .jpg file
# Figure Matplotlib


def save_matplotlib_fig(filename: str, data: Figure, zstd_format: bool = False, **kwargs: Any) -> None:
    buffer = BytesIO()
    data.savefig(buffer, **kwargs)
    fs.save_buffer(buffer=buffer, file=filename, zstd_format=zstd_format)
    buffer.close()


#################################################################################
# .pickle file
# Python vars


def load_pickle(filename: str, zstd_format: bool = False, **kwargs: Any) -> Any:
    """
    Load a python variables using pickle
    Args:
        filename (str) or (file-like): file name to load, can be an open file (mounted or not)
        **kwargs: will be passed on to "pickle.load"
    Returns:
        what was stored in the pickle file (can be dict, list, str, ...): loaded data
    """
    buffer = fs.read_buffer(file=filename, zstd_format=zstd_format)
    data = pickle.load(buffer, **kwargs)
    buffer.close()
    return data


def save_pickle(filename: str, data: Any, zstd_format: bool = False, **kwargs: Any) -> None:
    """
    Save python variables in a .pickle file

    Args:
        filename (str) or (file-like): file name to save, can be an open file (mounted or not)
        data: data to save
        **kwargs: will be passed on to "pickle.dump"
    """
    buffer = BytesIO()
    pickle.dump(data, buffer, **kwargs)
    fs.save_buffer(buffer=buffer, file=filename, zstd_format=zstd_format)
    buffer.close()


#################################################################################
# .txt file
# Text


def load_txt(filename: str, zstd_format: bool = False) -> Any:
    """
    Load a simple text file (not binary read)
    Args:
        filename (str) or (file-like): file name to load, can be an open file (mounted or not)
        mode (str): whether to load text in binary (open(..., "rb")) or not (open(..., "r")) (default is "r")
        **kwargs: will be passed on to "file.read"
    Returns:
        str or bytes (depending on argument 'mode'): loaded data
    """
    bytes_txt = fs.read_bytes(filename, zstd_format=zstd_format)
    return str(bytes_txt, encoding="utf8")


def save_txt(filename: str, data: str, zstd_format: bool = False) -> None:
    """
    Save a simple text file (not binary read)
    filename : (str) or (file-like) or (tuple(fs,str)) : file name to load, can contain filesystem in it
    Args:
        filename (str) or (file-like): file name to save, can be an open file (mounted or not)
        data (str): data to save
        mode (str): whether to save text in binary (open(..., "wb")) or not (open(..., "w")) (default is "w")
        **kwargs: will be passed on to "file.write"
    """
    bytes_txt = bytes(data, encoding="utf8")
    fs.save_bytes(bytes_txt, filename, zstd_format=zstd_format)


def _check_start_end(start: Optional[int], end: Optional[int], length: int) -> tuple[int, int]:
    """
    Make sure that start and end are valid delimiters (no None, and 0 <= start <= end <= length)
    """
    if start is None:
        start = 0
    if end is None:
        end = length
    if start < 0:
        start = length + start
        if start < 0:
            start = 0
    if end < 0:
        end = length + end
        if end < 0:
            end = 0
    if start > length:
        start = length
    if end > length:
        end = length
    return start, end


def read_txt_part(
    f: BytesIO | TextIOWrapper,
    start: Optional[int] = None,
    end: Optional[int] = None,
    length: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Read a part of a txt open file
    Args:
        f: (file-like): open file to read
        start: index of the first char to read (default is None/start of file)
        end: index of the last char to read (default is None/end of file)
        length: length of the file, if missing it will be computed
        **kwargs: will be passed on to "file.read"
    Returns:
        str or bytes (depending on opened mode): loaded data
    """
    if length is None:
        f.seek(0, 2)
        length_ = f.tell()
    else:
        length_ = length
    start, end = _check_start_end(start, end, length_)
    cursor = f.tell()
    if cursor != start:
        f.seek(start, 0)
    data = f.read(max(0, end - start), **kwargs)
    return data


def load_txt_per_parts(
    filename: str,
    mode: str = "r",
    chunksize: int = 10**6,
    start: Optional[int] = None,
    end: Optional[int] = None,
    read_func: Optional[Callable[[BytesIO | TextIOWrapper, Optional[int], Optional[int], Optional[int]], Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Generator that loads a simple text file, part by part
    "".join(load_txt_per_parts(f, chunksize=c, start=x, end=y)) returns the same as load_txt(f)[x:y]
    Consumes less memory than load_txt since data is loaded parts by parts
    Args:
        filename (str) or (file-like): file name to load, can be an open file (mounted or not)
        mode (str): whether to load text in binary (open(..., "rb")) or not (open(..., "r")) (default is "r")
        chunksize: size of each part of txt to return (default is 10**6)
        start: index of the first char to read (default is None/start of file)
        end: index of the last char to read (default is None/end of file)
        read_func: function that read a part of the txt, default=read_txt_part
        **kwargs: will be passed on to "file.read"
    Returns:
        str or bytes (depending on argument 'mode'): loaded data
    """
    if read_func is None:
        read_func = read_txt_part
    if mode in {"rb", "r"}:
        buffer: BytesIO | TextIOWrapper
        buffer = fs.read_buffer(filename)  # type: ignore
        if mode == "r":
            buffer_ = buffer
            buffer = TextIOWrapper(buffer)  # type: ignore
        buffer.seek(0, 2)  # place "reading cursor" at the end of the file
        length = buffer.tell()  # get cursor position
        start, end = _check_start_end(start, end, length)
        buffer.seek(start, 0)
        cursor = buffer.tell()
        while cursor < end:
            part_end = cursor + min(chunksize, end - cursor)
            data = read_func(buffer, start=cursor, end=part_end, length=length, **kwargs)  # type: ignore
            cursor = buffer.tell()
            yield data
        buffer.close()
        if mode == "r":
            buffer_.close()
    else:
        raise ValueError("Mode {} not supported by load_txt (not 'r' or 'rb')".format(mode))


def load_txt_part(
    filename: str,
    mode: str = "r",
    start: Optional[int] = None,
    end: Optional[int] = None,
    read_func: Optional[Callable[[Any, Optional[int], Optional[int], Optional[int]], Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Load a simple text file, and can specify which part of the file to load
    load_txt_part(f, start=x, end=y) returns the same as load_txt(f)[x:y]
    Consumes less memory than load_txt since only the returned part of the file is loaded in memory
    Args:
        filename (str) or (file-like): file name to load, can be an open file (mounted or not)
        mode (str): whether to load text in binary (open(..., "rb")) or not (open(..., "r")) (default is "r")
        start: index of the first char to read (default is None/start of file)
        end: index of the last char to read (default is None/end of file)
        read_func: function that read a part of the txt, default=read_txt_part
        **kwargs: will be passed on to "file.read"
    Returns:
        str or bytes (depending on argument 'mode'): loaded data
    """
    if read_func is None:
        read_func = read_txt_part
    if mode in {"rb", "r"}:
        buffer: BytesIO | TextIOWrapper
        buffer = fs.read_buffer(filename)  # type: ignore
        if mode == "r":
            buffer_ = buffer
            buffer = TextIOWrapper(buffer)  # type: ignore
        data = read_func(buffer, start=start, end=end, **kwargs)  # type: ignore
        buffer.close()
        if mode == "r":
            buffer_.close()
    else:
        raise ValueError("Mode {} not supported by load_txt (not 'r' or 'rb')".format(mode))
    return data


def load_txt_l(filename: str, sep: str = "\n", dtype: type = str) -> Any:
    """
    Load a simple text file (not binary read) and tranforms it to a list of str
    Args:
        filename (str) or (file-like): file name to load, can be an open file (mounted or not)
        mode (str): whether to load text in binary (open(..., "rb")) or not (open(..., "r")) (default is "r")
        sep (str): separator to use inside the text file
        dtype (type): type of elements in the list to load
        **kwargs: will be passed on to "file.read"
    Returns:
        str or bytes (depending on argument 'mode'): loaded data
    """
    data = load_txt(filename).split(sep)
    if dtype is not str:
        return [dtype(d) for d in data]
    return data


def save_txt_l(filename: str, data: Iterable[Any], sep: str = "\n") -> None:
    """
    Load a simple text file (not binary read)
    filename : (str) or (file-like) or (tuple(fs,str)) : file name to load, can contain filesystem in it

    Args:
        filename (str) or (file-like): file name to save, can be an open file (mounted or not)
        data (numpy array): data to save
        mode (str): whether to save text in binary (open(..., "wb")) or not (open(..., "w")) (default is "w")
        sep (str): separator to use inside the text file
        **kwargs: will be passed on to "file.write"
    """
    save_txt(filename, sep.join([str(d) for d in data]))


#################################################################################
# Pytorch load and save


def load_torch(filename: str, map_location: str = "cpu") -> Any:
    buffer = fs.read_buffer(filename)
    result = torch.load(f=buffer, map_location=torch.device(map_location))
    buffer.close()
    return result


def save_torch(filename: str, data: Any) -> None:
    buffer = BytesIO()
    torch.save(obj=data, f=buffer)
    fs.save_buffer(buffer=buffer, file=filename)
    buffer.close()
