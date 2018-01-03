import numbers

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn import model_selection

from sklearn.utils import check_random_state


def train_test_split(*arrays, **options):
    r"""Split arrays into random train and test subsets

    Notes
    -----
    Shuffling can be much more expensive

    Parameters
    ----------
    *arrays : array-like
    test_size
    train_size
    random_state
    shuffle
    stratify
    """
    # TODO(py3): keyword only arguments
    t = _validate_arrays(*arrays)
    test_size = options.pop('test_size', 'default')
    train_size = options.pop('train_size', None)
    random_state = check_random_state(options.pop("random_state", None))
    stratify = options.pop('stratify', None)
    shuffle = options.pop('shuffle', True)

    if options:
        raise TypeError("Invalid parameters passed: {}".format(options))
    if stratify:
        raise NotImplementedError("'stratify' is not currently supported.")

    if test_size == 'default':
        test_size = None

    if test_size is None and train_size is None:
        test_size = 0.25
        train_size = 0.75

    if t is np.ndarray or t is pd.DataFrame:
        return split_small(*arrays, random_state=random_state)

    if t is da.core.Array:
        return split_dask_arrays(*arrays,
                                 random_state=random_state,
                                 test_size=test_size,
                                 train_size=train_size,
                                 shuffle=shuffle)
    if issubclass(t, dd._Frame):
        return split_dask_dataframes(*arrays,
                                     random_state=random_state,
                                     test_size=test_size,
                                     train_size=train_size,
                                     shuffle=shuffle)


def _validate_arrays(*arrays):
    assert len(arrays) % 2 == 0
    assert len(arrays)
    t = type(arrays[0])
    # assert all(isinstance(arr, t) for arr in arrays)
    return t


def split_small(*arrays, **kwargs):
    return model_selection.train_test_split(*arrays, **kwargs)


def split_dask_arrays(*arrays, random_state, train_size, test_size, shuffle):
    # Let's assume that arange(n) fits in memory
    n_samples = len(arrays[0])
    assert all(len(x) == n_samples for x in arrays)
    result = []

    if shuffle is False:
        n_train, n_test = _validate_shuffle_split(n_samples,
                                                  test_size,
                                                  train_size)
        train_idx = slice(n_train)
        test_idx = slice(n_train, n_train + n_test)

    else:
        idx = np.arange(n_samples)
        train_idx, test_idx = model_selection.train_test_split(
            idx,
            random_state=random_state,
            train_size=train_size,
            test_size=test_size,
        )

    for array in arrays:
        result.append(array[train_idx])
        result.append(array[test_idx])

    return result


def split_dask_dataframes(*arrays, random_state, train_size, test_size,
                          shuffle):
    test_size, train_size = _validate_sizes_decimal(test_size, train_size)
    if shuffle is False:
        raise ValueError("'shuffle=False' is not supported for dask "
                         "dataframes.")
    if not isinstance(random_state, numbers.Integral):
        random_state = random_state.randint(2 ** 32 - 1)
    result = []
    for array in arrays:
        result.extend(array.random_split([train_size, test_size],
                                         random_state=random_state))
    return result


def _validate_shuffle_split(n_samples, test_size, train_size):
    # XXX: private method, re-implement
    from sklearn.model_selection import _split
    return _split._validate_shuffle_split(n_samples, test_size, train_size)


def _validate_sizes_decimal(test_size, train_size):
    if not isinstance(train_size, numbers.Real) or not (0 < train_size < 1):
        raise ValueError("'train_size' must be a float between 0 and 1. "
                         "Got {} instead".format(train_size))

    if test_size is None:
        test_size = 1 - train_size

    if not isinstance(test_size, numbers.Real) or not (0 < train_size < 1):
        raise ValueError("'train_size' must be a float between 0 and 1. "
                         "Got {} instead".foramt(train_size))

    return test_size, train_size


__all__ = [
    'train_test_split',
]
