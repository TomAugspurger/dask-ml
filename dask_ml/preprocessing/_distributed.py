from typing import Union

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask import delayed
from distributed import Client, default_client, wait
from toolz import first, second

from tornado import gen

from ..utils import check_array

DaskData = Union[da.Array, dd.DataFrame]


@gen.coroutine
def _colocate(client: Client, X: DaskData, y: DaskData) -> None:
    # Break apart Dask.array/dataframe into chunks/parts
    data_parts = X.to_delayed()
    label_parts = y.to_delayed()

    if isinstance(data_parts, np.ndarray):
        assert data_parts.shape[1] == 1
        data_parts = data_parts.flatten().tolist()

    if isinstance(label_parts, np.ndarray):
        assert label_parts.ndim == 1 or label_parts.shape[1] == 1
        label_parts = label_parts.flatten().tolist()

    # Arrange parts into pairs.  This enforces co-locality
    parts = list(map(delayed, zip(data_parts, label_parts)))
    parts = client.compute(parts)  # Start computation in the background
    yield parts

    for part in parts:
        if part.status == "error":
            yield part  # trigger error locally
    yield wait(parts)
    raise gen.Return(parts)


def colocate(X, y, client=None):
    if client is None:
        client = default_client()
    X = check_array(X)
    y = check_array(y, ensure_2d=False)

    parts = client.sync(_colocate, client, X, y)

    xs = client.map(first, parts)
    ys = client.map(second, parts)

    if X.ndim == 1:
        chunks = X.chunks
        P = ()
    else:
        chunks = X.chunks[0]
        P = X.chunks[1][0]

    X_arrs = [
        da.from_delayed(dask.delayed(x), (chunks[i], P), X.dtype)
        for i, x in enumerate(xs)
    ]
    y_arrs = [
        da.from_delayed(dask.delayed(y_), (chunks[i],), y.dtype)
        for i, y_ in enumerate(ys)
    ]
    X2 = da.concatenate(X_arrs)
    y2 = da.concatenate(y_arrs)
    return X2, y2
