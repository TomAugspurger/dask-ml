import pytest
import sklearn.model_selection
import dask.array.utils
import dask.dataframe.utils
import dask.dataframe as dd
from scipy import stats
from sklearn.svm import SVC

import dask_ml.model_selection
from dask_ml.datasets import make_blobs


def test_search_basic(xy_classification):
    X, y = xy_classification
    param_grid = {'class_weight': [None, 'balanced']}

    a = dask_ml.model_selection.GridSearchCV(SVC(kernel='rbf'), param_grid)
    a.fit(X, y)

    param_dist = {'C': stats.uniform}
    b = dask_ml.model_selection.RandomizedSearchCV(SVC(kernel='rbf'),
                                                   param_dist)
    b.fit(X, y)


@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('train_size', [25, 0.25, None])
def test_train_test_split(shuffle, train_size):
    X, y = make_blobs(chunks=50)

    assert_eq = dask.array.utils.assert_eq

    result = dask_ml.model_selection.train_test_split(X, y, random_state=0,
                                                      shuffle=shuffle,
                                                      train_size=train_size)
    expected = sklearn.model_selection.train_test_split(X.compute(),
                                                        y.compute(),
                                                        random_state=0,
                                                        shuffle=shuffle,
                                                        train_size=train_size)

    assert len(result) == len(expected)
    for r, x in zip(result, expected):
        assert_eq(r, x)


def test_train_test_split_frame():
    X, y = map(dd.from_dask_array, make_blobs(chunks=50))

    dask_ml.model_selection.train_test_split(X, y)
    dask_ml.model_selection.train_test_split(X, y, train_size=0.5)

    r1 = dask_ml.model_selection.train_test_split(X, y, random_state=0)
    r2 = dask_ml.model_selection.train_test_split(X, y, random_state=0)

    for a, b in zip(r1, r2):
        dask.dataframe.utils.assert_eq(a, b)
