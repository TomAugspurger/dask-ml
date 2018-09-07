from __future__ import division

import copy
import logging
import math

import numpy as np
from sklearn.model_selection import ParameterSampler

from ._incremental import BaseIncrementalSearchCV
from ._successive_halving import _SHA

logger = logging.getLogger(__name__)


def _get_hyperband_params(R, eta=3):
    """
    Parameters
    ----------
    R : int
        The maximum number of iterations desired.

    Returns
    -------
    N : list
        The number of models for each bracket
    R : list
        The number of iterations for each bracket
    brackets : list
        The bracket identifier.
    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R

    brackets = list(reversed(range(int(s_max + 1))))
    N = [math.ceil(B / R * eta ** s / (s + 1)) for s in brackets]
    R = [int(R * eta ** -s) for s in brackets]
    return list(map(int, N)), R, brackets


class HyperbandCV(BaseIncrementalSearchCV):
    """Find the best parameters for a particular model with an adaptive
    cross-validation algorithm.

    This algorithm is performant and only requires computational budget
    as input (performant := "finds the best parameters with minimal
    ``partial_fit`` calls). It does not require a trade-off between "evaluate many
    parameters" and "train for a long time" like RandomizedSearchCV. Hyperband
    will find close to the best possible parameters with the given
    computational budget [1]_.*

    :sup:`* This will happen with high probability, and "close" means "within
    a log factor of the lower bound"`

    Parameters
    ----------
    model : object
        An object that has support for ``partial_fit``, ``get_params``,
        ``set_params`` and ``score``. This can be an instance of scikit-learn's
        BaseEstimator
    params : dict, list
        The various parameters to search over. If dict, will be fed to
        :func:`~sklearn.model_selection.ParameterSampler`. If list, each
        element will be fed to the model.
    max_iter : int
        The maximum number of partial_fit calls to any one model. This should
        be the number of ``partial_fit`` calls required for the model to
        converge. See the notes on how to set this parameter.
    aggressiveness : int, default=3
        How aggressive to be in model tuning. It is not recommended to change
        this value, and if changed we recommend ``eta=4``.
        Some theory behind Hyperband suggests ``eta=np.e``. Higher
        values imply higher confidence in model selection.
    test_size : float, optional
        Hyperband uses one test set for all example, and this controls the
        size of that test set. It should be a floating point value between 0
        and 1 to represent the number of examples to put into the test set.
    random_state : int or np.random.RandomState
        A random state for this class.
    scoring : str or callable, optional
        The scoring method by which to score different classifiers.
    verbose : int
        Controls the verbosity of this object. Higher number means reporting
        values more often.

    Examples
    --------
    >>> import numpy as np
    >>> from dask_ml.model_selection import HyperbandCV
    >>> from dask_ml.datasets import make_classification
    >>> from sklearn.linear_model import SGDClassifier
    >>>
    >>> X, y = make_classification(chunks=20)
    >>> est = SGDClassifier(tol=1e-3)
    >>> params = {'alpha': np.logspace(-4, 0, num=1000),
    >>>           'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
    >>>           'average': [True, False]}
    >>>
    >>> search = HyperbandCV(est, params)
    >>> search.fit(X, y, classes=np.unique(y))
    >>> search.best_params_
    {'loss': 'log', 'average': False, 'alpha': 0.0080502}

    Attributes
    ----------
    cv_results_ : dict of lists
        Information about the cross validation scores for each model.
        All lists are ordered the same, and this value can be imported into
        a pandas DataFrame. This dict has keys of

        * ``rank_test_score``
        * ``model_id``
        * ``mean_fit_time``
        * ``mean_score_time``
        * ``std_fit_time``
        * ``std_score_time``
        * ``mean_test_score``
        * ``test_score``
        * ``partial_fit_calls``
        * ``params``

    metadata : dict
        Information about every model to be trained.
    metadata_ : dict
        Information about every model to be trained. These may differ as
        `metadata_` contains the *actual* number of partial fit calls.
    history_ : list of dicts
        Information about every model after every time it is scored.
    best_params_ : dict
        The params that produced the best performing model
    best_estimator_ : any
        The best performing model
    best_index_ : int
        The index of the best performing model to be used in ``cv_results_``.
    n_splits_ : int
        The number of cross-validation splits.
    best_score_ : float
        The best validation score on the test set.
    best_params_ : dict
        The params that are given to the model that achieves ``best_score_``.

    Notes
    -----

    Hyperband is an adaptive model selection scheme that spends time on
    high-performing models, because our goal is to find the highest performing
    model. This means that it stops training models that perform poorly.

    Setting Hyperband parameters
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    To set ``max_iter`` and the chunk size for ``X`` and ``y``, you need to
    know

    * how many "epochs" or "passes through ``X``" to train the model for
      (``epochs`` below)
    * how many parameters to sample (``params_to_sample`` below)

    To determine the chunk size and ``max_iter``,

    1. Set ``frac = epochs / params_to_sample``, where ``frac`` is ``chunks / len(X)``
    2. Set ``max_iter = params_to_sample``

    The number of parameters to sample will depend on how complex the search
    space is. If you're tuning many parameters, you'll need to increase
    ``params_to_sample``.

    Limitations
    ^^^^^^^^^^^
    There are some limitations to the `current` implementation of Hyperband:

    1. The full dataset is requested to be in distributed memory
    2. The testing dataset must fit in the memory of a single worker

    References
    ----------
    .. [1] "Hyperband: A novel bandit-based approach to hyperparameter
           optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A.
           Rostamizadeh, and A. Talwalkar.  https://arxiv.org/abs/1603.06560

    """

    def __init__(
        self,
        model,
        params,
        max_iter,
        aggressiveness=3,
        random_state=None,
        scoring=None,
        test_size=0.15,
        patience=np.inf,
        tol=0.001,
        verbose=0,
    ):
        self.max_iter = max_iter
        self.aggressiveness = aggressiveness
        self.test_size = test_size
        self.patience = patience
        self.tol = tol
        self.verbose = verbose
        self._metadata = None

        super(HyperbandCV, self).__init__(
            model, params, scoring=scoring, random_state=random_state
        )

    def _get_ids_and_args(self):
        N, R, brackets = _get_hyperband_params(self.max_iter, eta=self.aggressiveness)
        SHAs = {
            b: _SHA(
                n,
                r,
                limit=b + 1,
                eta=self.aggressiveness,
                patience=self.patience,
                tol=self.tol,
                verbose=self.verbose,
                bracket=b,
            )
            for n, r, b in zip(N, R, brackets)
        }
        if isinstance(self.params, list):
            _params = self.params[::-1]
            param_lists = [[_params.pop() for _ in range(n)] for n in N]
        else:
            param_lists = [list(ParameterSampler(self.params, n)) for n in N]

        return {
            bracket: (param_list, SHA)
            for (bracket, SHA), param_list in zip(SHAs.items(), param_lists)
        }

    def _update_results(self, results):
        meta = copy.deepcopy(self.metadata)
        brackets = meta["brackets"]
        bracket_ids_brackets = [
            (int(bracket["bracket"].split("=")[1]), bracket) for bracket in brackets
        ]
        # we need to update iters and partial_fit_calls based on actual numbers

        for bracket_id, bracket in bracket_ids_brackets:
            hist = results[bracket_id][2]
            # this mutats meta inplace (but not self.metadata)
            bracket["iters"] = sorted(list({h["partial_fit_calls"] for h in hist}))
            bracket["partial_fit_calls"] = sum(x["partial_fit_calls"] for x in hist)

        self.metadata_ = meta

    @property
    def metadata(self):
        """Get information about how much computation is required for
        :func:`~dask_ml.model_selection.HyperbandCV.fit`. This can be called
        before or after  :func:`~dask_ml.model_selection.HyperbandCV.fit`.

        Returns
        -------
        metadata : dict
            Information about the computation performed by ``fit``. Has keys

            * ``partial_fit_calls``, the total number of partial fit calls.
            * ``models``, the total number of models created.
            * ``brackets``, each of which has the same two keys as above.

        Notes
        ------
        This algorithm runs several loops in an "embarassingly parallel"
        manner. The ``brackets`` key represents each of these loops.
        """
        if self._metadata is None:
            bracket_info = _hyperband_paper_alg(self.max_iter, eta=self.aggressiveness)
            num_models = sum(b["models"] for b in bracket_info)
            for bracket in bracket_info:
                bracket["iters"].update({1})
                bracket["iters"] = sorted(list(bracket["iters"]))
            num_partial_fit = sum(b["partial_fit_calls"] for b in bracket_info)
            bracket_info = reversed(sorted(bracket_info, key=lambda x: x["bracket"]))

            info = {
                "partial_fit_calls": num_partial_fit,
                "models": num_models,
                "brackets": list(bracket_info),
            }
            self._metadata = info
        return self._metadata


def _hyperband_paper_alg(R, eta=3):
    """
    Algorithm 1 from the Hyperband paper [1]_.

    References
    ----------
    1. "Hyperband: A novel bandit-based approach to hyperparameter
       optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh,
       and A. Talwalkar.  https://arxiv.org/abs/1603.06560
    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R
    brackets = reversed(range(int(s_max + 1)))
    hists = {}
    for s in brackets:
        n = int(math.ceil(B / R * eta ** s / (s + 1)))
        r = R * eta ** -s
        r = int(r)
        T = set(range(n))
        hist = {"num_models": n, "models": {n: 0 for n in range(n)}, "iters": []}
        for i in range(s + 1):
            n_i = math.floor(n * eta ** -i)
            r_i = np.round(r * eta ** i).astype(int)
            L = {model: r_i for model in T}
            hist["models"].update(L)
            hist["iters"] += [r_i]
            to_keep = math.floor(n_i / eta)
            T = {model for i, model in enumerate(T) if i < to_keep}
        hists["bracket={s}".format(s=s)] = hist
    info = [
        {
            "bracket": k,
            "models": hist["num_models"],
            "partial_fit_calls": sum(hist["models"].values()),
            "iters": {int(h) for h in hist["iters"]},
        }
        for k, hist in hists.items()
    ]
    return info
