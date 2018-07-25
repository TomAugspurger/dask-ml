from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import check_scoring
from dask_ml.model_selection import _search


def main():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, train_size=0.75, test_size=0.25
    )

    tpot = TPOTClassifier(generations=10, population_size=10, verbosity=2, n_jobs=-1)
    tpot.fit(X_train, y_train)


if __name__ == "__main__":
    main()
