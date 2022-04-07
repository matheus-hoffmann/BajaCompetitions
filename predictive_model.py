import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, RandomizedSearchCV


def set_input(df: pd.DataFrame = None,
              safety: float = None,
              project: float = None,
              dynamics: float = None,
              endurance: float = None) -> np.ndarray:
    """
    :param df: Complete dataset.
    :param safety: Safety points.
    :param project: Project points.
    :param dynamics: Dynamic points.
    :param endurance: Endurance points.
    :return: Data combined to become input array.
    """
    iarray = np.array([safety, project, dynamics, endurance])
    if iarray[0] is None:
        iarray[0] = df['safety'].mean()
    if iarray[1] is None:
        iarray[1] = df['project'].mean()
    if iarray[2] is None:
        iarray[2] = df['dynamics'].mean()
    if iarray[3] is None:
        iarray[3] = df['endurance'].mean()
    return iarray.reshape((1, -1))


def set_model(x, y,
              train_percentage: float = 0.8,
              n_splits: int = 5,
              n_folds: int = 5,
              n_params: int = 5,
              n_repeats: int = 5):
    """
    :param x: Input
    :param y: Output
    :param train_percentage: Percentage of data to training set
    :param n_splits: Number of re-samples of data into train/test sets
    :param n_folds: Number of folds on K-Fold
    :param n_params: Number of random combinations of hyper-parameters to be tested
    :param n_repeats: Number of repetitions to mitigate random errors
    :return: Best trained model
    """
    params = {'n_estimators': range(3, 100),
              'max_depth': range(2, 10),
              'min_samples_leaf': range(2, 10)}
    best_model = None
    best_params = None
    best_score = 1e30
    best_seed = 0
    for seed in range(n_splits):
        xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                        train_size=train_percentage,
                                                        random_state=seed)
        model = RandomForestRegressor()
        cv = RepeatedKFold(n_splits=n_folds,
                           n_repeats=n_repeats,
                           random_state=seed)
        search = RandomizedSearchCV(estimator=model,
                                    param_distributions=params,
                                    cv=cv,
                                    n_iter=n_params,
                                    random_state=seed)
        search.fit(xtrain, ytrain)

        model = search.best_estimator_
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)

        residue = abs(ytest - ypred)
        score = max(residue)
        print(u"{} iter - [{} ~ {}] \u00B1 {}".format(seed, int(min(residue)), int(score), int(residue.std())))

        if best_score > score:
            best_model = model
            best_params = search.best_params_
            best_score = score
            best_seed = seed

    # Final model
    xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                    train_size=train_percentage,
                                                    random_state=best_seed)
    model = best_model
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    residue = abs(ytest - ypred)
    score = max(residue)

    print("Best Parameters: {}".format(best_params))
    print("Best seed: {}".format(best_seed))
    print("Maximum error: {}".format(int(score)))

    return model
