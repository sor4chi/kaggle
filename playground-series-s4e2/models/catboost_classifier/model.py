import catboost
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from constant import SEED
from utils import get_mapping


class Config:
    def __init__(
        self,
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_strength=1,
        bagging_temperature=1,
        od_type="IncToDec",
        od_wait=10,
        bootstrap_type="Bayesian",
        l2_leaf_reg=3,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.od_type = od_type
        self.od_wait = od_wait
        self.bootstrap_type = bootstrap_type
        self.l2_leaf_reg = l2_leaf_reg


class Model:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        self.model = catboost.CatBoostClassifier(
            iterations=self.config.iterations,
            depth=self.config.depth,
            learning_rate=self.config.learning_rate,
            random_strength=self.config.random_strength,
            bagging_temperature=self.config.bagging_temperature,
            od_type=self.config.od_type,
            od_wait=self.config.od_wait,
            verbose=0,
            random_state=SEED,
        )
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=2)

    def predict(self, X):
        return self.model.predict(X).flatten()


def optimize(X, y, n_trials=100):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
            "od_wait": trial.suggest_int("od_wait", 10, 50),
            "verbose": 0,
            "random_state": SEED,
        }
        model = catboost.CatBoostClassifier(**params)
        score = cross_val_score(
            model, X, y, cv=5, scoring=make_scorer(accuracy_score)
        ).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials)
    return study.best_params
