from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from constant import SEED

class Config:
    def __init__(
        self,
        bootstrap=True,
        max_depth=None,
        max_features="auto",
        max_leaf_nodes=None,
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

class Model:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        self.model = RandomForestClassifier(
            bootstrap=self.config.bootstrap,
            max_depth=self.config.max_depth,
            max_features=self.config.max_features,
            max_leaf_nodes=self.config.max_leaf_nodes,
            n_estimators=self.config.n_estimators,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=SEED,
        )
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


def optimize(X, y, n_trials=100):
    def objective(trial):
        params = {
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "max_depth": trial.suggest_int("max_depth", 1, 100),
            "max_features": trial.suggest_float("max_features", 0, 1.0),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 1, 1000),
            "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
            "min_samples_leaf": trial.suggest_int('min_samples_leaf',1,10),
            "random_state": SEED,
        }
        model = RandomForestClassifier(**params)
        score = cross_val_score(
            model, X, y, cv=5, scoring=make_scorer(accuracy_score)
        ).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials)
    return study.best_params
