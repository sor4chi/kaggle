import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from constant import SEED
from utils import get_mapping

class Config:
    def __init__(self, n_estimators=100, max_depth=-1, lambda_l1=0.0, lambda_l2=0.0, learning_rate=0.1, num_leaves=31, feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, min_child_samples=20):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.min_child_samples = min_child_samples

class Model:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        self.model = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            lambda_l1=self.config.lambda_l1,
            lambda_l2=self.config.lambda_l2,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            feature_fraction=self.config.feature_fraction,
            bagging_fraction=self.config.bagging_fraction,
            bagging_freq=self.config.bagging_freq,
            min_child_samples=self.config.min_child_samples,
            force_col_wise=True,
            random_state=SEED
        )
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration_)



def optimize(X, y, n_trials=100):
    def objective(trial):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        num_class = len(get_mapping()['NObeyesdad'])
        print('num_class:', num_class)
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': num_class,
            'n_estimators': trial.suggest_int('n_estimators', 50, 125),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 0.1, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

            'force_col_wise':True,
            'random_state': SEED
        }
        model = lgb.LGBMClassifier(**params)
        score = cross_val_score(model, X, y, scoring=make_scorer(accuracy_score), cv=5).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials)
    model = lgb.LGBMClassifier(**study.best_params)
    model.fit(X, y)
    return study.best_params

