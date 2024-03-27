import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from constant import SEED
from utils import get_mapping

class Config:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3, gamma=0, min_child_weight=1, subsample=1, colsample_bytree=1, reg_alpha=0, reg_lambda=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

class Model:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=SEED
        )
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    def predict(self, X):
        return self.model.predict(X)

def optimize(X, y, n_trials=100):
    def objective(trial):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        num_class = len(get_mapping()['NObeyesdad'])
        print('num_class:', num_class)
        params = {
            'objective': 'multi:softmax',
            'num_class': num_class
        }
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 125)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 7)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
        params['gamma'] = trial.suggest_float('gamma', 0, 1)
        params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1)
        params['reg_alpha'] = trial.suggest_float('reg_alpha', 0, 10)
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 0, 10)
        model = xgb.XGBClassifier(**params, random_state=SEED)
        return cross_val_score(model, X, y, cv=5, scoring=make_scorer(accuracy_score)).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials)
    return study.best_params
