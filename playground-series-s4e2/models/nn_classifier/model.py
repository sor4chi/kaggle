from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from constant import SEED

class Config:
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change

class Model:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def  fit(self, X, y):
        # NNは決定木系と違って、データのスケールに影響を受けるため、標準化が必要
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=SEED
        )
        print('hidden_layer_sizes:', self.config.hidden_layer_sizes)
        self.model = MLPClassifier(
            # hidden_layer_sizes=self.config.hidden_layer_sizes,
            activation=self.config.activation,
            solver=self.config.solver,
            alpha=self.config.alpha,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            learning_rate_init=self.config.learning_rate_init,
            max_iter=self.config.max_iter,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state,
            tol=self.config.tol,
            verbose=self.config.verbose,
            warm_start=self.config.warm_start,
            momentum=self.config.momentum,
            nesterovs_momentum=self.config.nesterovs_momentum,
            early_stopping=self.config.early_stopping,
            validation_fraction=self.config.validation_fraction,
            beta_1=self.config.beta_1,
            beta_2=self.config.beta_2,
            epsilon=self.config.epsilon,
            n_iter_no_change=self.config.n_iter_no_change,
        )
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


def optimize(X, y, n_trials=100):
    X_scaled = StandardScaler().fit_transform(X)

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 4)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_{i}', 1, 100))
        params = {
            "hidden_layer_sizes": tuple(layers),
            "activation": trial.suggest_categorical("activation", ['identity', 'logistic', 'tanh', 'relu']),
            "solver": trial.suggest_categorical("solver", ['lbfgs', 'sgd', 'adam']),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.01),
            "batch_size": trial.suggest_categorical("batch_size", ['auto', 16, 32, 64]),
            "learning_rate": trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.0001, 0.1),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "shuffle": trial.suggest_categorical("shuffle", [True, False]),
            "random_state": SEED,
        }
        model = MLPClassifier(**params)
        score = cross_val_score(
            model, X_scaled, y, cv=5, scoring=make_scorer(accuracy_score)
        ).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials)
    return study.best_params
