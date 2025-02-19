import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from warnings import catch_warnings, simplefilter

class BayesianOptimizationHPO:
    def __init__(self, f, param_space, verbose=1, random_state=None,
                 init_points=10, n_iter=50, acq='ucb', kappa=2.576, xi=0.0):

        """
        f: Objective function to minimize/maximize. Must accept parameters from param_space as kwargs
        param_space: List of dictionaries defining parameters (name, type, low/high). Categoricals not supported
        verbose: Verbosity level (0 = silent, 1 = basic, 2 = detailed). Default: 2
        random_state: Seed for random number generator. Default: None
        init_points: Number of initial random evaluations. Default: 5
        n_iter: Number of Bayesian optimization iterations. Default: 25
        acq: Acquisition function type ('ucb' = upper confidence bound, 'ei' = expected improvement). Default: 'ucb'
        kappa: Exploration-exploitation tradeoff parameter for UCB. Default: 2.576
        xi: Exploration threshold parameter for EI. Default: 0.0
        """
        self.f = f
        self.param_space = param_space
        self.verbose = verbose
        self.random_state = random_state
        self.init_points = init_points
        self.n_iter = n_iter
        self.acq = acq
        self.kappa = kappa
        self.xi = xi

        self._space = []
        self._values = []
        self.rng = np.random.RandomState(random_state)
        self.pbounds = {}

        for param in param_space:
            if 'low' not in param or 'high' not in param:
                raise ValueError(f"Parameter {param['name']} missing 'low' or 'high'")
            if param['low'] >= param['high']:
                raise ValueError(f"Invalid range for {param['name']}")

            if param['type'] in ['integer', 'float']:
                self.pbounds[param['name']] = (param['low'], param['high'])
            elif param['type'] == 'categorical':
                raise ValueError("Categorical params not supported in Bayesian optimization")

        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25,
            random_state=random_state
        )
        self._prime_subscriptions()

    def optimize(self):
        self.init(self.init_points)

        for i in range(self.n_iter):
            with catch_warnings():
                simplefilter("ignore")
                self.probe(self._next())

            if self.verbose >= 1:
                print(f"Iteration {i + 1}/{self.n_iter} | Best: {self.max['target']:.4f}")

        # Обработка лучших параметров
        best_params = {}
        for param in self.param_space:
            name = param['name']
            value = self.max['params'][name]
            if param['type'] == 'integer':
                best_params[name] = int(round(value))
            else:
                best_params[name] = value

        history = self._values.copy()

        if self.verbose > 0:
            print("═" * 50)
            print("Best params for Bayesian Optimization:")
            for k, v in best_params.items():
                print(f"▸ {k:20} : {v}")
            print(f"\nBest result: {self.max['target']:.4f}")
            print("═" * 50)

        return best_params, history

    def _next(self):
        self.gp.fit(self._space, self._values)
        best_acq = -np.inf
        best_x = None

        for _ in range(500):
            candidate = np.array([self.rng.uniform(low, high) for (low, high) in self._bounds])

            mu, sigma = self.gp.predict([candidate], return_std=True)

            if self.acq == 'ucb':
                acq_value = mu + self.kappa * sigma
            elif self.acq == 'ei':
                improvement = mu - self.max['target'] - self.xi
                z = improvement / sigma if sigma > 1e-12 else 0.0
                acq_value = (improvement * norm.cdf(z) + sigma * norm.pdf(z)) if sigma > 1e-12 else 0.0

            if acq_value > best_acq:
                best_acq = acq_value
                best_x = candidate

        return best_x

    def _prime_subscriptions(self):
        self._bounds = np.array([v for k, v in self.pbounds.items()])
        self.dim = len(self.pbounds)
        self.initialized = False

    def init(self, init_points):
        for _ in range(init_points):
            x = np.array([self.rng.uniform(low, high) for (low, high) in self._bounds])
            self.probe(x)

    def probe(self, x):
        params = dict(zip(self.pbounds.keys(), x))
        target = self.f(**params)

        self._space.append(x)
        self._values.append(target)

        if self.verbose > 1:
            print(f"Probe: {params} → Score: {target:.4f}")

    @property
    def max(self):
        idx = np.argmax(self._values)
        return {
            'params': dict(zip(self.pbounds.keys(), self._space[idx])),
            'target': self._values[idx]
        }