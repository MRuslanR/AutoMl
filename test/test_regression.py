from automl.models import BayesianOptimizationHPO, EvolutionaryStrategyHPO
from automl.comparison import compare_hpo_methods


from sklearn.ensemble import RandomForestRegressor
config_regression = {
    'task_type': 'regression',
    'model': {
        'class': RandomForestRegressor,
        'fixed_params': {'random_state': 50},
        'param_space': [
            {'name': 'n_estimators', 'type': 'integer', 'low': 50, 'high': 200},
            {'name': 'max_depth', 'type': 'integer', 'low': 3, 'high': 20},
            {'name': 'min_samples_split', 'type': 'integer', 'low': 2, 'high': 10},
            {'name': 'min_samples_leaf', 'type': 'integer', 'low': 1, 'high': 4},
            {'name': 'min_impurity_decrease', 'type': 'float', 'low': 0.0, 'high': 1.0}
        ]
    },
    'scoring': 'neg_mean_squared_error',
    'hpo_methods': {
        'Bayesian': {
            'hpo_class': BayesianOptimizationHPO,
            'hpo_params': {
                'verbose': 1,
                'random_state': 50,
            }
        },
        'Evolutionary': {
            'hpo_class': EvolutionaryStrategyHPO,
            'hpo_params': {
                'verbose': 1,
                'random_state': 50,
            }
        }
    }
}

from sklearn.datasets import load_diabetes
dataset = load_diabetes()
X, y = dataset.data, dataset.target
compare_hpo_methods(config_regression,X,y)