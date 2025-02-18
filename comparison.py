import time
import numpy as np
from sklearn.model_selection import cross_val_score
from warnings import catch_warnings, simplefilter

from algorithms.evolution_strategy import EvolutionaryStrategyHPO
from algorithms.bayesian_optimization import BayesianOptimizationHPO

def compare_hpo_methods(config):
    task_type = config['task_type']
    model_config = config['model']
    datasets_config = config['datasets']
    scoring = config['scoring']
    hpo_methods_config = config['hpo_methods']

    for dataset_name, dataset_info in datasets_config.items():
        # Загрузка и подготовка данных
        data = dataset_info['loader']()
        X, y = dataset_info['preprocess'](data)

        # Определение целевой функции
        def objective_function(**params):
            model_class = model_config['class']
            model_params = model_config['fixed_params'].copy()

            for param in model_config['param_space']:
                name = param['name']
                value = params[name]
                if param['type'] == 'integer':
                    model_params[name] = int(round(value))
                elif param['type'] == 'float':
                    model_params[name] = value
                elif param['type'] == 'categorical':
                    model_params[name] = value

            model = model_class(**model_params)
            with catch_warnings():
                simplefilter("ignore")
                scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
            return np.mean(scores)

        # Сравнение методов HPO
        results = {}
        for method_name, method_config in hpo_methods_config.items():
            HPOClass = method_config['hpo_class']
            hpo_params = method_config.get('hpo_params', {})

            # Инициализация оптимизатора
            hpo = HPOClass(
                f=objective_function,
                param_space=model_config['param_space'],
                **hpo_params
            )

            # Запуск оптимизации
            start_time = time.time()
            best_params, history = hpo.optimize()
            best_score = max(history)
            time_taken = time.time() - start_time

            results[method_name] = {
                'best_score': best_score,
                'time': time_taken,
                'history': history
            }

            # Вывод результатов
            print(f"\n=== {method_name} ({task_type}, {dataset_name}) ===")
            print(f"Best Score: {best_score:.4f}, Time: {time_taken:.2f}s")

        # Сводка по датасету
        print(f"\n=== Results for {dataset_name} ===")
        for method, res in results.items():
            print(f"{method}: Best Score = {res['best_score']:.4f}, Time = {res['time']:.2f}s")
        print("=" * 50 + "\n")

# Конфигурация и вызов
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier

config_classification = {
    'task_type': 'classification',
    'model': {
        'class': RandomForestClassifier,
        'fixed_params': {'random_state': 42},
        'param_space': [
            {'name': 'n_estimators', 'type': 'integer', 'low': 50, 'high': 200},
            {'name': 'max_depth', 'type': 'integer', 'low': 3, 'high': 20},
            {'name': 'min_samples_split', 'type': 'integer', 'low': 2, 'high': 10},
            {'name': 'min_samples_leaf', 'type': 'integer', 'low': 1, 'high': 4}
        ]
    },
    'datasets': {
        'iris': {
            'loader': load_iris,
            'preprocess': lambda data: (data.data, data.target)
        },
        'wine': {
            'loader': load_wine,
            'preprocess': lambda data: (data.data, data.target)
        }
    },
    'scoring': 'accuracy',
    'hpo_methods': {
        'Bayesian': {
            'hpo_class': BayesianOptimizationHPO,
            'hpo_params': {
                'verbose': 0,
                'random_state': 42,
                'init_points': 10,
                'n_iter': 50
            }
        },
        'Evolutionary': {
            'hpo_class': EvolutionaryStrategyHPO,
            'hpo_params': {
                'population_size': 20,
                'generations': 10,
                'mutation_rate': 0.1,
                'verbose': 0,
                'random_state': 42
            }
        }
    }
}

#compare_hpo_methods(config_classification)


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

config_regression = {
    'task_type': 'regression',
    'model': {
        'class': RandomForestRegressor,
        'fixed_params': {'random_state': 42},
        'param_space': [
            {'name': 'n_estimators', 'type': 'integer', 'low': 50, 'high': 200},
            {'name': 'max_depth', 'type': 'integer', 'low': 3, 'high': 20},
            {'name': 'min_samples_split', 'type': 'integer', 'low': 2, 'high': 10},
            {'name': 'min_samples_leaf', 'type': 'integer', 'low': 1, 'high': 4}
        ]
    },
    'datasets': {
        'diabetes': {
            'loader': load_diabetes,
            'preprocess': lambda data: (data.data, data.target)
        },

    },
    'scoring': 'neg_mean_squared_error',
    'hpo_methods': {
        'Bayesian': {
            'hpo_class': BayesianOptimizationHPO,
            'hpo_params': {
                'verbose': 0,
                'random_state': 42,
                'init_points': 10,
                'n_iter': 50
            }
        },
        'Evolutionary': {
            'hpo_class': EvolutionaryStrategyHPO,
            'hpo_params': {
                'population_size': 20,
                'generations': 10,
                'mutation_rate': 0.1,
                'verbose': 0,
                'random_state': 42
            }
        }
    }
}

compare_hpo_methods(config_regression)