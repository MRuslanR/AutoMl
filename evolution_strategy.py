from dataclasses import replace

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone


class EvolutionaryStrategyHPO:
    def __init__(self, estimator, param_space, scoring,  cv=3,
                 population_size=50, generations=20,
                 mutation_rate=0.1, elite_ratio=0.2):
        """
        Инициализация эволюционной стратегии

        :param estimator: Базовый estimator sklearn
        :param param_space: Пространство параметров для оптимизации
        :param scoring: Функция для подсчета скора
        :param cv: Стратегия кросс-валидации
        :param population_size: Размер популяции
        :param generations: Количество поколений
        :param mutation_rate: Вероятность мутации
        :param elite_ratio: Доля элитных особей
        """
        self.estimator = estimator
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.history = []
        self.fitness_cache = {}

        for param in param_space:
            if 'low' not in param:
                if param['type'] == 'integer':
                    param['low'] = 2
                elif param['type'] == 'float':
                    param['low'] = 0.0
            if 'high' not in param:
                if param['type'] == 'integer':
                    param['high'] = 100
                elif param['type'] == 'float':
                    param['high'] = 1.0

    def _initialize_individual(self):
        individual = {}
        for param in self.param_space:
            name = param['name']
            if param['type'] == 'float':
                individual[name] = np.random.uniform(param['low'], param['high'])
            elif param['type'] == 'integer':
                individual[name] = np.random.randint(param['low'], param['high'] + 1)
            elif param['type'] == 'categorical':
                individual[name] = np.random.choice(param['categories'])
        return individual

    def _mutate(self, individual):
        mutated = individual.copy()
        for param in self.param_space:
            name = param['name']
            if np.random.rand() < self.mutation_rate:
                if param['type'] == 'float':
                    new_val = individual[name] + np.random.uniform(-1, 1)
                    new_val = np.clip(new_val, param['low'], param['high'])
                    mutated[name] = new_val
                elif param['type'] == 'integer':
                    new_val = individual[name] + np.random.randint(-10, 11)
                    new_val = np.clip(new_val, param['low'], param['high'])
                    mutated[name] = new_val
                elif param['type'] == 'categorical':
                    mutated[name] = np.random.choice(param['categories'])
        return mutated

    def _crossover(self, parent1, parent2):
        child = parent1.copy()
        crossover_params = np.random.choice(
            list(self.param_space),
            size=np.random.randint(1, len(self.param_space) + 1),
            replace=False
        )

        for param in crossover_params:
            child[param['name']] = parent2[param['name']]
        return child

    def _evaluate(self, individual, X, y):
        key = tuple(sorted(individual.items()))
        if key in self.fitness_cache:
            return self.fitness_cache[key]

        model = clone(self.estimator)
        model.set_params(**individual)

        seed = abs(hash(key)) % (2**32)
        model.set_params(random_state=seed)

        scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
        mean_score = np.mean(scores)
        self.fitness_cache[key] = mean_score
        return mean_score

    def optimize(self, X, y):
        population = [self._initialize_individual() for _ in range(self.population_size)]
        elite_size = max(2, int(self.population_size * self.elite_ratio))

        for gen in range(self.generations):
            # Оценка приспособленности
            fitness = [self._evaluate(individual, X, y) for individual in population]

            # Отбор элитных особей
            elite_indices = np.argsort(fitness)[-elite_size:]
            elites = [population[i] for i in elite_indices]

            # Генерация потомков
            offspring = []
            while len(offspring) < self.population_size - elite_size:
                parents = np.random.choice(elites, 2, replace=False)
                child = self._crossover(parents[0], parents[1])
                child = self._mutate(child)
                offspring.append(child)

            # Формирование новой популяции
            population = elites + offspring

            # Логирование
            best_fitness = np.max(fitness)
            self.history.append(best_fitness)
            print(f"Generation {gen}: Best Fitness = {best_fitness:.4f}")

        # Возврат лучшей конфигурации
        best_idx = np.argmax([self._evaluate(ind, X, y) for ind in population])
        return population[best_idx], self.history


# Пример использования
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression

    # Создание синтетических данных
    X, y = make_regression(n_samples=500, n_features=8, noise=0.1)

    # Определение пространства параметров
    param_space = [
        {'name': 'n_estimators', 'type': 'integer'},
        {'name': 'max_depth', 'type': 'integer'},
        {'name': 'min_samples_split', 'type': 'integer'},
        {'name': 'criterion', 'type': 'categorical', 'categories': ['squared_error', 'absolute_error']}
    ]

    # Инициализация ES
    es = EvolutionaryStrategyHPO(
        estimator=RandomForestRegressor(),
        param_space=param_space,
        scoring="neg_mean_squared_error",
        population_size=100,
        generations=5,
        mutation_rate=0.2,
        elite_ratio = 0.2,
    )

    # Запуск оптимизации
    best_params, history = es.optimize(X, y)
    print("\nBest parameters found:")
    print(best_params)



