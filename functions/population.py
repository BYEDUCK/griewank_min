import numpy as np
from math import sqrt, pow, cos


def draw_backpack_items(nof_items, seed_number):
    np.random.seed(seed_number)
    w = np.round_(0.1 + 0.9 * np.random.random((nof_items, 1)), 1)
    p = np.round_(1 + 99 * np.random.random((nof_items, 1)))
    f = w * p
    return np.concatenate((w, p, f), axis=1)


def count_fitness_fnc_and_weights(items, population, max_weight):
    nof_items = items.shape[0]
    nof_individuals = population.shape[0]
    if nof_items != population.shape[1]:
        return 0
    result = np.zeros((2, nof_individuals))
    for j in range(nof_individuals):
        result[1, j] = np.sum(items[np.where(np.transpose(population[j, :]) == 1), 0])
        if result[1, j] <= max_weight:
            result[0, j] = np.sum(items[np.where(np.transpose(population[j, :]) == 1), 2])
        else:
            result[0, j] = 0.1
    return result


def fitness_fnc(items, population):
    nof_items = items.shape[0]
    nof_individuals = population.shape[0]
    if nof_items != population.shape[1]:
        return 0
    result = np.zeros(nof_individuals)
    for j in range(nof_individuals):
        result[j] = np.sum(items[np.where(np.transpose(population[j, :]) == 1), 2])
    return result


def fitness_fnc_griewan(population, c=1.5, type_of_opt="max"):
    nof_individuals = population.shape[0]
    result = np.zeros(nof_individuals)
    for j in range(nof_individuals):
        x1 = population[j, 0]
        x2 = population[j, 1]
        if -10 <= x1 <= 10 and -10 <= x2 <= 10:
            result[j] = -(pow(x1 - c, 2) + pow(x2 - c, 2)) / (40 * c) - (cos(2 * x1 - c) * cos((2 * x2 - c) / sqrt(2))) + 1
        else:
            if type_of_opt == "min":
                result[j] = 10  # kara
            else:
                result[j] = -10
    return result


def chose_winner(final_score, final_weights, max_weight):
    maximum = 0
    it = 0
    index = -1
    for i, j in zip(final_score, final_weights):
        if i > maximum and j <= max_weight:
            maximum = i
            index = it
        it += 1
    return index
