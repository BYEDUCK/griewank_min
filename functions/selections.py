from random import randint

import numpy as np

from functions import fitness_fnc_griewan


def roulette_selection(population, population_cardinality, type_of_opt="max"):
    goal_function_values = fitness_fnc_griewan(population, type_of_opt=type_of_opt)
    if type_of_opt == "min":
        goal_function_values = np.zeros(population_cardinality) - goal_function_values
    goal_function_values += 3  # in order to avoid negative numbers
    goal_function_values = np.round((goal_function_values / np.sum(goal_function_values) * 1000), 0)
    chosen_ones = [i for i in range(population_cardinality)]

    for i in range(population_cardinality - 1):  # create a pseudo-distribuant
        goal_function_values[i + 1] += goal_function_values[i]

    for j in range(population_cardinality):
        decision = randint(1, goal_function_values[-1])
        for n in range(population_cardinality):
            if decision <= goal_function_values[n]:
                chosen_ones[j] = n
                break
    return chosen_ones
