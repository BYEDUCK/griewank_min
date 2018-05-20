from random import randint
from numpy.random import normal


def mutate_binary(population, probability):
    nof_individuals = population.shape[0]
    nof_bits = population.shape[1]
    for i in range(nof_individuals):
        for j in range(nof_bits):
            decision = randint(1, 1000)
            if decision <= probability:
                population[i, j] = negate(population[i, j])
    return population


def negate(boolean_as_int):
    if boolean_as_int == 1:
        return 0
    else:
        if boolean_as_int == 0:
            return 1
    return -1


def classic_cross_over(one, two, nof_bits):
    cross_point = randint(1, nof_bits - 1)
    second_part = [i for i in range(cross_point, nof_bits)]
    temp = two[second_part]
    two[second_part] = one[second_part]
    one[second_part] = temp
    return one, two


def avg_recombination(one, two):
    a = normal(0, 1)
    temp = one
    one = a * one + (1 - a) * two
    two = a * two + (1 - a) * temp
    return one, two


def mutate_real(population, probability, sd=1):
    nof_individuals = population.shape[0]
    for i in range(nof_individuals):
        decision = randint(1, 1000)
        if decision <= probability:
            population[i, :] += normal(0, sd)
    return population
