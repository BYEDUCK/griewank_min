from random import randint
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import numpy as np
from math import pow, sqrt, cos
from scipy.optimize import minimize

import functions.operators as op
import functions.selections as sel
from functions import fitness_fnc_griewan


class GeneticAlgorithm:

    albumNumber = 277125
    numberOfIterations = 0
    populationCardinality = 0
    startingMutationProbability = 0
    recombinationProbability = 0
    numberOfItems = 0
    iters = 0
    running = True

    def __init__(self, number_of_iterations, population_cardinality, population,
                 recombination_probability, write_data_to_file=False, app=None,
                 mutation_strategy="mutate_real", recombination_strategy="avg_recombination",
                 selection="roulette_selection", type_of_opt="max", mut_adapt=True, intel_recomb=True):
        self.vars = []
        self.stds = []
        self.avgs = []
        self.mins = []
        self.maxes = []
        self.generations = []
        self.type_of_opt = type_of_opt
        self.mutation_strategy = getattr(op, mutation_strategy)
        self.recombination_strategy = getattr(op, recombination_strategy)
        self.selection = getattr(sel, selection)
        self.mut_adapt = mut_adapt
        self.intel_recomb = intel_recomb
        self.app = app
        self.numberOfIterations = number_of_iterations
        self.populationCardinality = population_cardinality
        self.recombinationProbability = recombination_probability
        self.writeDataToFile = write_data_to_file
        self.thePopulation = population
        if self.writeDataToFile:
            self.resultFile = open("result.txt", "w")
            if self.type_of_opt == "max":
                self.resultFile.write("Maximization\n")
            else:
                self.resultFile.write("Minimization\n")
            self.resultFile.write("Recombination probability: " + repr(self.recombinationProbability) + "%\n" +
                                  "Population cardinality: " + repr(self.populationCardinality) + "\n" +
                                  "Max number of iterations: " + repr(self.numberOfIterations) + "\n\n")

    @staticmethod
    def __neg_fnc(x):
        return -(-(pow(x[1:] - 1.5, 2) + pow(x[:-1] - 1.5, 2)) / (40 * 1.5) -
                 (cos(2 * x[1:] - 1.5) * cos((2 * x[:-1] - 1.5) / sqrt(2))) + 1)

    @staticmethod
    def __fnc(x):
        return -(pow(x[1:] - 1.5, 2) + pow(x[:-1] - 1.5, 2)) / (40 * 1.5) - (
                    cos(2 * x[1:] - 1.5) * cos((2 * x[:-1] - 1.5) / sqrt(2))) + 1

    def start(self):
        if self.app is not None:
            animation = anm.FuncAnimation(self.app.figure, self.animate_plot, interval=1000)

        if self.check_if_correct():
            for iteration in range(self.numberOfIterations):
                if self.running:
                    self.iters += 1
                    mutationProbability = 25
                    mutationSD = 1
                    progress = iteration/self.numberOfIterations

                    # mutation
                    if self.mut_adapt:
                        if progress <= 0.33:
                            mutationProbability = 100
                            mutationSD = 3
                        else:
                            if progress <= 0.67:
                                mutationProbability = 50
                                mutationSD = 1.5
                            else:
                                if progress <= 1:
                                    mutationProbability = 35
                                    mutationSD = 1
                    self.thePopulation = self.mutation_strategy(self.thePopulation, mutationProbability, sd=mutationSD)

                    # recombination
                    decision = randint(1, 100)
                    if decision <= self.recombinationProbability:
                        crossing = False
                        one = 0
                        two = 0
                        for n in range(self.populationCardinality):
                            one = randint(0, self.populationCardinality - 1)
                            two = randint(0, self.populationCardinality - 1)
                            if one != two:
                                if self.intel_recomb:
                                    if self.type_of_opt == "min":
                                        f = self.__fnc
                                    else:
                                        f = self.__neg_fnc
                                    res1 = minimize(f, self.thePopulation[one, :], method='nelder-mead',
                                                    options={'xtol': 1e-8, 'disp': False})
                                    res2 = minimize(f, self.thePopulation[two, :], method='nelder-mead',
                                                    options={'xtol': 1e-8, 'disp': False})
                                    if res1.x[0] == res2.x[0] and res1.x[-1] == res2.x[-1]:
                                        crossing = True
                                        break
                                else:
                                    crossing = True
                                    break

                        if crossing:
                            self.thePopulation[one, :], self.thePopulation[two, :] = self.recombination_strategy(
                                self.thePopulation[one, :],
                                self.thePopulation[two, :])

                    # selection
                    after_selection = self.selection(self.thePopulation, self.populationCardinality, type_of_opt=self.type_of_opt)
                    self.thePopulation = self.thePopulation[after_selection, :]

                    fitness = fitness_fnc_griewan(self.thePopulation)
                    temp_var = np.var(fitness)
                    temp_std = np.std(fitness)
                    temp_avg = np.average(fitness)
                    temp_min = np.min(fitness)
                    temp_max = np.max(fitness)
                    if self.app is not None:
                        self.app.values_changed(temp_std, temp_var, temp_avg, temp_min, temp_max)
                    self.vars.append(temp_var)
                    self.stds.append(temp_std)
                    self.avgs.append(temp_avg)
                    self.mins.append(temp_min)
                    self.maxes.append(temp_max)
                    self.generations.append(iteration + 1)

                    if self.writeDataToFile:
                        self.resultFile.write("Generation [" + repr(iteration + 1) + "]:\n" +
                                              "\nVariance: " + repr(self.vars[iteration]) +
                                              "\nStandard deviation: " + repr(self.stds[iteration]) +
                                              "\nMax: " + repr(self.maxes[iteration]) +
                                              "\nMin: " + repr(self.mins[iteration]) +
                                              "\nAverage: " + repr(self.avgs[iteration]) + "\n\n")

            final = fitness_fnc_griewan(self.thePopulation)

            if self.type_of_opt == "max":
                the_winner = self.thePopulation[np.argmax(final), :]
                score = np.max(final)
            else:
                the_winner = self.thePopulation[np.argmin(final), :]
                score = np.min(final)

            print("\n\n")
            print("And the winner is... :\n")
            print(the_winner)
            print("\nWith the score of: ")
            print(score)

            if self.writeDataToFile:
                self.resultFile.write("\nWinner: " + repr(the_winner) + "\n")
                self.resultFile.write("\nWinner's result: " + repr(score) + "\n")
                self.resultFile.close()

            generation = [i for i in range(1, self.iters + 1)]
            plt.figure(1)
            plt.plot(generation, self.vars, 'b-', generation, self.avgs, 'y-',
                     generation, self.mins, 'g-', generation, self.maxes, 'r-')
            plt.grid(True)
            plt.xlabel("Generation")
            plt.legend(["Variance", "Average", "Min", "Max"])
            plt.title("Iterations=%d; Populations cardinality=%d;\nRecombination probability=%d%%"
                      % (self.iters, self.populationCardinality, self.recombinationProbability))
            plt.show()
        else:
            print("Data out of bounds or no data!")

    def check_if_correct(self):
        if 0 < self.numberOfIterations < 5000 and 0 < self.populationCardinality < 2000 \
            and 1000 >= self.startingMutationProbability >= 0 \
                and 0 <= self.recombinationProbability <= 100:
            return True
        else:
            return False

    def animate_plot(self, i):
        self.app.graph.clear()
        self.app.graph.plot(self.generations, self.stds, 'b-', self.generations, self.avgs, 'y-',
                            self.generations, self.mins, 'g-', self.generations, self.maxes, 'r-')
