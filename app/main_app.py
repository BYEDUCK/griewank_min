from math import cos, sqrt, pow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from app import GeneticAlgorithm
from mpl_toolkits.mplot3d import Axes3D


def griewan(x1, x2, c):
    return -(pow(x1 - c, 2) + pow(x2 - c, 2)) / (40 * c) - (cos(2 * x1 - c) * cos((2 * x2 - c) / sqrt(2))) + 1


def griewan_np(x1, x2, c):
    return -(np.power(x1 - c, 2) + np.power(x2 - c, 2)) / (40 * c) - (np.cos(2 * x1 - c) * np.cos((2 * x2 - c) / sqrt(2))) + 1


if __name__ == "__main__":

    # plot the function #
    # x = np.arange(-10, 10, 0.01)
    # y = x
    # f = griewan_np(x, y, 1.5)
    # X, Y = np.meshgrid(x, y)
    # Z = griewan_np(X, Y, 1.5)
    # print("Max = %f.05,  Min = %f.05" % (np.max(f), np.min(f)))
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # plt.show()

    print("\nMaximization/minimization of Griewank's function using evolution algorithm implemented byDUCK\n\n")
    nofIterations = int(input("Type in the max iteration number (1-5000): "))
    popCardinality = int(input("Type in the population cardinality (1-2000): "))
    crossProbability = int(input("Type in the crossing probability (0-100)%: "))
    opt_type = input("Type in the type of optimization you want to perform (max/min)(default: max): ")
    adapt_mut_d = input("Do you want to use adapting mutation (y/n)(default: yes): ")
    recomb_intel_d = input("Do you want to recombine only when parents are drawn by the same point (y/n)(default: yes): ")
    write_to_file_d = input("Do you want to write stats of every generation to 'result.txt' file? (y/n)(default: no): ")
    print("\n\n")

    adapt_mut = True
    cross_intel = True
    write_to_file = False

    if adapt_mut_d == "n":
        adapt_mut = False
    if recomb_intel_d == "n":
        cross_intel = False
    if write_to_file_d == "y":
        write_to_file = True
    if opt_type != "min" and opt_type != "max":
        opt_type = "max"

    input("Press Enter to continue...")

    population = np.random.random_sample((popCardinality, 2)) * 20 - 10

    myAlgorithm = GeneticAlgorithm(nofIterations, popCardinality, population, crossProbability,
                                   write_data_to_file=write_to_file, type_of_opt=opt_type, mut_adapt=adapt_mut, intel_recomb=cross_intel)

    print("Algorithm starting... Please wait... ('end' will be displayed when it ends)")
    myAlgorithm.start()

    print("end")
