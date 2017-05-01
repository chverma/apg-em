#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random
from EM import ModelEM
import numpy as np
import sys
import argparse

def main(ndata=50, numIter=10, task='task1', g1=None, g2=None):
    '''
    Reproduce an example similar to the previous example with two
    unidimensional distributions.
    -TASK1 - with equal and known VARIANCE where the MEANS and π1, π2 are unknown.
    -TASK2 - with equal and known MEAN where the VARIANCES and π1, π2 are unknown.
    -BASE - with equal VARIANCES and and known MEAN where π1, π2 are unknown.
    '''
    # Initialize plotter
    fig, ax1 = plt.subplots()
    # Create initial model in order to generate data
    if g1 is not None and g2 is not None:
        genModel = ModelEM(gaussParams=[[g1[1], g1[2]], [g2[1], g2[2]]], probabilities=[g1[0], 1-g1[0]])
    else:
        genModel = ModelEM()
    #Print generate model parameters
    print("Generated model")
    genModel.getStatus()
    # Generate data
    data = genModel.generateData(ndata)
    # Plot histogram
    ax1.hist(data, bins=30, range=(-15, 15), ec='gray', alpha=0.75, align='left')

    # Plot density distribution of generated model
    ax2 = ax1.twinx()
    t1 = np.linspace(-15,15,200)
    ax2.plot(t1, genModel.getDensity(t1), 'r-', lw=1, label='Gen model', alpha=0.9)

    # Create new EM model with random prior probabilities
    p = random.uniform(0.1, 0.9)
    # If task1, reestimate MEANS
    if task == 'task1':
        resMeans = True
        resVariance = False
        m1 = random.randint(-10, 10)
        m2 = random.randint(-10, 10)
        model = ModelEM(probabilities=[p, 1-p], gaussParams=[[m1, g1[2]], [m2, g2[2]]])
    # If task2, reestimate VARIANCE
    elif task == 'task2':
        resVariance = True
        resMeans = False
        var1 = random.uniform(0.1, 6)
        var2 = random.uniform(0.1, 6)
        model = ModelEM(probabilities=[p, 1-p], gaussParams=[[g1[1], var1], [g2[1], var2]])
    # If base, reestimate only probabilities
    else:
        resMeans = False
        resVariance = False
        model = ModelEM(probabilities=[p, 1-p], gaussParams=[[g1[1], g1[2]], [g2[1], g2[2]]])

    #Print intial model parameters
    print("Initial model")
    model.getStatus()
    # Plot initial distribution
    ax2.plot(t1, model.getDensity(t1), color='gray', ls='--', lw=2, label=r'$\theta_{%i}$'%(0))

    # x data to plot
    t1 = np.linspace(-15,15,200)

    # Set number of reestimations (iterations)
    for i in range(numIter):
        # Run EM and reestimate parameteres depend on task
        model.run(data, r_means=resMeans, r_vars=resVariance)
        # Print current model parameters
        print("Iteration %i"%(i))
        model.getStatus()
        if i < numIter-1:
            # Plot current distribution
            ax2.plot(t1, model.getDensity(t1), '--', label=r'$\theta_{%i}$'%(i+1))
    # Plot final distribution
    ax2.plot(t1, model.getDensity(t1), '--', lw=2, color='k', label=r'$\theta_{%i}$'%(i+1))

    # Show legend
    plt.legend()
    plt.show()

class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, np.array(values))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=main.__doc__)
    parser.add_argument('iterations', metavar='I', type=int, nargs=1,
                       help='number of iterations of EM algorithm')
    parser.add_argument('ndata', metavar='N', type=int, nargs=1,
                       help='number of data to be generated')
    parser.add_argument('task',
                       choices=['task1', 'task2', 'base'], help='select task to be reproduced')

    parser.add_argument('-g1', action=Store_as_array, type=float, nargs='+', help='Specify base gaussian 1 and its π. Format: π1 mean1 var1')
    parser.add_argument('-g2', action=Store_as_array, type=float, nargs='+', help='Specify base gaussian 2 and its π. Format: π2 mean2 var2')
    args = parser.parse_args()
    main(ndata=args.ndata[0], numIter=args.iterations[0], task=args.task, g1=args.g1, g2=args.g2)
