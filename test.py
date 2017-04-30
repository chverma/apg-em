#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random
from EM import ModelEM
import numpy as np
import sys
import argparse

def main(ndata=50, numIter=10, task='task1'):
    '''
    Reproduce an example similar to the previous example with two
    unidimensional distributions.
    -TASK1 - with equal and known VARIANCE where the MEANS and π1, π2 are unknown.
    -TASK2 - with equal and known MEAN where the VARIANCES and π1, π2 are unknown.
    '''
    # Initialize plotter
    fig, ax1 = plt.subplots()
    # Create initial model in order to generate data
    genModel = ModelEM()
    # Generate data
    data = genModel.generateData(ndata)
    # Plot histogram
    ax1.hist(data, bins=30, range=(-15, 15), ec='gray', alpha=0.75)

    # Plot density distribution of generated model
    ax2 = ax1.twinx()
    t1 = np.linspace(-15,15,200)
    ax2.plot(t1, genModel.getDensity(t1), 'r--', lw=2, label='Gen model', alpha=0.9)

    # Create new EM model with random prior probabilities
    p = random.uniform(0.1, 0.9)
    model = ModelEM(probabilities=[p, 1-p])
    # Plot initial distribution
    ax2.plot(t1, model.getDensity(t1), color='gray', ls='--', lw=2, label=r'$\theta_{%i}$'%(0))

    # x data to plot
    t1 = np.linspace(-15,15,200)
    resMeans = False
    resVariance = False
    # If task1, reestimate priors and MEANS
    if task == 'task1':
        resMeans = True
    else:
        resVariance  =True
    # Set number of reestimations (iterations)
    for i in range(numIter):
        # Run EM and reestimate parameteres depend on task
        model.run(data, r_means=resMeans, r_vars=resVariance)
        # Print current model parameters
        print("Iteration %i"%(i))
        for ind, (p, g) in enumerate(zip(model.getProbabilities(), model.getGaussianParameters())):
            print('g%i: prob:%.2f mean:%.2f var:%.2f'%(ind, p, g[0], g[1]))
        if i < numIter-1:
            # Plot current distribution
            ax2.plot(t1, model.getDensity(t1), '--', label=r'$\theta_{%i}$'%(i+1))
    # Plot final distribution
    ax2.plot(t1, model.getDensity(t1), '--', lw=2, color='k', label=r'$\theta_{%i}$'%(i+1))

    # Show legend
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=main.__doc__)
    parser.add_argument('iterations', metavar='I', type=int, nargs=1,
                       help='number of iterations of EM algorithm')
    parser.add_argument('ndata', metavar='N', type=int, nargs=1,
                       help='number of data to be generated')
    parser.add_argument('task',
                       choices=['task1', 'task2'], help='select task to be reproduced')

    args = parser.parse_args()
    main(ndata=args.ndata[0], numIter=args.iterations[0], task=args.task[0])
