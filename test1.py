#!/usr/bin/python3.6
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random
'''
    Reproduce an example similar to the previous example with two
    unidimensional distributions, with equal and known variance where the mean of each
    distribtuion and π 1 , π 2 are unknown. Obtain the corresponding plots.
'''
from EM import modelEM
import numpy as np
if __name__ == '__main__':
    # Initialize plotter
    fig, ax1 = plt.subplots()
    # Create initial model in order to generate data
    genModel = modelEM()
    # Generate data
    data = genModel.generateData(500)
    # Plot histogram
    ax1.hist(data, bins=30, range=(-15, 15), ec='gray', alpha=0.75)

    # Plot density distribution of generated model
    ax2 = ax1.twinx()
    t1 = np.linspace(-15,15,200)
    ax2.plot(t1, genModel.getDensityPoint(t1), 'r--', lw=2, label='Gen model', alpha=0.9)

    # Create new EM model with random prior probabilities
    p = random.uniform(0.1, 0.9)
    model = modelEM(probabilities=[p, 1-p])
    # Plot initial distribution
    ax2.plot(t1, model.getDensityPoint(t1), color='gray', ls='--', lw=2, label=r'$\theta_{%i}$'%(0))
    # Set number of reestimations (iterations)
    numIter = 10
    # x data to plot
    t1 = np.linspace(-15,15,200)
    for i in range(numIter):
        # Reestimate priorities and mean
        model.reestimate(data, r_priors=True, r_means=True, r_vars=False)
        # Print current model parameters
        print("Iteration %i"%(i))
        for ind, (p, g) in enumerate(zip(model.probabilities, model.gaussParams)):
            print('g%i: prob:%.2f mean:%.2f var:%.2f'%(ind, p, g[0], g[1]))
        if i < numIter-1:
            # Plot current distribution
            ax2.plot(t1, model.getDensityPoint(t1), '--', label=r'$\theta_{%i}$'%(i+1))
    # Plot final distribution
    ax2.plot(t1, model.getDensityPoint(t1), '--', lw=2, color='k', label=r'$\theta_{%i}$'%(i+1))

    # Show legend
    plt.legend()
    plt.show()
