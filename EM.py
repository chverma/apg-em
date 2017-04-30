#!/usr/bin/python3.6
import numpy as np
import random


class ModelEM:
    """
    ModelEM implements Expectation-Maximization algorithm with some features
    in order to reproduce some experiments on APG subject
    """
    _probabilities = None
    _gaussParams = None
    _nGauss = None

    def __init__(self, gaussParams=[[-6, 2], [2, 2]], probabilities=[0.4, 0.6]):
        self._probabilities = probabilities
        self._gaussParams = gaussParams
        self._nGauss = len(self._probabilities)

    def getNumOfGaussians(self):
        """
        Retrieve the number of gaussians used by the model

        Returns:
        out:int

        """
        return self._nGauss

    def getGaussianParameters(self):
        """
        Retrieve the gaussian parameters used by the model

        Returns:
        out:list

        """
        return self._gaussParams

    def getProbabilities(self):
        """
        Retrieve prior probabilties of gaussians

        Returns:
        out:list

        """
        return self._probabilities

    def generateData(self, n=50):
        """
        Retrieve generated data with gaussian distribution

        Returns:
        out:ndarray

        """
        # Initialize points vector
        points = np.zeros(n)
        for x in range(n):
            # Choose one gaussian over possibilities with know probabilities like weigths
            gauss = random.choices(self.getGaussianParameters(), weights=self.getProbabilities())[0]
            # Generate a point using choosed gaussian
            points[x]  = np.random.normal(gauss[0], gauss[1])
        return points

    def stepExpectation(self, data):
        """
        Do expectation step over the instanced EM model

        Parameters:
        data:ndarray

        """
        #Prepare Z matrix
        Z = np.zeros((self.getNumOfGaussians(), len(data)))
        #Iterate over data
        for m, x in enumerate(data):
            sumF = 0
            for prior, gauss in zip(self.getProbabilities(), self.getGaussianParameters()):
                sumF += self.getJointProbability(x, prior, gauss)
            for k, (prior, gaussiana) in enumerate(zip(self.getProbabilities(), self.getGaussianParameters())):
                    Z[k, m] = self.getJointProbability(x, prior, gaussiana) / sumF
        return Z

    def stepMaximization(self, Z, data, r_means=False, r_vars=False):
        """
        Do maximization step over the instanced EM model

        Parameters:
        Z:ndarray
        data:ndarray
        r_priors:boolean
        r_means:boolean
        r_vars:boolean

        """
        n = np.zeros(self.getNumOfGaussians())
        for k in range(self.getNumOfGaussians()):
            n[k] = sum(Z[k, :])


        # Reestimate priorities
        M = len(data)
        for ind, n_k in enumerate(n):
            self.getProbabilities()[ind] = n_k / M

        if r_means:
            # Reestimate means
            for k in range(self.getNumOfGaussians()):
                tSum = 0
                for m, point in enumerate(data):
                    tSum += Z[k, m] * point
                self.getGaussianParameters()[k][0] = tSum / n[k]
        if r_vars:
            # Reestimate variance
            for k in range(self.getNumOfGaussians()):
                tSum = 0
                for m, point in enumerate(data):
                    tSum += (Z[k,m] * (point - self.getGaussianParameters()[k][0]) ** 2)
                self.getGaussianParameters()[k][1] = np.sqrt( tSum / n[k])

    def run(self, data, r_means=False, r_vars=False):
        """
        Do algorithm: first execute expectation; then maximization

        Parameters:
        Z:ndarray
        data:ndarray
        r_means:boolean
        r_vars:boolean

        """
        Z = self.stepExpectation(data)
        self.stepMaximization(Z, data, r_means, r_vars)

    def getDensity(self, x):
        """
        Retrieve density by a point

        Parameters:
        x:float

        Returns:
        out:float

        """
        out = 0
        for p, g in zip(self.getProbabilities(), self.getGaussianParameters()):
            out += self.getJointProbability(x, p, g)
        return out

    def getJointProbability(self, point, priori, gaussiana):
        """
        Retrieve joint density probability

        Parameters:
        point:float
        priori:float
        gaussiana: list

        Returns:
        out:float

        """
        mean, std = gaussiana
        var = std ** 2
        posteriorProb = np.exp(- (point - mean)**2 / (2*var)) / np.sqrt(2* np.pi * var)
        return priori * posteriorProb
