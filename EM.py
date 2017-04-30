#/usr/bin/python3.6
import numpy as np
import random

class modelEM:
    def __init__(self, gaussParams=[[-6, 2], [2, 2]], probabilities=[0.4, 0.6]):
        self.probabilities=probabilities
        self.gaussParams=gaussParams

    def expectation(self,data):
        Z = np.zeros((len(self.probabilities), len(data)))
        for m, p in enumerate(data):
            sumF = sum(self.estimate_conjunt_density(p, prior, gauss) for prior, gauss in zip(self.probabilities, self.gaussParams))
            for k, (prior, gaussiana) in enumerate(zip(self.probabilities, self.gaussParams)):
                    Z[k,m] = self.estimate_conjunt_density(p, prior, gaussiana) / sumF
        return Z

    def reestimateVariance(self):
        for k, gauss in enumerate(self.gaussParams):
                self.gaussParams[k][1] = np.sqrt( sum( Z[k,m]*(point - self.gaussParams[k][0])**2 for m,point in enumerate(data)) / n[k])
    def maximization(self, Z, data, r_priors=False, r_means=False, r_vars=False):
        n = [sum( Z[k, :]) for k in range(len(self.probabilities))]
        m = len(data)
        if r_priors:
                self.probabilities = [n_k / m for n_k in n]
        if r_means:
                for k in range(len(self.gaussParams)):
                        self.gaussParams[k][0] = sum(Z[k,m]*point for m,point in enumerate(data)) / n[k]

        if r_vars:
            self.reestimateVariance()

    def reestimate(self, data, r_priors=False, r_means=False, r_vars=False):
        Z = self.expectation(data)
        self.maximization(Z, data, r_priors, r_means, r_vars)

    def getDensityPoint(self, x):
        res = 0
        for p, g in zip(self.probabilities, self.gaussParams):
            res += self.estimate_conjunt_density(x, p, g)
        return res

    def getPropFromNormal(self, point, gaussiana):
        mean, std = gaussiana
        var = std ** 2
        return 1 / np.sqrt(2* np.pi * var) * np.exp(- (point - mean)**2 / (2*var))

    def generateData(self, n=500):
        points = np.zeros(n)
        for x in range(n):
            gauss = random.choices(self.gaussParams, weights=self.probabilities)[0]
            points[x]  = np.random.normal(gauss[0], gauss[1])
        return points

    def estimate_conjunt_density(self, point, priori, gaussiana):
        return priori * self.getPropFromNormal(point, gaussiana)
