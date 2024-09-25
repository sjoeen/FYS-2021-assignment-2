import numpy as np


class Gaussian:

    def __init__(self,x):

        self.x = x
        self.N = len(x)
        self.mean = x.mean()
        self.var = 0


        for i in range(self.N):
            self.var += (x[i]-self.mean)**2
        self.var /= self.N
        
        self.std = np.sqrt(self.var)


    def log_likelihood(self,predictions):
        """
        This function takes the log likelihood in order to check how likely it is that a 
        set of values is under our gaussian distribution. Proof of this formula is under
        point 2.2.
        """


        predictions = np.array(predictions,dtype=np.float64)
        
        prefactors = -np.log(self.std * (np.sqrt  (2*np.pi) ) )
        exponent = (-1/2) * ((predictions-self.mean)/self.std)**2
        return prefactors + exponent

class Gamma:

    def __init__(self,x,gamma=2):

        self.x = x
        self.N = len(x)
        self.gamma = gamma
        self.beta = 0

        for i in range(self.N):
            self.beta += x[i]
        self.beta /= ( self.N * self.gamma )

    def log_likelihood(self,predictions):

        predictions = np.array(predictions,dtype=np.float64)
        predictions = np.clip(predictions, 1e-8, None)

        prefactors = -2*self.N *np.log(self.beta) + np.log(predictions)
        factor = (-1/self.beta) * predictions

        return prefactors + factor


class BayesClassifier:

    def __init__(self,x,y,classes):

        self.x = x