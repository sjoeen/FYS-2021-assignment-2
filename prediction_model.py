import numpy as np
from typing import Optional, Tuple


class Gaussian:
    """
    class that represents a gaussian distrubution of
    a dataset, it contains mean,variance and a log
    likelihood function.
    """

    def __init__(self,x: np.array,prior: float):

        self.x = x
        self.N = len(x)
        self.mean = x.mean()
        self.var = 0
        self.prior = prior


        for i in range(self.N):
            self.var += (x[i]-self.mean)**2
        self.var /= self.N
        
        self.std = np.sqrt(self.var)


    def log_likelihood(self,predictions: np.array) -> np.array:
        """
        This function takes the log likelihood in order to check how likely it is that a 
        set of values is under our gaussian distribution. Proof of this formula is under
        point 2.2.
        """
        predictions = np.array(predictions,dtype=np.float64)
        
        term1 = -np.log(self.std * (np.sqrt  (2*np.pi) ) )
        term2 = (-1/2) * ((predictions-self.mean)/self.std)**2
        log_likelihoods = term1 + term2 + np.log(self.prior)

        return log_likelihoods

class Gamma:
    """
    class that represents a gamma distrubution of
    a dataset, it contains beta,gamma and a log
    likelihood function.

    NOTE: 
    This function was constructed with a given
    gamma
    """

    def __init__(self,
                 x:np.array,
                 prior: np.array,
                 gamma=2):

        self.x = x
        self.N = len(x)
        self.gamma = gamma
        self.beta = 0
        self.prior = prior

        for i in range(self.N):
            self.beta += x[i]
        self.beta /= ( self.N * self.gamma )

    def log_likelihood(self,predictions: np.array) -> np.array:
        """
        This function takes the log likelihood in order to check how likely it is that a 
        set of values is under our gamma distribution. Proof of this formula is under
        point 2.2.
        """

        predictions = np.array(predictions,dtype=np.float64)
        predictions = np.clip(predictions, 1e-8, None)
            #gamma distrubution can't handle negative values
            #chatgpt helped with this cmd when problem shooting. 

        term1 = self.gamma * np.log(self.beta)
        term2 = np.log(self.gamma)
        term3 = np.log(predictions)
        term4 = - predictions  / self.beta

        log_likelihoods = self.N * (term1 + term2 + 
                                    term3 + term4 + 
                                    np.log(self.prior))

        return log_likelihoods


class BayesClassifier:
    """
    Bayes classifier class compares the log likelihood of each class
    in the classes list and return the accuracy.
    """

    def __init__(self,
                 x: np.array,
                 classes: list,
                 y: np.array):

        self.x = x
        self.num_samples = len(x)
        self.classes = classes
        self.y = y

    def _compare(self) -> np.array:
        """
        A help function for the compare function.
        This function returns a  numpy list of the most likely
        class for each sample.
        """

        log_likelihoods = []

        for model in self.classes:
            predictions = model.log_likelihood(self.x)
            log_likelihoods.append(predictions)

        log_likelihoods = (np.column_stack(log_likelihoods))
                #got help from chatgpt with this cmd

        return np.argmax(log_likelihoods,axis=1)
    
    def compare(self) -> float:
            """
            A compare function that returns the accuracy by comparing
            the samples to the labels.
            """

            predictions = self._compare()
            correct_counter = 0
            correct_list = (predictions == self.y) 

            for pred in correct_list:

                if pred:
                    correct_counter+=1

            accuracy = round(correct_counter / self.num_samples,4) 

            return accuracy * 100
    
    def miscalulation(self) -> list:
        """
        A function that returns a list of misclassifications.
        """
        predictions = self._compare()
        incorrect_list = (predictions != self.y)

        data_points = []

        for index,pred in enumerate(incorrect_list):
            if pred:
                data_points.append(self.x[index])


        return data_points