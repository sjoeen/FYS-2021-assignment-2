import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from prediction_model import Gaussian
from prediction_model import Gamma
from prediction_model import BayesClassifier





def _load_data(file_name):
    """
    A helping function that opens a file
    """
    data = pd.read_csv(file_name, sep=',',header=None)

    return data.to_numpy()




def classify(file_name):
    """
    This function splits the classes using their labels.
    """

    dataset = _load_data(file_name) 
 
    labels = dataset[1, :] 
    values = dataset[0, :]  
    class_0 = values[labels == 0]  
    class_1 = values[labels == 1]
        #Defining the classes

    return class_0,class_1




if __name__ == "__main__":


    class_gamma,class_gaussian = classify("data_problem2.csv")



    """
    Cleaning up,splitting and giving labels to the classes in the dataset.
    """

    gamma_train,gamma_test = train_test_split(class_gamma,test_size=0.2)
    gaussian_train,gaussian_test = train_test_split(class_gaussian,test_size=0.2)
    gamma_test = np.column_stack((gamma_test, np.zeros(gamma_test.shape[0])))
        #giving labels back
    gaussian_test = np.column_stack((gaussian_test, np.ones(gaussian_test.shape[0])))
        #giving labels back
    test_set = np.concatenate((gamma_test,gaussian_test))
        #adding the distrubutions into one test set
    test_data = test_set[:, 0]  
    test_labels = test_set[:, 1]  
        #splitting labels and datapoint (X and Y)

    """
    Defining the classes:
    """

    gassuian_prior = 2000/3600
    gamma_prior = 1600/3600

    gaussian = Gaussian(gaussian_train,gassuian_prior)
    gamma = Gamma(gamma_train,gamma_prior)
    class_list = [gamma,gaussian]
    bayes = BayesClassifier(test_data,class_list,test_labels)
    print(bayes.compare())
    missclassified = (bayes.miscalulation())

    """
    plot data and missclassified samples.
    """

    plt.figure(figsize=(10,6))
    plt.hist(class_gamma, bins=10, alpha=0.5, label="Class 0")
    plt.hist(class_gaussian, bins=10, alpha=0.5, label="Class 1")
    plt.hist(missclassified, bins=10, alpha=1, label="miss-classification")
    plt.xlabel("Class value")
    plt.ylabel("frequency")
    plt.title('Histogram of class 0 and class 1')
    plt.legend()
    plt.show()
