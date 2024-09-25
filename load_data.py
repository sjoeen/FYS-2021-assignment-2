import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from prediction_model import Gaussian
from prediction_model import Gamma




def load_data(file_name):
    data = pd.read_csv(file_name, sep=',',header=None)

    return data.to_numpy()


if __name__ == "__main__":

    dataset = load_data("data_problem2.csv")
    #print(dataset)

    class_0 = dataset[0, :]  
    class_1 = dataset[1,:]

    plt.figure(figsize=(10,6))

    # Plot class 0
    plt.hist(class_0, bins=10, alpha=0.5, label="Class 0")

    # Plot class 1
    plt.hist(class_1, bins=10, alpha=0.5, label="Class 1")

    # Add labels and title
    plt.xlabel("Class value")
    plt.ylabel("frequency")
    plt.title('Histogram of class 0 and class 1')
    plt.legend()

    # Show the plot
    plt.show()
    
    class_0_train,class_0_test = train_test_split(class_0,test_size=0.2,random_state=21)
    class_1_train,class_1_test = train_test_split(class_1,test_size=0.2,random_state=21)

    test_set = np.concatenate((class_0_test,class_1_test))
    
    a = Gaussian(class_0_train)
    #print(a.var)
    print(a.log_likelihood(test_set))
    b = Gamma(class_1_train)
    print(b.beta)
    print(b.log_likelihood(test_set))



