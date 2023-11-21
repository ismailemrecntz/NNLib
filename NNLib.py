import numpy as np
import pandas as pd



class NNLib:
    def __init__(self,learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate


    def get_train_test_data(self,train_ratio:int, data: pd.DataFrame, label = "",random_state = 10,return_np_array = 1):
        
        train = data.sample(frac = train_ratio * 0.01,random_state = random_state)
        y_train = train[label]
        x_train = train.drop(label,axis=1)
        
        
        test = data[~data.isin(train)].dropna()
        test = test.astype(int)
        y_test = test[label]
        x_test = test.drop(label,axis=1)
        
        if return_np_array == 1:
            x_train = x_train.to_numpy()
            y_train = y_train.to_numpy()
            x_test = x_test.to_numpy()
            y_test = y_test.to_numpy()
        
        return x_train, y_train, x_test, y_test
        

    def apply_normalization(self,data,normalization=""):
        norm_data = []
        
        if normalization == "gaussian":
            for i in data.T:
                mean = np.mean(i)
                std = np.std(i)
                norm = ((i - mean) /std)
                norm_data.append(norm)
                
            return np.array(norm_data).T
        