import numpy as np
import pandas as pd

class Layer:
    def __init__(self,dimension = 0,activation_function = "",is_input_layer = False, is_output_layer = False):
        if dimension != 0:
            self.dimension = dimension
            self.activation_function = activation_function
        else:
            raise ValueError("Dimension must be different from zero!")
    

class NNLib:
    def __init__(self,learning_rate):
        self.weights = {}
        self.bias = {}
        self.model = {}
        self.layers = []
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
    
    def apply_activation_function(self,data,function=""):
        if function == "relu":
            return np.maximum(0,data)
            
    def derivative_relu(self,data):
        data = [1 if value>0 else 0 for value in data]
        return np.array(data, dtype=float)
    
    def create_network(self,train_data):
        layer = Layer(train_data.shape[1],is_input_layer=True)
        self.layers.append(layer)
             
    def initialize_parameters(self, method = ""):
        if method == "He":
            if len(self.layers) < 2:
                raise ValueError("There is no added layer!")
            else:
                self.weights["0"] = np.random.randn(self.layers[1].dimension, self.layers[0].dimension) * np.sqrt(2/self.layers[0].dimension)
                self.bias["0"] = np.zeros((self.layers[1].dimension, 1))
                
    def add_layer(self,dimension,activation_function):
        layer = Layer(dimension,activation_function)
        self.layers.append(layer)
        
    def train(self):
        pass
    
    def predict(self):
        pass