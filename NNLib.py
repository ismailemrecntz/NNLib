import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,precision_score,recall_score,f1_score

class Layer:
    def __init__(self,dimension = 0,activation_function = ""):
        if dimension != 0:
            self.dimension = dimension
            self.activation_function = activation_function.lower()
        else:
            raise ValueError("Dimension must be different from zero!")
        
    def apply_activation_function(self,data):
        if self.activation_function == "relu":
            return np.maximum(0,data)
        elif self.activation_function == "softmax":
            e_x = np.exp(data - np.max(data, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)
        
        else:
            raise ValueError("Activation function is invalid!")
        
    def derivative_relu(self,data):
        data = [1 if value>0 else 0 for value in data]
        return np.array(data, dtype=float)
    

class NNLib:
    def __init__(self):
        self.weights = {}
        self.bias = {}
        self.forward_prop_variables = {}
        self.layers = []
        self.learning_rate = 0
        self.momentum = 0
        self.initilization_method = ""
        self.mini_batch_size = 0
        self.loss_function = ""
        self.training_loss_sum = []
        self.trainin_loss_per_epoch = []


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
            y_train = y_train.to_numpy().reshape(y_train.shape[0],1)
            x_test = x_test.to_numpy()
            y_test = y_test.to_numpy().reshape(y_test.shape[0],1)
        
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
        
    def apply_one_hot_encoding(self,data):  
        one_hot_encode = []

        for i in range(len(data)):
            if data[i] == 0:
                one_hot_encode.append(1)
                one_hot_encode.append(0)
            else:
                one_hot_encode.append(0)
                one_hot_encode.append(1)
    
        one_hot_encode = np.array(one_hot_encode).reshape(len(data),2)
        return one_hot_encode
    
    def create_network(self,train_data, parameters_initilization_method,learning_rate, loss_function):
        layer = Layer(train_data.shape[1])
        self.layers.append(layer)
        self.initilization_method = parameters_initilization_method.lower()
        self.learning_rate = learning_rate
        self.loss_function = loss_function.lower()
             
    def initialize_parameters(self, method = ""):
        if method == "he":
            if len(self.layers) < 2:
                raise ValueError("There is no added layer!")
            else:
                for i in range(1,len(self.layers)):
                    self.weights[i] = np.random.randn(self.layers[i].dimension, self.layers[i-1].dimension) * np.sqrt(2/self.layers[i-1].dimension)
                    self.bias[i] = np.zeros((self.layers[i].dimension, 1))
                
    def add_layer(self,dimension = 0,activation_function = ""):
        layer = Layer(dimension,activation_function)
        self.layers.append(layer)
      
    def cross_entropy(self,y_pred, y_true):
        total_loss = 0
        for i in range(len(y_pred)):
            loss = np.sum((-1 * y_true[i]*np.log(y_pred[i])))
            total_loss += loss

        return total_loss / len(y_pred) 
    
    def apply_loss_function(self,y_pred,y_true):
        if self.loss_function == "cross_entropy":
            return self.cross_entropy(y_pred,y_true)
        else:
            raise ValueError("Loss function is invalid!")
        
    def train(self, x_train, y_train, mini_batch_size,epoch = 1000):
        self.mini_batch_size = mini_batch_size
        self.initialize_parameters(self.initilization_method)
        for i in range(epoch):
            self.trainin_loss_per_epoch = []
            for batch_index in range(0, (int(x_train.shape[0] / mini_batch_size) - 1) ):
                X = x_train[batch_index*mini_batch_size: (batch_index+1) * mini_batch_size,:]
                Y = y_train[batch_index*mini_batch_size: (batch_index+1) * mini_batch_size,:]
                forward_values = self.forward(X)
                loss = self.apply_loss_function(forward_values["X"+str(len(self.layers) - 1)], Y)
                self.trainin_loss_per_epoch.append(loss)
                gradients = self.backward(forward_values,Y)
                self.update_parameters(gradients)
                
            X = x_train[(batch_index + 1) * mini_batch_size: ,:]
            Y = y_train[(batch_index + 1) * mini_batch_size: ,:]
            forward_values = self.forward(X)
            loss = self.apply_loss_function(forward_values["X"+str(len(self.layers) - 1)], Y)
            self.trainin_loss_per_epoch.append(loss)
            print("Epoch = " + str(i+1) + " Average Loss = " + str(np.sum(self.trainin_loss_per_epoch) / (batch_index + 1)))
            self.training_loss_sum.append(self.trainin_loss_per_epoch)
            gradients = self.backward(forward_values,Y)
            self.update_parameters(gradients)
            print("----------------------------------------------------------")
        
        average_loss_for_all_epochs =  ([sum(x)/len(self.training_loss_sum) for x in zip(*self.training_loss_sum)])
        self.draw(average_loss_for_all_epochs,title= "Loss for average of all epochs")

           
    def forward(self,X):
        self.forward_prop_variables = {"X0" : X}
        for i in range(1,(len(self.layers))):
            self.forward_prop_variables["Z"+str(i)] = np.dot(self.forward_prop_variables["X"+str(i-1)],self.weights[i].T) + self.bias[i].T
            self.forward_prop_variables["X"+str(i)] = self.layers[i].apply_activation_function(self.forward_prop_variables["Z"+str(i)])

        return self.forward_prop_variables
    
    def backward(self,forward_prop_variables,Y):
        gradients = {}
        size = len(self.layers) - 1
        
        for i in range(size,0,-1):
            m = forward_prop_variables["X"+str(i-1)].shape[0]
            
            if i == size:
                gradients["dZ"+str(i)] = forward_prop_variables["X"+str(i)] - Y 

            else:
                relu_derivative = forward_prop_variables["X"+str(i)] > 0
                gradients["dZ"+str(i)] = np.multiply(gradients["dX"+str(i)].T, relu_derivative)
            
            gradients["dW"+str(i)] = (1/m) * np.dot(gradients["dZ"+str(i)].T, forward_prop_variables["X"+str(i-1)])
            gradients["db"+str(i)] = (1/m) * np.sum(gradients["dZ"+str(i)], axis=0, keepdims=True)
            gradients["dX"+str(i-1)] = np.dot(self.weights[i].T, gradients["dZ"+str(i)].T)
        return gradients
 
    def update_parameters(self,gradients):
        velocity_weights = 0
        velocity_bias = 0
        for i in range(1, len(self.layers)):
            self.weights[i] -= self.learning_rate * gradients["dW"+str(i)]
            self.bias[i] -= self.learning_rate * gradients["db"+str(i)].T  
    
    def predict(self, X):

        forward_values = self.forward(X)
        Length = len(self.layers) - 1
        result = []
        
        for i in range(len(X)):
            
            if forward_values["X"+str(Length)][i,0] > forward_values["X"+str(Length)][i,1]:
                result.append(0)
            else: 
                result.append(1)
                
        return np.array(result).reshape(len(X),1)
    
    def evaluate(self,y_true,y_pred):
        
        accuracy = accuracy_score(y_true,y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print("Accuracy = " +str(accuracy))
        print("Precision = " +str(precision))
        print("Recall = " +str(recall))
        print("Auc Score = " +str(auc_score))
        print("F1 Score = " +str(f1))
        print("")
        print("Confusion Matrix")
        print("-------------------")
        print(confusion_matrix(y_true, y_pred))
        
    
    def draw(self,data,title = ""):
        plt.figure()
        plt.title(title)
        plt.plot(range(0,len(data)), data, linestyle='dashed')
        plt.show()