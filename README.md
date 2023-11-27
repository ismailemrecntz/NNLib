# NNLib

NNLib can be used to create neural network models. This module does not use Tensorflow or other ai libraries because this module aims to understand background mathematics of a neural
network model and easily make hyperparameter tuning.



## Requirements
* Python 3.x
* Numpy
* Pandas
* Matplotlib
* Sklearn.metrics


## Example
  model = NNLib()
  
  x_train, y_train, x_test, y_test = model.get_train_test_data(train_ratio= 70, data = df, label = "label", return_np_array= 1)
  
  
  x_train_norm = model.apply_normalization(data= x_train,normalization="gaussian")
  
  
  x_test_norm  = model.apply_normalization(data= x_test,normalization="gaussian")
  
  
  y_train_one_hot_encoding = model.apply_one_hot_encoding(y_train)
  
    
  model.create_network(x_train_norm, parameters_initilization_method="He",learning_rate = 0.01,loss_function="cross_entropy")


  model.add_layer(64, activation_function = "Relu")

  
  model.add_layer(32, activation_function = "Relu")

  
  model.add_layer(16, activation_function = "Relu")

  
  model.add_layer(2, activation_function  = "softmax")

  
  model.train(x_train_norm, y_train_one_hot_encoding, mini_batch_size=10)

  
  predictions = model.predict(x_test_norm)

  
  model.evaluate(y_pred=predictions,y_true=y_test)
