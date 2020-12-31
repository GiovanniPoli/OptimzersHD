import torch
import tensorflow as tf
import numpy as np
from copy import deepcopy

import Optimizers_from_peaper as HdOpts
import CustomOptimizers as MyOpts 


from Models import Softmax_Model
from sklearn.datasets import load_iris

def train_step_tf(model, optimizer, x_train, y_train): # TensorFlow
  with tf.GradientTape() as tape:
    predictions = model(x_train, training=True)
    loss = loss_object_tf(y_train, predictions)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  train_loss_tf(loss)
  train_accuracy_tf(y_train, predictions)

def one_hot_encode(vector): # Torch
    n_classes = len(vector.unique())  
    one_hot = torch.zeros((vector.shape[0], n_classes))\
        .type(torch.LongTensor)  
    return one_hot\
        .scatter(1, vector.type(torch.LongTensor).unsqueeze(1), 1) 
        
### Comuni
iris_data = load_iris() 

learning_rate = 1e-5
beta= 0.001

### PyTorch
X_numpy = iris_data.data
y_numpy= iris_data.target

X = torch.tensor(X_numpy, dtype=torch.float32)  
y = torch.tensor(y_numpy, dtype=torch.int64)
 
y_one_hot = one_hot_encode(y)
random_indices = torch.randperm(X.shape[0])  

X_train = X[random_indices]  
y_train = y[random_indices]

model_torch = torch.nn.Sequential(torch.nn.Linear(4, 3))
optimizer_torch = HdOpts.SGDHD(model_torch.parameters(), lr=learning_rate,
                               nesterov= True, hypergrad_lr=beta, momentum=0.9)

loss_function_torch = torch.nn.CrossEntropyLoss()
w_copy = deepcopy(model_torch.state_dict()["0.weight"])
weight = np.transpose(w_copy.numpy())
b_copy = deepcopy(model_torch.state_dict()["0.bias"])
bias = np.transpose(b_copy.numpy())

### Tensorflow
x_tf = iris_data.data
y_tf = iris_data.target.reshape(-1, 1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_tf, y_tf))
train_dataset = train_dataset.shuffle(150).batch(150)

model_keras = Softmax_Model(input_shape=(4,), n_class=3)

model_keras.set_weights([weight,bias])

loss_object_tf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer_tf = MyOpts.SGDN_HD(alpha_0=learning_rate, beta=beta, mu=0.9)

train_loss_tf = tf.keras.metrics.Mean('train_loss', dtype = tf.float32)
train_accuracy_tf = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

## Check Condizioni Iniziali

print(model_keras.trainable_weights[0].numpy())
print(model_keras.trainable_weights[1].numpy())
print(model_torch[0].bias)
print(model_torch[0].weight)
print("\n\n")

n_iterations = 10

for i in range(1, n_iterations + 1):
    ## PyTorch
    Z = model_torch(X_train)  
    loss_torch = loss_function_torch(Z, y_train)  
    optimizer_torch.zero_grad()  
    loss_torch.backward()  
    optimizer_torch.step() 
    
    ## TensorFlow
    for (x_train_tf, y_train_tf) in train_dataset: # Inutile, serve solo per non cambiare "sintassi" 
        with tf.GradientTape() as tape:
            predictions_tf = model_keras(x_train_tf, training=True)
            loss_tf_cycle = loss_object_tf(y_train_tf, predictions_tf)
        grads_tf = tape.gradient(loss_tf_cycle, model_keras.trainable_variables)
        optimizer_tf.apply_gradients(zip(grads_tf, model_keras.trainable_variables))

        train_loss_tf(loss_tf_cycle)
        train_accuracy_tf(y_train_tf, predictions_tf)
        
    print("\n\nPyTorch Loss at iteration {}:    {}".format(i, loss_torch.detach().numpy().round(5)))
    print("TensorFlow Loss at iteration {}: {}".format(i, train_loss_tf.result().numpy().round(5)))
    print("\ngradients (PyTorch):")
    print(np.transpose(model_torch[0].weight.grad))
    print(model_torch[0].bias.grad)
    print("gradients (TensorFlow):")
    print(grads_tf[0].numpy().round(4))
    print(grads_tf[1].numpy().round(4))
    print("\nweights (PyTorch):")
    print(np.transpose(model_torch[0].weight.detach().numpy()).round(4))
    print(model_torch[0].bias.detach().numpy().round(4))
    print("weights (TensorFlow):")
    print(model_keras.get_weights()[0].round(4))
    print(model_keras.get_weights()[1].round(4))
    print("\nalpha_t: {}".format(optimizer_torch.state_dict()["param_groups"][0]["lr"].numpy().round(6)))
    print("alpha_t: {}".format(optimizer_tf._get_alpha().numpy().round(6)))
    
    # Reset metrics 
    train_loss_tf.reset_states()
    train_accuracy_tf.reset_states()
    
    


