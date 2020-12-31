import tensorflow as tf

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import Optimizers_from_peaper as HdOpts
import CustomOptimizers as MyOpts 

from Models import Fully_Connected_NN
from sklearn.datasets import load_digits

tf.random.set_seed(0)
torch.manual_seed(0)

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
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


### Comuni
iris_data = load_digits() 

learning_rate = 0.0001
beta= 1e-8

### PyTorch
X_numpy = iris_data.data.astype(np.float32)
X_numpy = (X_numpy - X_numpy.mean())/ X_numpy.mean()
y_numpy= iris_data.target

X = torch.tensor(X_numpy, dtype=torch.float32)  
y = torch.tensor(y_numpy, dtype=torch.int64)
 
y_one_hot = one_hot_encode(y)
random_indices = torch.randperm(X.shape[0])  

X_train = X[random_indices]  
y_train = y[random_indices]

model_torch = MLP(input_dim=64, hidden_dim=1000, output_dim=10)

# PyTorch Optimizers
#optimizer_torch = HdOpts.SGDHD(model_torch.parameters(), lr=learning_rate,
#                               hypergrad_lr=beta)
#optimizer_torch = HdOpts.SGDHD(model_torch.parameters(), lr=learning_rate, nesterov = True,
#                                hypergrad_lr=beta, momentum = 0.9, dampening = 0)
optimizer_torch = HdOpts.AdamHD(model_torch.parameters(), lr=learning_rate,
                                hypergrad_lr=beta)

ws = []
for k in ['lin1','lin2','lin3']:
    loss_function_torch = torch.nn.CrossEntropyLoss()
    w_copy = deepcopy(model_torch.state_dict()[k+".weight"])
    weight = np.transpose(w_copy.numpy())
    b_copy = deepcopy(model_torch.state_dict()[k+".bias"])
    bias = np.transpose(b_copy.numpy())
    
    ws.append(weight)
    ws.append(bias)

### Tensorflow
x_tf = X_numpy
y_tf = iris_data.target.reshape(-1, 1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_tf, y_tf))
train_dataset = train_dataset.shuffle(1797).batch(1797)

model_keras = Fully_Connected_NN(input_shape=(64,), hidden_dim=1000,
                                 n_class=10, kernel_pen=0.0)

model_keras.set_weights(ws)
model_keras.get_weights()


loss_object_tf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# TensorFlow Optimizer

#optimizer_tf = MyOpts.SGD_HD(alpha_0=learning_rate, beta=beta)
#optimizer_tf = MyOpts.SGDN_HD(alpha_0=learning_rate, beta=beta)
optimizer_tf = MyOpts.Adam_HD(alpha_0=learning_rate, beta=beta)

train_loss_tf = tf.keras.metrics.Mean('train_loss', dtype = tf.float32)
train_accuracy_tf = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

## Check Condizioni Iniziali

n_iterations = 1000

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
        
# =============================================================================
    print("PyTorch Loss at iteration {}:    {}".format(i, loss_torch))
    print("TensorFlow Loss at iteration {}: {}".format(i, train_loss_tf.result().numpy()))
    print("\nalpha_t: {}".format(optimizer_torch.state_dict()["param_groups"][0]["lr"]*1))
    print("alpha_t: {}".format(optimizer_tf._get_alpha().numpy()*1))
    print("___")

    
    # Reset metrics 
    train_loss_tf.reset_states()
    train_accuracy_tf.reset_states()
