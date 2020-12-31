import tensorflow as tf
import numpy as np
from tensorflow import keras

from keras.datasets import mnist

import pickle
import time as time
from copy import deepcopy
from math import isnan

import matplotlib.pyplot as plt

from Models import Softmax_Model
import CustomOptimizers as MyOpts 
     
def test_step(model, x_test, y_test):
  predictions = model(x_test)
  loss = loss_object(y_test, predictions)
  
  test_loss(loss)
  test_accuracy(y_test, predictions)

def save_result(obj, name ):
    with open(''+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_result(name ):
    with open('' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    


tf.random.set_seed(3)

print("Versione tensorfolow:", tf.__version__)
print('Grid Serch')

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32)
x_test  = x_test.astype(np.float32)

mean =  6./7. * x_train.mean() + 1./7. * x_test.mean()
sd   = (6./7. * x_train.var()  + 1./7. * x_test.var())**0.5  

x_train, x_test = (x_train - mean)/ sd, (x_test - mean) / sd

train_dataset_base = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(128)


MAX_EPOCHS = 50
Done  = False 

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

Grid = [[float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')],
        [float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')],
        [float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')],
        [float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')],
        [float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')],
        [float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')]]



model_base = Softmax_Model((28,28), 10, kernel_pen=1e-4)
weight = model_base.get_weights()

alphas = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
betas =  [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1, float('nan')]
for i in range(6):
  for j in range(7):

    train_dataset = train_dataset_base.shuffle(60000,seed=0).batch(128)
    Not_done = True  
    alpha = alphas[i]
    beta = betas[j]
    print_ = True
    
    if isnan(beta):
        opt = keras.optimizers.Adam(lr=alpha,epsilon=1e-8,
                                    beta_1=0.9,beta_2=0.999)
    else: 
        opt = MyOpts.Adam_HD(alpha_0= alpha, beta= beta, epsilon=1e-8,
                                    beta_1=0.9,beta_2=0.999)
    
    model = Softmax_Model((28,28), 10, kernel_pen=1e-4, bias_pen=1e-4)
    model.set_weights(deepcopy(weight))
       
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')  
    
    iteration = 0
    print('alpha= {}, beta={}'.format(alpha, beta))
    for epoch in range(MAX_EPOCHS):
        if Not_done:
            t0 = time.time() 
            for (x_train, y_train) in train_dataset:
                if Not_done:
                    iteration += 1
                    with tf.GradientTape() as tape:
                      predictions = model(x_train, training=True)
                      loss = loss_object(y_train, predictions)
                      loss_pen = loss + tf.add_n(model.losses)

                    grads = tape.gradient(loss_pen, model.trainable_variables)
                    new_learning_rate = 0.01 
                    
                    if isnan(beta):
                        opt.lr.assign(alpha/(iteration)**0.5)
                    
                    opt.apply_gradients(zip(grads, model.trainable_variables))
                    
                    train_loss(loss)
                    train_accuracy(y_train, predictions)
                      
                    if loss.numpy() <= 0.29:
                        Not_done = False 
                        print_ = False
                        print('Training ended during iteretion {} of epoch {}\n'.format(iteration-(epoch*469),epoch+1))
                            
            for (x_test, y_test) in test_dataset:
                test_step(model, x_test, y_test)
            t1 = time.time()
                
            if print_:
                template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Time:   {}s'
                print (template.format(epoch+1,
                                         round(train_loss.result().numpy()*1,5), 
                                         round(train_accuracy.result().numpy()*100,2),
                                         round(test_loss.result().numpy()*1,5), 
                                         round(test_accuracy.result().numpy()*100,2),
                                         round(t1-t0*1,1)))
                
            # Reset metrics every epoch
            train_loss.reset_states()
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()
        
    Grid[i][j] = iteration
    

Grid_np = np.array(Grid)
save_result(Grid_np, 'Output_GridSerch_Adam')
# Grid_np = load_result('Output_GridSerch_Adam')

fig, ax = plt.subplots(figsize=(18, 8))

xlab = ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$",
        "$10^{-3}$","$10^{-2}$","$10^{-1}$","Ref. Adam"]
ylab = ["$10^{-1}$", "$10^{-2}$", "$10^{-3}$",
        "$10^{-4}$","$10^{-5}$","$10^{-6}$"]

col = -1
for i in Grid_np:
    col += 1
    row = -1
    for j in i:
        row += 1
        if j==MAX_EPOCHS*469:
            Grid_np[col,row] = 0.0
            
_min = Grid_np.min()
_max = Grid_np.max()
im = ax.imshow(Grid_np)
cut = (_min + _max) /2.

c = plt.pcolor([[],[]], vmin= _min ,vmax= _max)

ax.set_xticks(np.arange(len(xlab)))
ax.set_yticks(np.arange(len(ylab)))
ax.set_xticklabels(xlab,size=16)
ax.set_yticklabels(ylab,size=16)

plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
         rotation_mode="anchor")
plt.plot([5.5,5.5],[-0.5,5.5], linewidth=3, c='white')
    

# Loop over data dimensions and create text annotations.
for i in range(len(xlab)):
    for j in range(len(ylab)):
        it= Grid_np[j, i]
        if it<= cut:
            col = "w"
        else:
            col = "black"
        
        if it!=0:
            text = ax.text(i, j, "{}".format(it),
                           ha="center", va="center", color=col,size=16)
        else:
            text = ax.text(i, j, "",
                           ha="center", va="center",size=16)

plt.plot([-.5,2.5,2.5,3.5,3.5,4.5,4.5],
         [0.5,0.5,1.5,1.5,2.5,2.5,5.5],
         linewidth=2, c='r')

plt.fill([4.5,4.5,5.5,5.5],[-.5,5.5,5.5,-.5],
         c='grey', alpha=1)

plt.fill([6.5,6.5,5.5,5.5],[2.5,5.5,5.5,2.5],
         c='grey', alpha=1)
plt.fill([-.5,2.5,2.5,3.5,3.5,4.5,4.5,5.5,5.5,-.5],
         [0.5,0.5,1.5,1.5,2.5,2.5,5.5,5.5,-.5,-.5],
         c='r', alpha=0.2)
text = ax.text(6, 4, " More then \n {}".format(MAX_EPOCHS*469),
                          ha="center", va="center", color="white",size=15)
text = ax.text(5, 2.5, " More then \n {}".format(MAX_EPOCHS*469),
                          ha="center", va="center", color="white",size=15)

ax.set_title("Grid serch for Adam-HD",size=20)
plt.xlabel(r'$\beta$',size=20)
plt.ylabel(r'$\alpha$',size=20)
fig.tight_layout()
fig.colorbar(c)

plt.show()