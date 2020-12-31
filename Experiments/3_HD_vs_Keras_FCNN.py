import tensorflow as tf
import numpy as np
from tensorflow import keras

from keras.datasets import mnist

import pickle
import time as time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from Models import Fully_Connected_NN as FCNN
import CustomOptimizers as MyOpts 

tf.random.set_seed(104)

def save_result(obj, name ):
    with open(''+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_result(name ):
    with open('' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def train_step(model, optimizer, x_train, y_train):
  with tf.GradientTape() as tape:
    predictions = model(x_train, training=True)
    loss = loss_object(y_train, predictions)
    loss_pen = loss + tf.add_n(model.losses)
  grads = tape.gradient(loss_pen, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  
  train_loss(loss)
  iteration_loss.append(loss.numpy())
  train_accuracy(y_train, predictions)

def test_step(model, x_test, y_test):
  predictions = model(x_test)
  loss = loss_object(y_test, predictions)

  test_loss(loss)
  test_accuracy(y_test, predictions)

print("Versione tensorfolow:", tf.__version__)

alpha_0 = 0.001

FC_NN = {'SGD' : {'HD'   :{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[alpha_0], 'loss_it':[], 'val_acc':[0.0]},
                    'Keras':{'loss':[], 'val_loss':[],'loss_it' :[], 'val_acc':[0.0]}},
           'SGDN': {'HD'   :{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[alpha_0], 'loss_it':[], 'val_acc':[0.0]},
                    'Keras':{'loss':[], 'val_loss':[],'loss_it' :[], 'val_acc':[0.0]}},
           'Adam': {'HD'   :{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[alpha_0], 'loss_it':[], 'val_acc':[0.0]},
                    'Keras':{'loss':[], 'val_loss':[],'loss_it' :[], 'val_acc':[0.0]}}}

Opts = [['SGD','HD',
         MyOpts.SGD_HD(alpha_0 = alpha_0, beta = 1e-3)],
        ['SGD','Keras',
         keras.optimizers.SGD(lr = alpha_0, momentum=0, nesterov= False)],
        ['SGDN','HD',
         MyOpts.SGDN_HD(alpha_0 = alpha_0,  mu = 0.9, beta = 1e-3)],
         ['SGDN','Keras',
         keras.optimizers.SGD(lr = alpha_0, momentum = 0.9, nesterov= True)],
        ['Adam','HD',
         MyOpts.Adam_HD(alpha_0 = alpha_0, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8,
                        beta= 1e-7)],
        ['Adam','Keras',
         keras.optimizers.Adam(lr = alpha_0, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)]]

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32)
x_test  = x_test.astype(np.float32)

mean =  6./7. * x_train.mean() + 1./7. * x_test.mean()
sd   = (6./7. * x_train.var()  + 1./7. * x_test.var())**0.5  

x_train, x_test = (x_train - mean)/ sd, (x_test - mean) / sd

train_dataset_base = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(128)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model_base = FCNN(input_shape=(28,28), n_class=10, 
                          bias_pen=1e-4, kernel_pen=1e-4)

w_base = model_base.get_weights()

predictions = model_base(x_train, training=False)
start_loss_train = loss_object(y_train, predictions).numpy()
  
predictions = model_base(x_test)
start_loss_val = loss_object(y_test, predictions).numpy()

EPOCHS = 100
First  = True 
iter_for_epoch = 60000//128 + 1 
count = 0 
for Opt in Opts:
    count = count + 1
    algorithm = Opt[0]
    method    = Opt[1]    
    optm      = Opt[2]

    model = FCNN(input_shape=(28,28), n_class=10, 
                          bias_pen=1e-4, kernel_pen=1e-4)
    model.set_weights(w_base)
    
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
    
    FC_NN[algorithm][method]['loss'].append(start_loss_train)
    FC_NN[algorithm][method]['val_loss'].append(start_loss_val)

    train_dataset = train_dataset_base.shuffle(60000, seed=10).batch(128)
    iteration_loss = []
    


    if method == 'HD':
       print('\n'+algorithm +'-HD, alpha_0='+ str(optm._alpha_0)+', beta='+
             str(optm._beta)+' (MNIST).')
       for epoch in range(EPOCHS):
           t0 = time.time() 
           alphas_means = 0
           for (x_train, y_train) in train_dataset:
               train_step(model, optm, x_train, y_train)
               FC_NN[algorithm]['HD']['alpha_it'].append(optm._get_alpha().numpy())
               alphas_means += optm._get_alpha().numpy()
           for (x_test, y_test) in test_dataset:
               test_step(model, x_test, y_test)
           t1 = time.time()
 
           FC_NN[algorithm]['HD']['alpha_epoch'].append(alphas_means/iter_for_epoch)
           FC_NN[algorithm]['HD']['loss'].append(train_loss.result().numpy()*1)
           FC_NN[algorithm]['HD']['val_loss'].append(test_loss.result().numpy()*1)
           FC_NN[algorithm]['HD']['val_acc'].append(test_accuracy.result().numpy()*1)
          
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
    
       FC_NN[algorithm]['HD']['loss_it'] = FC_NN[algorithm]['HD']['loss_it']  + iteration_loss

    else:
       print('\n'+algorithm +', alpha=0.001 (MNIST).')
       for epoch in range(EPOCHS):
           t0 = time.time() 
           for (x_train, y_train) in train_dataset:
               train_step(model, optm, x_train, y_train)
           for (x_test, y_test) in test_dataset:
               test_step(model, x_test, y_test)
           t1 = time.time()
 
           FC_NN[algorithm]['Keras']['loss'].append(train_loss.result().numpy()*1)
           FC_NN[algorithm]['Keras']['val_loss'].append(test_loss.result().numpy()*1)
           FC_NN[algorithm]['Keras']['val_acc'].append(test_accuracy.result().numpy()*1)
          
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
    
       FC_NN[algorithm]['Keras']['loss_it'] = FC_NN[algorithm]['Keras']['loss_it']  + iteration_loss

save_result(FC_NN, 'FC_NN_performances_result')

fig = plt.figure(figsize=(15, 12))

ax = fig.add_subplot(221)


plt.plot([0,EPOCHS],[alpha_0,alpha_0], color ='goldenrod',
         linestyle = "--", label= 'SGD')
plt.plot(FC_NN['SGD']['HD']['alpha_epoch'],
             color='tab:orange', linestyle='-', label = 'SGD-HD')
plt.plot([0,EPOCHS],[alpha_0,alpha_0], color ='palevioletred',
         linestyle = "--", label = 'SGDN')
plt.plot(FC_NN['SGDN']['HD']['alpha_epoch'],
             color='mediumvioletred', linestyle='-', label = 'SGDN-HD')
plt.plot([0,EPOCHS],[alpha_0,alpha_0],
             color='tab:purple', linestyle='--', label = 'Adam')
plt.plot(FC_NN['Adam']['HD']['alpha_epoch'],
             color='indigo', linestyle='-', label = 'Adam-HD')

plt.legend(loc='upper right')
plt.ylabel(r'$\alpha_t$')
plt.xlabel('Epoch')
plt.grid()



ax = fig.add_subplot(222)
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), FC_NN['Adam']['HD']['alpha_it'],
             color='indigo')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), FC_NN['SGDN']['HD']['alpha_it'],
             color='mediumvioletred')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), FC_NN['SGD']['HD']['alpha_it'],
             color='tab:orange')

plt.xlabel('Iteration')
plt.ylabel(r'$\alpha_t$')
plt.xscale('log')
plt.legend(loc='upper right')

plt.plot([1,EPOCHS*iter_for_epoch],[alpha_0,alpha_0], color ='palevioletred',
         linestyle = "--")
plt.grid()

ax = fig.add_subplot(223)
plt.plot(FC_NN['Adam']['HD']['loss'], linestyle='-', color='indigo')
plt.plot(FC_NN['Adam']['Keras']['loss'], linestyle='--', color='tab:purple')
plt.plot(FC_NN['SGDN']['HD']['loss'], linestyle='-', color='mediumvioletred')
plt.plot(FC_NN['SGDN']['Keras']['loss'], linestyle='--', color='palevioletred')
plt.plot(FC_NN['SGD']['HD']['loss'], linestyle='-', color='tab:orange')
plt.plot(FC_NN['SGD']['Keras']['loss'], linestyle='--', color='goldenrod')
plt.ylabel('Train loss')
plt.xlabel('Epoch')
plt.yscale('log')
plt.grid()
inset_axes(ax, width="50%", height="35%", loc=1)

plt.plot(range(1, EPOCHS*iter_for_epoch + 1), FC_NN['SGD']['HD']['loss_it'],
             color='tab:orange')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), FC_NN['SGD']['Keras']['loss_it'],
             color='goldenrod', linestyle='--')

plt.plot(range(1, EPOCHS*iter_for_epoch + 1), FC_NN['SGDN']['HD']['loss_it'],
             color='mediumvioletred')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), FC_NN['SGDN']['Keras']['loss_it'],
             color='palevioletred', linestyle='--')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), FC_NN['Adam']['Keras']['loss_it'],
             color='tab:purple', linestyle = "--")
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), FC_NN['Adam']['HD']['loss_it'],
             color='indigo')
plt.xlabel('Iteration')
plt.ylabel('Train loss')
plt.xscale('log')
plt.grid()

ax = fig.add_subplot(224)

plt.plot(FC_NN['Adam']['HD']['val_loss'],
             color='indigo')
plt.plot(FC_NN['Adam']['Keras']['val_loss'], linestyle = "--",
             color='tab:purple') 
plt.plot(FC_NN['SGDN']['HD']['val_loss'], 
             color='mediumvioletred')
plt.plot(FC_NN['SGDN']['Keras']['val_loss'], linestyle = "--", 
             color='palevioletred')
plt.plot(FC_NN['SGD']['HD']['val_loss'], 
             color='tab:orange')
plt.plot(FC_NN['SGD']['Keras']['val_loss'], linestyle = "--", 
             color='goldenrod')
plt.grid() 
plt.ylabel('Validation loss')
plt.xlabel('Epoch')
plt.yscale('log')
plt.show()