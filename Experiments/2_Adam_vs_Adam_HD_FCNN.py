import tensorflow as tf
import numpy as np
from tensorflow import keras

from keras.datasets import mnist

import pickle
import time as time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from Models import Fully_Connected_NN
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

model_base = Fully_Connected_NN(input_shape=(28,28), n_class=10, 
                          bias_pen=1e-4, kernel_pen=1e-4)
w_base = model_base.get_weights()

predictions = model_base(x_train, training=False)
start_loss_train = loss_object(y_train, predictions).numpy()
  
predictions = model_base(x_test)
start_loss_val = loss_object(y_test, predictions).numpy()

AdamFC ={'0.1'     : {'HD':{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]},
                       'Keras': {'loss':[], 'val_loss':[], 'loss_it':[], 'val_acc':[0.0]},
                       'HD Multiplicative':{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]}},
        '0.01'    : {'HD':{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]},
                       'Keras': {'loss':[], 'val_loss':[], 'loss_it':[], 'val_acc':[0.0]}},
        '0.001'   : {'HD':{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]},
                       'Keras': {'loss':[], 'val_loss':[], 'loss_it':[], 'val_acc':[0.0]}},
        '0.0001'  : {'HD':{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]},
                       'Keras': {'loss':[], 'val_loss':[], 'loss_it':[], 'val_acc':[0.0]}},
        '0.00001' : {'HD':{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]},
                       'Keras': {'loss':[], 'val_loss':[], 'loss_it':[], 'val_acc':[0.0]}},
        '0.000001': {'HD':{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]},        
                       'Keras': {'loss':[], 'val_loss':[], 'loss_it':[], 'val_acc':[0.0]}}}


EPOCHS = 50
First  = True 
iter_for_epoch = 60000//128 + 1 

for i in ('0.1','0.01','0.001','0.0001','0.00001','0.000001'):
    alpha_test = float(i)
    
    optm_HD = MyOpts.Adam_HD(alpha_0 = alpha_test,
                             beta_1 = 0.9, beta_2 = 0.999,
                             beta = 1e-7, epsilon = 1e-8)
    optm_keras = keras.optimizers.Adam(lr = alpha_test, beta_1 = 0.9, beta_2=0.999,
                                       epsilon = 1e-8)
            
    model = Fully_Connected_NN(input_shape=(28,28), n_class=10, 
                          bias_pen=1e-4, kernel_pen=1e-4)
    model.set_weights(w_base)
    AdamFC[i]['HD']['alpha_epoch'].append(float(i))
    AdamFC[i]['HD']['loss'].append(start_loss_train)
    AdamFC[i]['HD']['val_loss'].append(start_loss_val)
    
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    train_dataset = train_dataset_base.shuffle(60000, seed=10).batch(128)
    iteration_loss = []
    
    print('\nAdam-HD, alpha_0='+i+', beta= 10^-7 (MNIST).')
    for epoch in range(EPOCHS):
      t0 = time.time() 
      alphas_means = 0
      for (x_train, y_train) in train_dataset:
        train_step(model, optm_HD, x_train, y_train)
        AdamFC[i]['HD']['alpha_it'].append(optm_HD._get_alpha().numpy())
        alphas_means += optm_HD._get_alpha().numpy()
    
      for (x_test, y_test) in test_dataset:
        test_step(model, x_test, y_test)
      t1 = time.time()
      
      AdamFC[i]['HD']['alpha_epoch'].append(alphas_means/iter_for_epoch)
      AdamFC[i]['HD']['loss'].append(train_loss.result().numpy()*1)
      AdamFC[i]['HD']['val_loss'].append(test_loss.result().numpy()*1)
      AdamFC[i]['HD']['val_acc'].append(test_accuracy.result().numpy()*1)
      
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
      
    AdamFC[i]['HD']['loss_it'] = AdamFC[i]['HD']['loss_it']  + iteration_loss
    
    AdamFC[i]['Keras']['loss'].append(start_loss_train)
    AdamFC[i]['Keras']['val_loss'].append(start_loss_val)
    
    model = Fully_Connected_NN(input_shape=(28,28), n_class=10, 
                          bias_pen=1e-4, kernel_pen=1e-4)
    
    model.set_weights(w_base)
    
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    train_dataset = train_dataset_base.shuffle(60000, seed=10).batch(128)
    iteration_loss = []
    print('\nAdam, alpha='+i+', (MNIST).')
    for epoch in range(EPOCHS):
      t0 = time.time() 
      alphas_means = 0
      for (x_train, y_train) in train_dataset:
        train_step(model, optm_keras, x_train, y_train)
    
      for (x_test, y_test) in test_dataset:
        test_step(model, x_test, y_test)
      t1 = time.time()
      
      AdamFC[i]['Keras']['loss'].append(train_loss.result().numpy()*1)
      AdamFC[i]['Keras']['val_loss'].append(test_loss.result().numpy()*1)
      AdamFC[i]['Keras']['val_acc'].append(test_accuracy.result().numpy()*1)
      
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
    
    AdamFC[i]['Keras']['loss_it'] = AdamFC[i]['Keras']['loss_it']  + iteration_loss
 
## Extra moltiplicative

AdamFC['0.1']['HD Multiplicative']={'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]}

optm_HD = MyOpts.Adam_HD_Mult_v2(alpha_0 = 0.001,
                               beta_1 = 0.9, beta_2=0.999, epsilon = 1e-8,
                               beta= 0.02)
            
model = Fully_Connected_NN(input_shape=(28,28), n_class=10, 
                          bias_pen=1e-4, kernel_pen=1e-4)
model.set_weights(w_base)

AdamFC['0.1']['HD Multiplicative']['alpha_epoch'].append(0.001)
AdamFC['0.1']['HD Multiplicative']['loss'].append(start_loss_train)
AdamFC['0.1']['HD Multiplicative']['val_loss'].append(start_loss_val)

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

train_dataset = train_dataset_base.shuffle(60000, seed=10).batch(128)
iteration_loss = []    

print('\nAdam-HD (Multiplicative), alpha_0=0.1, beta= 0.02 (MNIST).')
for epoch in range(EPOCHS):
      t0 = time.time() 
      alphas_means = 0
      for (x_train, y_train) in train_dataset:
        train_step(model, optm_HD, x_train, y_train)
        AdamFC['0.1']['HD Multiplicative']['alpha_it'].append(optm_HD._get_alpha().numpy())
        alphas_means += optm_HD._get_alpha().numpy()
    
      for (x_test, y_test) in test_dataset:
        test_step(model, x_test, y_test)
      t1 = time.time()
      
      AdamFC['0.1']['HD Multiplicative']['alpha_epoch'].append(alphas_means/iter_for_epoch)
      AdamFC['0.1']['HD Multiplicative']['loss'].append(train_loss.result().numpy()*1)
      AdamFC['0.1']['HD Multiplicative']['val_loss'].append(test_loss.result().numpy()*1)
      AdamFC['0.1']['HD Multiplicative']['val_acc'].append(test_accuracy.result().numpy()*1)
      
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

AdamFC['0.1']['HD Multiplicative']['loss_it'] = AdamFC['0.1']['HD Multiplicative']['loss_it']  + iteration_loss

save_result(AdamFC, 'Output_Alpha_impact_AdamvsAdam_HD_FCNN')


labsHD = [r'Adam HD $\alpha_0=10^{-1}$',r'Adam HD $\alpha_0=10^{-2}$',
        r'Adam HD $\alpha_0=10^{-3}$',r'Adam HD $\alpha_0=10^{-4}$',
        r'Adam HD $\alpha_0=10^{-5}$',r'Adam HD $\alpha_0=10^{-6}$']

labs = [r'Adam        $\alpha=10^{-1}$',r'Adam        $\alpha=10^{-2}$',
        r'Adam        $\alpha=10^{-3}$',r'Adam        $\alpha=10^{-4}$',
        r'Adam        $\alpha=10^{-5}$',r'Adam        $\alpha=10^{-6}$']

cols = ['indigo','tab:purple','tab:blue','tab:green','tab:orange','tab:red']
lcols = ['darkviolet','mediumorchid','lightskyblue','limegreen','gold','indianred'] 

fig = plt.figure(figsize=(15, 12))

ax = fig.add_subplot(221)

plt.plot(AdamFC['0.1']['HD Multiplicative']['alpha_epoch'],
             color='black', label=r'AdamM HD $\alpha_0=10^{-3}$') 
count = 0
for i in list(AdamFC.keys()):
    plt.plot(AdamFC[i]['HD']['alpha_epoch'],
             color=cols[count], linestyle='-',  label=labsHD[count])
    plt.plot([0,EPOCHS],[float(i),float(i)],
              color=lcols[count], linestyle='--', label=labs[count])
    count += 1   
plt.legend(loc='upper right')
plt.ylabel(r'$\alpha_t$')
plt.xlabel('Epoch')
#plt.tick_params(labeltop=False, labelbottom=False, bottom=False, top=False, labelright=False)
plt.grid()



ax = fig.add_subplot(222)
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), AdamFC['0.1']['HD Multiplicative']['alpha_it'],
             color='black', label=r'AdamM HD $\alpha_0=10^{-1}$')
count = 0
for i in list(AdamFC.keys()):
    plt.plot(range(1, EPOCHS*iter_for_epoch + 1), AdamFC[i]['HD']['alpha_it'],
             color=cols[count], linestyle='-',  label=labsHD[count])
    plt.plot([1, EPOCHS*iter_for_epoch], [float(i),float(i)],
             color=lcols[count], linestyle='--',  label=labs[count])
    count += 1
plt.xlabel('Iteration')
plt.ylabel(r'$\alpha_t$')
plt.xscale('log')
#plt.legend(loc='upper right')
plt.grid()
ax = fig.add_subplot(223)
plt.plot(AdamFC['0.1']['HD Multiplicative']['loss'],
             color='black', label=r'AdamM HD $\alpha_0=10^{-1}$')
plt.yscale('log')
count = 0
for i in list(AdamFC.keys()):
    plt.plot(AdamFC[i]['HD']['loss'],
             color=cols[count], linestyle='-',  label=labsHD[count])
    plt.plot(AdamFC[i]['Keras']['loss'],
             color=lcols[count], linestyle='--',  label=labs[count])
    count += 1
plt.yscale('log')
plt.ylabel('Train loss')
plt.xlabel('Epoch')
plt.grid()
inset_axes(ax, width="50%", height="35%", loc=1)
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), AdamFC['0.1']['HD Multiplicative']['loss_it'],
             color='black', label=r'AdamM HD $\alpha_0=10^{-1}$')
count = 0
for i in list(AdamFC.keys()):
    plt.plot(range(1, EPOCHS*iter_for_epoch + 1), AdamFC[i]['HD']['loss_it'],
             color=cols[count], linestyle='-',  label=labsHD[count])
    plt.plot(range(1, EPOCHS*iter_for_epoch + 1), AdamFC[i]['Keras']['loss_it'],
             color=lcols[count], linestyle='--',  label=labs[count])
    count += 1
plt.xlabel('Iteration')
plt.ylabel('Train loss')
plt.xscale('log')
plt.grid()

ax = fig.add_subplot(224)

plt.plot(AdamFC['0.1']['HD Multiplicative']['val_loss'],
             color='black', label=r'AdamM HD $\alpha_0=10^{-3}$') 
count = 0
for i in list(AdamFC.keys()):
    plt.plot(AdamFC[i]['HD']['val_loss'],
             color=cols[count], linestyle='-',  label=labsHD[count])
    plt.plot(AdamFC[i]['Keras']['val_loss'],
             color=lcols[count], linestyle='--',  label=labs[count])
    count += 1
plt.yscale('log')
plt.ylabel('Validation loss')
plt.xlabel('Epoch')
plt.grid()


plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.show()