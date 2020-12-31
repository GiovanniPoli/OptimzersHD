import tensorflow as tf
import numpy as np
from tensorflow import keras

from keras.datasets import cifar10

import pickle
import time as time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from Models import VGG_Net as VGG

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

(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32)
x_test  = x_test.astype(np.float32)

mean =  5./6. * x_train.mean(axis=(0,1,2)) + 1./6. * x_test.mean(axis=(0,1,2))
sd   = (5./6. * x_train.var(axis=(0,1,2))  + 1./6. * x_test.var(axis=(0,1,2)))**0.5  

train_dataset_base = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset_base.shuffle(50000, seed=3).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(128)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = VGG(n_class=10, input_shape = (32,32,3), kernel_pen = 1e-4,
                                                 bias_pen = 1e-4)
model.load_weights('base_CNN_weights.h5')

start_loss_train, start_loss_val = load_result('losses_CNN') 
VGG_Net = load_result('VGG_performances_result_final')

optm = keras.optimizers.SGD(lr = 0.001, momentum=0.9, nesterov=True)
            

VGG_Net['SGDN']['Keras']['loss'].append(start_loss_train)
VGG_Net['SGDN']['Keras']['val_loss'].append(start_loss_val)

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

iteration_loss = []    

EPOCHS = 150
iter_for_epoch = 50000//128 + 1 

print('\nSGDN, alpha_0=0.001, (CIFAR10).')
for epoch in range(EPOCHS):
      t0 = time.time() 
      for (x_train, y_train) in train_dataset:
        train_step(model, optm, x_train, y_train)

      print("testing...")
      for (x_test, y_test) in test_dataset:
        test_step(model, x_test, y_test)
      t1 = time.time()

      VGG_Net['SGDN']['Keras']['loss'].append(train_loss.result().numpy()*1)
      VGG_Net['SGDN']['Keras']['val_loss'].append(test_loss.result().numpy()*1)
      VGG_Net['SGDN']['Keras']['val_acc'].append(test_accuracy.result().numpy()*1)
      
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

VGG_Net['SGDN']['Keras']['loss_it'] = VGG_Net['SGDN']['Keras']['loss_it']  + iteration_loss

save_result(VGG_Net,"VGG_performances_result_final")
model.save_weights('SGDN_final.h5')

fig = plt.figure(figsize=(15, 12))

ax = fig.add_subplot(221)


plt.plot([0, EPOCHS + 1], [0.001,0.001],
             color='tab:purple', linestyle='-')

plt.ylabel(r'$\alpha_t$')
plt.xlabel('Epoch')
plt.grid()



ax = fig.add_subplot(222)
plt.plot([1, EPOCHS*iter_for_epoch + 1], [0.001,0.001],
             color='tab:purple')
plt.xlabel('Iteration')
plt.ylabel(r'$\alpha_t$')
plt.xscale('log')
#plt.legend(loc='upper right')
plt.grid()

ax = fig.add_subplot(223)
plt.plot(VGG_Net['SGDN']['Keras']['loss'], linestyle='-', color='tab:purple')
plt.xlabel('Epoch')
plt.yscale('log')
plt.grid()
inset_axes(ax, width="50%", height="35%", loc=1)
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), VGG_Net['SGDN']['Keras']['loss_it'],
             color='tab:purple')
plt.xlabel('Iteration')
plt.ylabel('Train loss')
plt.xscale('log')
plt.grid()

ax = fig.add_subplot(224)

plt.plot(VGG_Net['SGDN']['Keras']['val_loss'],
             color='tab:purple', label=r'SGDM HD $\alpha_0=10^{-1}$') 
plt.yscale('log')
plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.show()