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
import CustomOptimizers as MyOpts 

tf.random.set_seed(104)

def save_result(obj, name ):
    with open(''+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_result(name ):
    with open('' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def batch_evaluate_train_loss(model, x_train, y_train):
  predictions = model(x_train, training=False)
  loss = loss_object(y_train, predictions)

  train_loss(loss)

def batch_evaluate_test_loss(model, x_test, y_test):
  predictions = model(x_test, training=False)
  loss = loss_object(y_test, predictions)

  test_loss(loss)

print("Versione tensorfolow:", tf.__version__)

VGG_Net = {'SGD' : {'HD'   :{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]},
                    'Keras':{'loss':[], 'val_loss':[],'loss_it' :[], 'val_acc':[0.0]}},
           'SGDN': {'HD'   :{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]},
                    'Keras':{'loss':[], 'val_loss':[],'loss_it' :[], 'val_acc':[0.0]}},
           'Adam': {'HD'   :{'loss':[], 'val_loss':[],'alpha_it':[],'alpha_epoch':[], 'loss_it':[], 'val_acc':[0.0]},
                    'Keras':{'loss':[], 'val_loss':[],'loss_it' :[], 'val_acc':[0.0]}}}

save_result(VGG_Net, 'VGG_performances_result_final')

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

model_base = VGG(n_class=10, input_shape = (32,32,3))
model_base.save_weights('base_CNN_weights.h5')

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

for (x_train, y_train) in train_dataset:
  batch_evaluate_train_loss(model_base, x_train, y_train)

for (x_test, y_test) in test_dataset:
  batch_evaluate_test_loss(model_base, x_test, y_test)
 
start_losses = (train_loss.result().numpy()*1,
                test_loss.result().numpy()*1)

train_loss.reset_states()
test_loss.reset_states()

save_result(start_losses,'losses_CNN')