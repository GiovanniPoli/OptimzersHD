import tensorflow as tf
import numpy as np

import ExampleOptimizer as MyOpts 

from Models import Softmax_Model
from sklearn.datasets import load_iris

tf.random.set_seed(0)

def train_step_tf(model, optimizer, x_train, y_train): # TensorFlow
  with tf.GradientTape() as tape:
    predictions = model(x_train, training=True)
    loss = loss_object_tf(y_train, predictions)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  
  train_loss_tf(loss)
  train_accuracy_tf(y_train, predictions)
   

iris_data = load_iris() 

learning_rate = 1e-1
beta= 0.0001

x_tf = iris_data.data
y_tf = iris_data.target.reshape(-1, 1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_tf, y_tf))
train_dataset = train_dataset.shuffle(150).batch(150)

model_keras = Softmax_Model(input_shape=(4,), n_class=3)

loss_object_tf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer_tf = MyOpts.Adam_Test(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                              epsilon=1e-8)

train_loss_tf = tf.keras.metrics.Mean('train_loss', dtype = tf.float32)
train_accuracy_tf = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

## Check Condizioni Iniziali

n_iterations = 1000
print_int = 10

for i in range(1, n_iterations + 1):  
    for (x_train_tf, y_train_tf) in train_dataset:  
        with tf.GradientTape() as tape:
            predictions_tf = model_keras(x_train_tf, training=True)
            loss_tf_cycle = loss_object_tf(y_train_tf, predictions_tf)
        grads_tf = tape.gradient(loss_tf_cycle, model_keras.trainable_variables)
        optimizer_tf.apply_gradients(zip(grads_tf, model_keras.trainable_variables))

        train_loss_tf(loss_tf_cycle)
        train_accuracy_tf(y_train_tf, predictions_tf)
        if(i%print_int==0.0):
            print("TensorFlow Loss at iteration {}: {}".format(i, train_loss_tf.result().numpy()))

    # Reset metrics 
    train_loss_tf.reset_states()
    train_accuracy_tf.reset_states()
    
print(optimizer_tf._get_iter_tester())
print(optimizer_tf._get_ops_tester())
