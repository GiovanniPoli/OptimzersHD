import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization



def Softmax_Model(input_shape, n_class, kernel_pen= 0.0, bias_pen= 0.0):
    iniz = tf.keras.initializers.VarianceScaling(
    scale=1/3, mode="fan_in", distribution="uniform")
    return keras.models.Sequential([keras.layers.Flatten(input_shape=input_shape),
             keras.layers.Dense(n_class, activation='linear',
                                bias_initializer=iniz,
                                kernel_initializer=iniz,
                                bias_regularizer=regularizers.l2(bias_pen),
                                kernel_regularizer=regularizers.l2(kernel_pen))])

def Fully_Connected_NN(input_shape, n_class, hidden_dim=1000, kernel_pen=0.0, bias_pen=0.0):
    iniz = tf.keras.initializers.VarianceScaling(
    scale=1/3, mode="fan_in", distribution="uniform")
    
    return keras.models.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(hidden_dim, activation="relu",
                       bias_regularizer = regularizers.l2(bias_pen),
                       kernel_regularizer = regularizers.l2(kernel_pen),
                       bias_initializer=iniz,
                       kernel_initializer=iniz),  
    keras.layers.Dense(hidden_dim, activation="relu",  
                       bias_regularizer = regularizers.l2(bias_pen),
                       kernel_regularizer=regularizers.l2(kernel_pen),
                       bias_initializer=iniz,
                       kernel_initializer=iniz),
    keras.layers.Dense(n_class, activation="linear",
                       bias_regularizer = regularizers.l2(bias_pen),
                       kernel_regularizer=regularizers.l2(kernel_pen),
                       bias_initializer=iniz,
                       kernel_initializer=iniz)])
  
    
def VGG_Net(n_class,input_shape, kernel_pen=0, bias_pen = 0):
        iniz_dens = tf.keras.initializers.VarianceScaling(
            scale=1/3, mode="fan_in", distribution="uniform")
        iniz_conv = tf.keras.initializers.VarianceScaling(
               scale=2./9., mode="fan_in", distribution="normal")

        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(64, (3, 3), 
                                      input_shape = input_shape,
                                      padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Conv2D(64, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(128, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(128, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(256, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(256, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(256, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512,
                                      bias_initializer =   iniz_dens,
                                      kernel_initializer = iniz_dens,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(n_class))
        model.add(keras.layers.Activation('linear'))
        
        return model
    
    
def VGG_Net2(n_class,input_shape, kernel_pen=0, bias_pen = 0):
        iniz_dens = tf.keras.initializers.VarianceScaling(
            scale=1/3, mode="fan_in", distribution="uniform")
        iniz_conv = tf.keras.initializers.VarianceScaling(
               scale=2./9., mode="fan_out", distribution="normal")

        model = tf.keras.Sequential([])

        model.add(keras.layers.Conv2D(64, (3, 3), 
                                      input_shape = input_shape,
                                      padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Conv2D(64, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(128, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(128, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(256, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(256, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(256, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same',
                                      bias_initializer = keras.initializers.Zeros(),
                                      kernel_initializer = iniz_conv,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512,
                                      bias_initializer =   iniz_dens,
                                      kernel_initializer = iniz_dens,
                                      bias_regularizer = regularizers.l2(bias_pen),
                                      kernel_regularizer=regularizers.l2(kernel_pen)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(n_class))
        model.add(keras.layers.Activation('linear'))
        
        return model