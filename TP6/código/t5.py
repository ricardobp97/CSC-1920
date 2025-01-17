#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
#from keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import Conv2D
#from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#K.set_image_dim_ordering('th') #pode ser 'th' ou 'tf'
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt

# fixar random seed para se puder reproduzir os resultados
seed = tf.random.set_seed(91195003)

# Etapa 1 - preparar o dataset
import os
import pickle
from keras.utils.data_utils import get_file

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding='bytes') 
    d_decoded = {} # decode utf8 for k, v in d.items():
    d_decoded[k.decode('utf8')] = v 
    d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]
    data = data.reshape(data.shape[0], 3, 32, 32) 
    return data, labels

def load_cfar10_dataset():
    #path = downloadDataset("./", "CINIC-10.tar.gz", "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz")
    #dirname = 'cifar-10-batches-py'
    #origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file("./CINIC-10.tar.gz", untar=True)
    num_train_samples = 50000
    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8') 
    y_train = np.zeros((num_train_samples,), dtype='uint8')
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i)) 
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data 
        y_train[(i - 1) * 10000: i * 10000] = labels
        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = load_batch(fpath)
        y_train = np.reshape(y_train, (len(y_train), 1)) 
        y_test = np.reshape(y_test, (len(y_test), 1))
        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1) 
        return (x_train, y_train), (x_test, y_test)

#util para visualizar a topologia da rede num ficheiro em pdf ou png
def print_model(model,fich):
    tf.keras.utils.plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)

#utils para visulaização do historial de aprendizagem
def print_history_accuracy(history):
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Etapa 2 - Definir a topologia da rede (arquitectura do modelo) e compilar
def create_compile_model_cnn_cifar10_simples(num_classes,epochs):
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', data_format='channels_last', activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding = 'same', data_format='channels_last', activation='relu', kernel_constraint=maxnorm(3))) 
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # Compile model
    lrate = 0.01
    decay = lrate/epochs
    sgd = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# Etapa 2 - Definir a topologia da rede (arquitectura do modelo) e compilar
def create_compile_model_cnn_cifar10_plus(num_classes,epochs):
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32),  padding = 'same', data_format='channels_last', activation='relu')) 
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding = 'same', data_format='channels_last', activation='relu')) 
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), padding = 'same', data_format='channels_last', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding = 'same', data_format='channels_last', activation='relu')) 
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), padding = 'same', data_format='channels_last', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Conv2D(128, (3, 3), padding = 'same', data_format='channels_last', activation='relu')) 
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_constraint=maxnorm(3))) 
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_constraint=maxnorm(3))) 
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # Compile model
    lrate = 0.01
    decay = lrate/epochs
    sgd = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    return model

def cfar10_utilizando_cnn_simples():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #(X_train, y_train), (X_test, y_test) = load_cfar10_dataset()
    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32).astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    # transformar o label que é um inteiro em categorias binárias, o valor passa a ser o correspondente à posição
    # a classe 5 passa a ser a lista [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    epochs = 5 #25
    model = create_compile_model_cnn_cifar10_simples(num_classes,epochs) 
    print(model.summary())
    print_model(model,"cifar10_simples.png")
    # treino do modelo: epochs=5, batch size = 32
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                      batch_size=32, verbose=2) 
    print_history_accuracy(history) 
    #print_history_loss(history)
    # Avaliação final com os casos de teste
    scores = model.evaluate(X_test, y_test, verbose=0) 
    print('Scores: ', scores)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("Erro modelo CNN cifar10 simples: %.2f%%" % (100-scores[1]*100))

def cfar10_utilizando_cnn_plus():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #(X_train, y_train), (X_test, y_test) = load_cfar10_dataset()
    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32).astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    # transformar o label que é um inteiro em categorias binárias, o valor passa a ser o correspondente à posição
    # a classe 5 passa a ser a lista [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    epochs = 5
    model = create_compile_model_cnn_cifar10_plus(num_classes,epochs) 
    print(model.summary())
    print_model(model,"model.png")
    # treino do modelo: epochs=5, batch size = 64
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                      batch_size=64, verbose=2) 
    print_history_accuracy(history) 
    #print_history_loss(history)
    # Avaliação final com os casos de teste
    scores = model.evaluate(X_test, y_test, verbose=0) 
    print('Scores: ', scores)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("Erro no modelo: %.2f%%" % (100-scores[1]*100))

if __name__ == '__main__':
    cfar10_utilizando_cnn_plus()
