#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
#from keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import Conv2D
#from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K 
import matplotlib.pyplot as plt

#K.set_image_dim_ordering('th') #pode ser 'th' ou 'tf' import matplotlib.pyplot as plt
K.set_image_data_format('channels_last')

# fixar random seed para se puder reproduzir os resultados
seed = tf.random.set_seed(91195003)

# Etapa 1 - preparar o dataset
def load_mnist_dataset():
    #load mnist training and test data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
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
def create_compile_model_cnn_plus(num_classes):
    model = tf.keras.Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), padding = 'same', data_format='channels_last', activation='relu')) 
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(15, (3, 3), padding = 'same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model

def mnist_utilizando_cnn_plus():
    #(X_train, y_train), (X_test, y_test) = load_mnist_dataset('mnist.npz')
    (X_train, y_train), (X_test, y_test) = load_mnist_dataset()
    # transformar para o formato [instancias][pixeis][largura][altura]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    # normalizar os valores dos pixeis de 0-255 para 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # transformar o label que é um inteiro em categorias binárias, o valor passa a ser o correspondente à posição
    # o 5 passa a ser a lista [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    # definir a topologia da rede e compilar
    model = create_compile_model_cnn_plus(num_classes)
    print_model(model,"model_t6_2.png")
    # treinar a rede
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=200,
                        verbose=2)
    print_history_accuracy(history) 
    #print_history_loss(history)
    # Avaliação final com os casos de teste
    scores = model.evaluate(X_test, y_test, verbose=0) 
    print('Scores: ', scores)
    print("Erro modelo MLP: %.2f%%" % (100-scores[1]*100))

if __name__ == '__main__':
    mnist_utilizando_cnn_plus()
