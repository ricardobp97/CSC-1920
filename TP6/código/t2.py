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
'''
fazer o download do MNIST dataset com imagens de digitos escritos à mão para fazer a sua classificação (já pré-preparados)
dataset: https://s3.amazonaws.com/img-datasets/mnist.npz
O ficheiro já tem tudo separado nos ficheiros {x_test.npy, x_train.npy, y_test.npy, y_train.npy}
Os atributos de entrada estão com matrizes 3D(imagem, largura,altura) e os atributos de saída é uma lista com o número correspondente
'''
#def load_mnist_dataset(path='mnist.npz'):
def load_mnist_dataset():
    #path = get_file(path, origin='https://s3.amazonaws.com/img-datasets/mnist.npz')
    '''
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()
    '''
    #load mnist training and test data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
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
''' 
criar uma rede neuronal convolucionária simples.
- a primeira camada escondida é uma camada convolucionária chamada Convolution2D.
A camada tem 32 feature maps , cada um de dimensão 5×5 e uma função de activação 'rectifier activation function'.
trata-se de uma camada de input, à espera de imagens com a estrutura [pixels][width][height].
- A segunda camada é de pooling que utiliza o max de MaxPooling2D.
está configurado para uma pool size de 2×2.
- A camada seguinte é de regularização que usa Dropout.
Está configurado para excluir aleatóriamente 20% dos neuronios na camada para reduzir overfitting.
- A camada seguinte converte os dados da matriz 2D num vector chamado Flatten.
Assim permite-se que esse output seja tratado por uma camada completamente ligada standard.
- A camada seguinte é uma completamente ligada com 128 neuronios e uma fução de activação 'rectifier activation function'.
- Finalmente a camada de saida tem 10 neuronios correspondentes às 10 classes e uma função de activação softmax
para apresentar na saida uma especie de probabilidade para cada classe.
- O modelo é treinado utilizando logaritmic loss e o algoritmo de gradient descent ADAM.
'''
def create_compile_model_cnn_plus(num_classes):
    model = tf.keras.Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), padding = 'same', data_format='channels_last', activation='relu')) 
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(15, (3, 3), padding = 'same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model


def mnist_utilizando_cnn_simples():
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
    model = create_compile_model_cnn_simples(num_classes) 
    print_model(model,"model_simples.png")
    # treinar a rede
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10,
                        batch_size=200, verbose=2) 
    print_history_accuracy(history) 
    #print_history_loss(history)
    # Avaliação final com os casos de teste
    scores = model.evaluate(X_test, y_test, verbose=0) 
    print('Scores: ', scores)
    print("Erro modelo MLP: %.2f%%" % (100-scores[1]*100))

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
    print_model(model,"model_plus.png")
    # treinar a rede
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200,
                        verbose=2)
    print_history_accuracy(history) 
    #print_history_loss(history)
    # Avaliação final com os casos de teste
    scores = model.evaluate(X_test, y_test, verbose=0) 
    print('Scores: ', scores)
    print("Erro modelo MLP: %.2f%%" % (100-scores[1]*100))

if __name__ == '__main__': 
    #mnist_utilizando_cnn_simples() 
    mnist_utilizando_cnn_plus()
