# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import matplotlib
#from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
#import imageio as im
# vamos precisar desta biblioteca para aceder às camadas do modelo
from keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file
from keras import backend as K
# K.set_image_dim_ordering('tf')
K.image_data_format() == 'channels_last'
#from keras.utils import np_utils

#from keras.datasets import mnist

fashion_mnist = keras.datasets.fashion_mnist


def visualize_mnist(X_train):
    #(X_train, y_train), (X_test, y_test) = load_mnist_dataset('mnist.npz')
    plt.subplot(321)
    plt.imshow(X_train[0], cmap='gray')
    plt.subplot(322)
    plt.imshow(X_train[1], cmap='gray')
    plt.subplot(323)
    plt.imshow(X_train[2], cmap='gray')
    plt.subplot(324)
    plt.imshow(X_train[3], cmap='gray')
    plt.subplot(325)
    plt.imshow(X_train[4], cmap='gray')
    plt.subplot(326)
    plt.imshow(X_train[5], cmap='gray')
    plt.show()


def data_preparation():
    #(X_train, y_train), (X_test, y_test) = load_mnist_dataset('mnist.npz')
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print(X_train.shape)
    visualize_mnist(X_train)
    # transformar para o formato [instancias][pixeis][largura][altura]
    print("shape[0]: ", X_train.shape[0])
    print("shape antes: ", X_train.shape)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    print("shape depois: ", X_train.shape)
    # normalizar os valores dos pixeis de 0-255 para 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # transformar o label que é um inteiro em categorias binárias, o valor passa a ser o correspondente à posição
    # o 5 passa a ser a lista [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    num_classes = y_test.shape[1]
    return X_train, X_test, y_train, y_test, num_classes


X_train, X_test, y_train, y_test, num_classes = data_preparation()


def create_model_cnn_plus(num_classes):
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def create_model_cnn_plus_plus(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


model = create_model_cnn_plus_plus(num_classes)

model.summary()

# Compile model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        # saving in Keras HDF5 (or h5), a binary data format
        # path where to save the model
        filepath='./T2/my_model_{epoch}_{val_loss:.3f}.hdf5',
        monitor='val_loss',  # the val_loss score has improved
        verbose=0,  # verbosity mode
        save_best_only=True,  # overwrite the current checkpoint if and only if
        save_weights_only=False)  # if True, only the weights are saved

]

#checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)


#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200, verbose=1, callbacks=[checkpointer])
history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=50, batch_size=150, verbose=1, callbacks=callbacks)


# model.load_weights('best_weights.hdf5')
# model.load_weights('my_model_{epoch}_{val_loss:.3f}.hdf5')
# model.save('mnist_cnn_plus.h5')
# model.save('mnist_cnn_plus_plus.h5')

# para utilizar:
# model=load_model('mnist_cnn_plus_plus.h5')

def print_history_accuracy(history):
    # print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def print_history_loss(history):
    # print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


print_history_accuracy(history)
print_history_loss(history)


def visualize_previsao(x_test, y_test, img_a_mostrar):
    print("X_test:", x_test.shape)
    print("y_test:", y_test.shape)
    print("Imagem:", x_test[img_a_mostrar].shape)
    plt.imshow(x_test[img_a_mostrar, :, :, 0], cmap='gray')
    plt.show()
    print("label:", y_test[img_a_mostrar])
    print("antes do aumento de mais uma dimensão:",
          x_test[img_a_mostrar].shape)
    imagem_tensor = np.expand_dims(x_test[img_a_mostrar], axis=0)
    print("depois do aumento de mais uma dimensão:", imagem_tensor.shape)
    print("previsão:", model.predict(imagem_tensor))
    classes = model.predict_classes(imagem_tensor)
    print('Classe prevista:', classes)
    return imagem_tensor


imagem_tensor = visualize_previsao(X_test, y_test, 3)

# Vamos buscar os outputs das primeiras 5 (plus) camadas da rede ou 12 para a plus_plus
camadas_outputs = []
for layer in model.layers[:12]:
    print(layer.output.shape)
    camadas_outputs.append(layer.output)

activation_model = tf.keras.models.Model(
    inputs=model.input, outputs=camadas_outputs)
# e depois criamos um modelo que retorna estes outputs dado os inputs do modelo

activations = activation_model.predict(imagem_tensor)


first_layer_activation = activations[0]
print(first_layer_activation.shape)


plt.imshow(first_layer_activation[0, :, :, 9], cmap='viridis')
plt.show()
'''
activation_model = models.Model(inputs=model.input, outputs=camadas_outputs) 


first_layer_activation = activations[0]
print(first_layer_activation.shape)
'''
nomes_camadas = []
for camada in model.layers[:12]:
    # para puder colocar o nome da cada camada nas visualizações
    nomes_camadas.append(camada.name)

imagens_por_linha = 16

# o zip permite iterar simultaneamente em 2 listas
for nome_camada, ativacao_camada in zip(nomes_camadas, activations):
    # Numero de features no feature map, pois é o que está na ultima dimensão
    n_features = ativacao_camada.shape[-1]
    # O feature map tem shape (1, tamanho, tamanho, numero_features).
    size = ativacao_camada.shape[1]
    # Empilha os canais de ativação nesta matriz
    n_linhas = -(-n_features // imagens_por_linha)
    print("nome_camada:", nome_camada)
    print("n_features:", n_features)
    print("size:", size)
    print("n_linhas:", n_linhas)
    display_grid = np.zeros((size * n_linhas, imagens_por_linha * size))
    for col in range(n_linhas):  # para fazer o display com 15 imagens por linha
        for lin in range(imagens_por_linha):
            # verificar aqui se a imagem existe
            # isto pode dar erro de out-of-range
            imagem = ativacao_camada[0, :, :, col * imagens_por_linha + lin]
            imagem -= imagem.mean()  # pos-processamento para melhor visualização
            imagem /= imagem.std()
            imagem *= 64
            imagem += 128
            # valores <0 ficam 0 e >255 ficam = 255
            imagem = np.clip(imagem, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
                         lin * size: (lin + 1) * size] = imagem
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(nome_camada)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
