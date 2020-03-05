import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#tensorflow version being used
print('Version: ' + tf.__version__)
#is tf executing eagerly?
print('Executing eagerly: ' + str(tf.executing_eagerly()))

#load mnist training and test data
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

#data shape and cardinality
print('Train data shape: ' + str(x_train.shape))
print('Test data shape: ' + str(x_test.shape))
print('Number of training samples: ' + str(x_train.shape[0]))
print('Number of testing samples: ' + str(x_test.shape[0]))

#plotting some numbers!
for i in range(25):
    plt.subplot(5,5,i+1)    #Add a subplot as 5 x 5
    plt.xticks([])          #get rid of labels
    plt.yticks([])          #get rid of labels
    plt.imshow(x_test[i],cmap='gray')
plt.show()

#reshape the input to have a list of 784 (28*28) and normalize it (/255)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_train = x_train.astype('float')/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_test = x_test.astype('float32')/255

#building a three-layer sequential model
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
#compiling the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#training it
model.fit(x_train, y_train,
            batch_size=32,
            epochs=20,
            verbose=1,
            validation_data=(x_test, y_test))
#evaluating it
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('\nTest accuracy:', test_acc)

#finally, generating predictions (the output of the last layer)
print('\nGenerating predictions for the first fifteen samples...')
predictions = model.predict(x_test[:15], batch_size=128, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
print('Predictions shape:', predictions.shape)
for i, prediction in enumerate(predictions):
    #tf.argmax returns the INDEX with the largest value across axes of a tensor
    predicted_value = tf.argmax(prediction)
    label = y_test[i]
    print('Predicted a %d. Real value is %d.' %(predicted_value, label))