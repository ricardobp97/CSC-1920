from random import randint
import tensorflow as tf
import numpy as np

def generate_sequence(length, nr_features):
    return [randint(0,nr_features-1) for _ in range(length)]

def one_hot_encode(sequence, nr_features):
    encoded = list()
    for value in sequence:
        one_hot_encoded = np.zeros(nr_features)
        one_hot_encoded[value] = 1
        encoded.append(one_hot_encoded)
    return np.array(encoded)

def one_hot_decode(encoded_seq):
    return [np.argmax(value) for value in encoded_seq]

def generate_sample(length, nr_features, out_index):
    sequence = generate_sequence(length, nr_features)
    encoded = one_hot_encode(sequence, nr_features)
    X = encoded.reshape((1, length, nr_features))
    y = encoded[out_index].reshape(1, nr_features)
    return X, y

timesteps = 10
features = 25
out_index = 2
epochs = 100

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(timesteps, features)))
model.add(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.5))
model.add(tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.5))
model.add(tf.keras.layers.Dense(features, activation = 'tanh'))
model.compile(
    loss= tf.keras.losses.categorical_crossentropy,
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print(model.summary())

for i in range(epochs):
    X, y = generate_sample(timesteps, features, out_index)
    history = model.fit(X, y, shuffle=False, epochs=1, verbose=0)
    print('Epoch: %d; Loss: %.2f; Accuracy: %.2f' %(i,
history.history['loss'][0], history.history['accuracy'][0]))

correct = 0
for i in range(100):
    X, y = generate_sample(timesteps, features, out_index)
    yhat = model.predict(X)
    if one_hot_decode(yhat) == one_hot_decode(y):
        correct += 1
print('Accuracy: %.2f' %((correct/100)*100.0))

X, y = generate_sample(timesteps, features, out_index)
yhat = model.predict(X)

print('Sequence: %s' % [one_hot_decode(x) for x in X])
print('Expected: %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))