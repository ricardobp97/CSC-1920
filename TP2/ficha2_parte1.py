import tensorflow as tf

def func1():
    a = tf.random.uniform(shape=[])
    b = tf.random.uniform(shape=[])
    if a > b:
        return a+b
    else:
        return a-b

def func2():
    a = tf.random.uniform(shape=[], minval=-1., maxval=1.)
    b = tf.random.uniform(shape=[], minval=-1., maxval=1.)
    if a < b:
        return a+b
    else:
        return a-b

def func3():
    return tf.math.equal(tf.Variable([[1, 2, 0], [3, 0, 2]]), tf.Variable([[0, 0, 0], [0, 0, 0]]))

def func4():
    a = tf.random.uniform(shape=[20], minval=1., maxval=10.)
    return tf.gather(a,tf.where(tf.greater(a,tf.constant(7,dtype=tf.float32))))

print('---- Função 1 ----')
print(func1())
print('---- Função 2 ----')
print(func2())
print('---- Função 3 ----')
print(func3())
print('---- Função 4 ----')
print(func4())