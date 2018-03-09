import tensorflow as tf
import numpy as np
import pandas as pd


def standardization(df):
    return (df - df.mean()) / (df.std())


def normalization(df):
    return (df - df.min()) / (df.max() - df.min())


def import_hypothesis(file_location):
    df = pd.read_csv(file_location)
    y = df['OS']
    x = df.drop(['OS'], axis=1)
    y[df['OS'] <= 300] = 0
    y[(df['OS'] > 300) * (df['OS'] <= 450)] = 1
    y[df['OS'] > 450] = 2
    y = pd.get_dummies(y).as_matrix()
    x = standardization(x)
    x = normalization(x).as_matrix()
    return x, y


def calc_specifity_sensitivity(conf_matrix):
    nclasses = np.shape(conf_matrix)[0]
    sens = {}
    spec = {}
    for i in range(0, nclasses):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[i, :]) - tp
        fn = np.sum(conf_matrix[:, i]) - tp
        tn = np.sum(conf_matrix) - (tp + fp + fn)
        sens[i] = tp / (tp + fn)
        spec[i] = tn / (tn + fp)

    return sens, spec


x_train, y_train = import_hypothesis("30_train_features.csv")
x_test, y_test = import_hypothesis("30_test_features.csv")

x = tf.placeholder(tf.float32, [None, 30])
y_ = tf.placeholder(tf.float32, [None, 3])
W = tf.get_variable("Weights", [30, 3], initializer=tf.variance_scaling_initializer())
b = tf.get_variable("Bias", [3], initializer=tf.variance_scaling_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
optimize = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
confusion = tf.confusion_matrix(tf.arg_max(y, 1), tf.arg_max(y_, 1))

sess = tf.InteractiveSession()
sess.run(init)
for i in range(10000):
    _, acc_train, loss_train = sess.run([optimize, accuracy, cross_entropy], feed_dict={x: x_train, y_: y_train})
    acc_test, loss_test = sess.run([accuracy, cross_entropy], feed_dict={x: x_test, y_: y_test})
    if i % 100 == 0:
        print('Training Step:' + str(i) + '  Accuracy =  ' + str(acc_train) + '  Loss = ' + str(loss_train))
        print('Test Step:' + str(i) + '  Accuracy =  ' + str(acc_test) + '  Loss = ' + str(loss_test))

conf_matrix, weights = sess.run([confusion, W], feed_dict={x: x_test, y_: y_test})
print(weights)
sens, spec = calc_specifity_sensitivity(conf_matrix)
print(sens, spec)
