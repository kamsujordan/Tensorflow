import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("input/data", one_hot= True)

"""
    Get a look at the dataset
"""
# print(mnist.train.labels[0])
# plt.imshow(mnist.train.images[0].reshape(28,28), cmap= "gray")
# plt.show()

# Neuron Layer
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="Weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X,W) + b
        
        if activation is not None:
            return activation(Z)
        else:
            return Z

n_inputs = 784 #28x28 Mnist
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10  #10 class

# Graph Inputs
tf_features = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
tf_targets = tf.placeholder(tf.float32, [None,  n_outputs])

# Output Layer
with tf.name_scope("dnn"):
    # hidden1 = neuron_layer(tf_features, n_hidden1, name="hidden1", activation=tf.nn.relu)
    # hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = neuron_layer(tf_features, n_outputs, name="outputs") #without hidden Layer, tf_feature is the Graph inputs



#Error
with tf.name_scope("error"):
    X_Entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf_targets, logits=logits)
    error = tf.reduce_mean(X_Entropy, name="loss")

# Train
learning_rate = 0.5
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(error)

# Metrics: Accuracy
with tf.name_scope("eval"):
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(tf_targets, 1))
    #correct_prediction = tf.nn.in_top_k(logits, tf_targets, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

softmax = tf.nn.softmax(logits)

init = tf.global_variables_initializer() 
# saver = tf.train.Saver() 
  
n_epochs = 5
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        true_cls = []
        py_cls = []
        for iteration in range(mnist.train.num_examples // batch_size):
            batch_features, batch_targets = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={
                tf_features: batch_features, 
                tf_targets: batch_targets
                })
    
        acc_train = sess.run(accuracy, feed_dict={
            tf_features: batch_features,
            tf_targets: batch_targets
            })
        acc_test = sess.run(accuracy, feed_dict={
            tf_features: mnist.test.images,
            tf_targets: mnist.test.labels
            })        
            
        py = sess.run(softmax, feed_dict={
            tf_features: [mnist.train.images[iteration]]
            })
            
        true_cls.append(np.argmax(mnist.train.labels[iteration]))
        py_cls.append(np.argmax(py))
            
        print (epoch, "Train Accuracy:", acc_train, "Test Accuracy:", acc_test)
        print ("True targets class", true_cls)
        print ("Predict Class", py_cls)        
        print ("logits", py)
            
        print(mnist.train.labels[iteration])
        plt.imshow(mnist.train.images[iteration].reshape(28,28), cmap= "gray")
        plt.show()
    
    # save_path = saver.save(sess, "./my_model_final.ckpt")