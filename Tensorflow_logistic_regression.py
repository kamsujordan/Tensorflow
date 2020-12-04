###################################################################################################################################
"""
    
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def get_dataset():
    """
       #Method used to generate the dataset
    """

    #Numbers of row per class
    row_per_class = 100
    #Generate rows
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    sick_2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

    healthy = np.random.randn(row_per_class, 2) + np.array([-2, 2])
    healthy_2 = np.random.randn(row_per_class, 2) + np.array([2, -2])

    features = np.vstack([sick, sick_2, healthy, healthy_2])
    targets = np.concatenate((np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))

    targets = targets.reshape(-1, 1)

    return features, targets

        
if __name__== '__main__':
    features, targets = get_dataset()
    
    #plot points
    plt.scatter(features[:,0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.show()

    #Input of the graph

    tf_features = tf.placeholder(tf.float32, features.shape)
    tf_targets = tf.placeholder(tf.float32, targets.shape)

    #First Layer
    w1 = tf.Variable(tf.random_normal([2, 3]))
    b1 = tf.Variable(tf.zeros([3]))


    #Operations
    z1 = tf.matmul(tf_features, w1) + b1
    a1 = tf.nn.sigmoid(z1)

    #Output neuron
    w2 = tf.Variable(tf.random_normal([3, 1]))
    b2 = tf.Variable(tf.zeros([1]))

    #Operations
    z2 = tf.matmul(a1, w2) + b2
    py = tf.nn.sigmoid(z2)


    #Cost
    cost = tf.reduce_mean(tf.square(py -tf_targets))

    #Accuracy
    correct_prediction = tf.equal(tf.round(py), tf_targets)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Optimizer
    learning_rate= 0.1
    optimizer = tf.train.GradientDescentOptimizer( learning_rate )
    train = optimizer.minimize(cost)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    sess.run(train, feed_dict={
        tf_features: features,
        tf_targets: targets
    })

    print("Vor dem Training: accuracy= ", sess.run(accuracy, feed_dict={
      tf_features: features,
      tf_targets: targets
      }))
    
    Accs_train = []
    Losses_train = []

    
    for epoch in range(100000):
        sess.run(train, feed_dict={
            tf_features: features,
            tf_targets: targets
        })
        if epoch % 10000 == 0:
            print("Cost:", sess.run(cost, feed_dict={
                tf_features: features,
                tf_targets: targets
            }))
            acc_train = sess.run(accuracy, feed_dict={
                tf_features: features,
                tf_targets: targets
                })
            Loss_train = sess.run(cost, feed_dict={
                tf_features: features,
                tf_targets: targets
                })
            
            Accs_train.append(acc_train)       
            Losses_train.append(Loss_train)
    
    #save_path = saver.save(sess, "./my_model.ckpt")
plt.plot(Accs_train, label="Train")
plt.legend(loc='upper left')
plt.title("Accuracy")
plt.savefig('TLR_Accuracy.png')
plt.show()   

plt.plot(Losses_train, label="Train")
plt.legend(loc='upper left')
plt.title("Loss")
plt.savefig('TLR_Loss.png')
plt.show()

   
print("Nach dem Training: accuracy= ", sess.run(accuracy, feed_dict={
        tf_features: features,
        tf_targets: targets
        }))

