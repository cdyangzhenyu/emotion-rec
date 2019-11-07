import tensorflow as tf
import numpy as np
import string

size = 5
outputs = 2

X = tf.placeholder(tf.float32, [1, size,size,1])
weights = tf.placeholder(tf.float32, [3,3,1,outputs])
w1 = tf.placeholder(tf.float32, [3,3,2,outputs])

x=[1,0,1,1,-1,1,1,0,-1,0,1,0,0,-1,0,1,0,1,1,-1,-1,0,-1,1,1]
w11=[1,-1,1,0,0,0,1,0,0,1,0,-1,0,1,1,0,0,-1]
w2=[1,-1,1,0,0,0,1,-1,1,0,0,0,1,-1,1,0,0,0,0,1,1,0,0,-1,0,1,1,0,0,-1,1,0,0,1,-1,0]
in_x=np.reshape(np.array(x).transpose(),[1,size,size,1])

in_weights = np.reshape(np.array(w11).transpose(),[3,3,1,outputs])
in_w1 = np.reshape(np.array(w2).transpose(),[3,3,2,outputs])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y1 = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
    y = tf.nn.conv2d(y1, w1, strides=[1, 1, 1, 1], padding='SAME')
    
    result = sess.run(y, feed_dict={X: in_x, weights: in_weights, w1: in_w1})

    print result
