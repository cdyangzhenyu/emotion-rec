import tensorflow as tf
import sys
sys.path.append('../../../src')
import processMif as mif

#in_x=np.reshape(np.array(x).transpose(),[1,size,size,1])

img1 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
img2 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
img = tf.concat(values=[img1,img2],axis=3)
filter1 = tf.constant(value=0, shape=[3,3,1,1],dtype=tf.float32)
filter2 = tf.constant(value=1, shape=[3,3,1,1],dtype=tf.float32)
filter3 = tf.constant(value=2, shape=[3,3,1,1],dtype=tf.float32)
filter4 = tf.constant(value=3, shape=[3,3,1,1],dtype=tf.float32)
filter_out1 = tf.concat(values=[filter1,filter2],axis=2)
filter_out2 = tf.concat(values=[filter3,filter4],axis=2)
filter = tf.concat(values=[filter_out1,filter_out2],axis=3)

point_filter = tf.constant(value=1, shape=[1,1,4,4],dtype=tf.float32)
#out_img = tf.nn.depthwise_conv2d(input=img, filter=filter, strides=[1,1,1,1],rate=[1,1], padding='VALID')
#out_img = tf.nn.conv2d(input=out_img, filter=point_filter, strides=[1,1,1,1], padding='VALID')

'''also can be used'''
#out_img = tf.nn.separable_conv2d(input=img, depthwise_filter=filter, pointwise_filter=point_filter, strides=[1,1,1,1], rate=[1,1], padding='VALID')

def separable_conv2d(input, depthwise_filter, pointwise_filter):
    net = tf.nn.depthwise_conv2d(input=input, filter=depthwise_filter, strides=[1,1,1,1],rate=[1,1], padding='SAME')
    net = tf.nn.conv2d(input=net, filter=pointwise_filter, strides=[1,1,1,1], padding='SAME')
    return net

with tf.Session() as sess:
    x = img.eval()
    print img
    print filter
    print point_filter
    with tf.device("device:XLA_CPU:0"):
        #out_img1 = tf.nn.depthwise_conv2d(img, filter, strides=[1,1,1,1],rate=[1,1], padding='VALID')
        out_img1 = tf.nn.conv2d(img, filter, strides=[1,1,1,1], padding='SAME')
        out_img = tf.nn.conv2d(out_img1, tf.reshape(point_filter,[1,2,2,4]), strides=[1,1,1,1], padding='VALID')
        #out_img = tf.nn.conv2d(out_img1, point_filter, strides=[1,1,1,1], padding='SAME')
    print 'result:'
    print(sess.run(out_img, feed_dict={img: x}))

    mif.createMem([x,filter.eval(), point_filter.eval()])
