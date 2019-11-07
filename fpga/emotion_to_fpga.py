import tensorflow as tf
import tflearn
import numpy as np
import sys
import imutils
import cv2
import datetime
from keras.preprocessing.image import img_to_array
sys.path.append('../../../src')
import processMif as mif
import additionalOptions as options


def separable_conv2d(input, depthwise_filter, pointwise_filter):
    net = tf.nn.depthwise_conv2d(input=input, filter=depthwise_filter, strides=[1,1,1,1],rate=[1,1], padding='SAME')
    net = tf.nn.conv2d(input=net, filter=pointwise_filter, strides=[1,1,1,1], padding='SAME')
    return net


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

# We use our "load_graph" function
graph = load_graph("../models/model.pb")

for op in graph.get_operations():
    print(op.name)  

detection_model_path = '../haarcascade_files/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]

imagepath = "../test_pic/angry.png"
image = cv2.imread(imagepath)
image = imutils.resize(image,width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

if len(faces) > 0:
    faces = sorted(faces, reverse=True,
    key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
    # the ROI for classification via the CNN
    roi = gray[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)


with tf.Session(graph=graph) as sess:
    pred = graph.get_tensor_by_name("predictions/Softmax:0")
    starttime = datetime.datetime.now()
    preds = sess.run(pred, feed_dict={"input_1:0": roi})
    endtime = datetime.datetime.now()
    delta = (endtime - starttime).microseconds/1000.0
    print preds

    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]

    print label
    print "The emotion is %s, and prob is %s" % (label, emotion_probability)
    print "The processing cost time: %s ms" % delta

# =============convert to fpga verilog=================

    #x = graph.get_tensor_by_name('input_1:0')
    w_c1 = graph.get_tensor_by_name('conv2d_1/kernel:0')
    w_c2 = graph.get_tensor_by_name('conv2d_2/kernel:0')
    w_c3 = graph.get_tensor_by_name('conv2d_3/kernel:0')
    w_c4 = graph.get_tensor_by_name('conv2d_4/kernel:0')
    w_c5 = graph.get_tensor_by_name('conv2d_5/kernel:0')
    w_c6 = graph.get_tensor_by_name('conv2d_6/kernel:0')
    w_c7 = graph.get_tensor_by_name('conv2d_7/kernel:0')
    b_c7 = graph.get_tensor_by_name('conv2d_7/bias:0')

    w_depthwise_kernel_c1 = graph.get_tensor_by_name('separable_conv2d_1/depthwise_kernel:0')
    w_pointwise_kernel_c1 = graph.get_tensor_by_name('separable_conv2d_1/pointwise_kernel:0')
    w_depthwise_kernel_c2 = graph.get_tensor_by_name('separable_conv2d_2/depthwise_kernel:0')
    w_pointwise_kernel_c2 = graph.get_tensor_by_name('separable_conv2d_2/pointwise_kernel:0')
    w_depthwise_kernel_c3 = graph.get_tensor_by_name('separable_conv2d_3/depthwise_kernel:0')
    w_pointwise_kernel_c3 = graph.get_tensor_by_name('separable_conv2d_3/pointwise_kernel:0')
    w_depthwise_kernel_c4 = graph.get_tensor_by_name('separable_conv2d_4/depthwise_kernel:0')
    w_pointwise_kernel_c4 = graph.get_tensor_by_name('separable_conv2d_4/pointwise_kernel:0')
    w_depthwise_kernel_c5 = graph.get_tensor_by_name('separable_conv2d_5/depthwise_kernel:0')
    w_pointwise_kernel_c5 = graph.get_tensor_by_name('separable_conv2d_5/pointwise_kernel:0')
    w_depthwise_kernel_c6 = graph.get_tensor_by_name('separable_conv2d_6/depthwise_kernel:0')
    w_pointwise_kernel_c6 = graph.get_tensor_by_name('separable_conv2d_6/pointwise_kernel:0')
    w_depthwise_kernel_c7 = graph.get_tensor_by_name('separable_conv2d_7/depthwise_kernel:0')
    w_pointwise_kernel_c7 = graph.get_tensor_by_name('separable_conv2d_7/pointwise_kernel:0')
    w_depthwise_kernel_c8 = graph.get_tensor_by_name('separable_conv2d_8/depthwise_kernel:0')
    w_pointwise_kernel_c8 = graph.get_tensor_by_name('separable_conv2d_8/pointwise_kernel:0')

    bn_1_moving_mean = graph.get_tensor_by_name('batch_normalization_1/moving_mean:0')
    bn_1_moving_variance = graph.get_tensor_by_name('batch_normalization_1/moving_variance:0')
    bn_1_beta = graph.get_tensor_by_name('batch_normalization_1/beta:0')
    bn_1_gamma = graph.get_tensor_by_name('batch_normalization_1/gamma:0')

    bn_2_moving_mean = graph.get_tensor_by_name('batch_normalization_2/moving_mean:0')
    bn_2_moving_variance = graph.get_tensor_by_name('batch_normalization_2/moving_variance:0')
    bn_2_beta = graph.get_tensor_by_name('batch_normalization_2/beta:0')
    bn_2_gamma = graph.get_tensor_by_name('batch_normalization_2/gamma:0')

    bn_3_moving_mean = graph.get_tensor_by_name('batch_normalization_3/moving_mean:0')
    bn_3_moving_variance = graph.get_tensor_by_name('batch_normalization_3/moving_variance:0')
    bn_3_beta = graph.get_tensor_by_name('batch_normalization_3/beta:0')
    bn_3_gamma = graph.get_tensor_by_name('batch_normalization_3/gamma:0')

    bn_4_moving_mean = graph.get_tensor_by_name('batch_normalization_4/moving_mean:0')
    bn_4_moving_variance = graph.get_tensor_by_name('batch_normalization_4/moving_variance:0')
    bn_4_beta = graph.get_tensor_by_name('batch_normalization_4/beta:0')
    bn_4_gamma = graph.get_tensor_by_name('batch_normalization_4/gamma:0')

    bn_5_moving_mean = graph.get_tensor_by_name('batch_normalization_5/moving_mean:0')
    bn_5_moving_variance = graph.get_tensor_by_name('batch_normalization_5/moving_variance:0')
    bn_5_beta = graph.get_tensor_by_name('batch_normalization_5/beta:0')
    bn_5_gamma = graph.get_tensor_by_name('batch_normalization_5/gamma:0')

    bn_6_moving_mean = graph.get_tensor_by_name('batch_normalization_6/moving_mean:0')
    bn_6_moving_variance = graph.get_tensor_by_name('batch_normalization_6/moving_variance:0')
    bn_6_beta = graph.get_tensor_by_name('batch_normalization_6/beta:0')
    bn_6_gamma = graph.get_tensor_by_name('batch_normalization_6/gamma:0')

    bn_7_moving_mean = graph.get_tensor_by_name('batch_normalization_7/moving_mean:0')
    bn_7_moving_variance = graph.get_tensor_by_name('batch_normalization_7/moving_variance:0')
    bn_7_beta = graph.get_tensor_by_name('batch_normalization_7/beta:0')
    bn_7_gamma = graph.get_tensor_by_name('batch_normalization_7/gamma:0')

    bn_8_moving_mean = graph.get_tensor_by_name('batch_normalization_8/moving_mean:0')
    bn_8_moving_variance = graph.get_tensor_by_name('batch_normalization_8/moving_variance:0')
    bn_8_beta = graph.get_tensor_by_name('batch_normalization_8/beta:0')
    bn_8_gamma = graph.get_tensor_by_name('batch_normalization_8/gamma:0')

    bn_9_moving_mean = graph.get_tensor_by_name('batch_normalization_9/moving_mean:0')
    bn_9_moving_variance = graph.get_tensor_by_name('batch_normalization_9/moving_variance:0')
    bn_9_beta = graph.get_tensor_by_name('batch_normalization_9/beta:0')
    bn_9_gamma = graph.get_tensor_by_name('batch_normalization_9/gamma:0')

    bn_10_moving_mean = graph.get_tensor_by_name('batch_normalization_10/moving_mean:0')
    bn_10_moving_variance = graph.get_tensor_by_name('batch_normalization_10/moving_variance:0')
    bn_10_beta = graph.get_tensor_by_name('batch_normalization_10/beta:0')
    bn_10_gamma = graph.get_tensor_by_name('batch_normalization_10/gamma:0')

    bn_11_moving_mean = graph.get_tensor_by_name('batch_normalization_11/moving_mean:0')
    bn_11_moving_variance = graph.get_tensor_by_name('batch_normalization_11/moving_variance:0')
    bn_11_beta = graph.get_tensor_by_name('batch_normalization_11/beta:0')
    bn_11_gamma = graph.get_tensor_by_name('batch_normalization_11/gamma:0')

    bn_12_moving_mean = graph.get_tensor_by_name('batch_normalization_12/moving_mean:0')
    bn_12_moving_variance = graph.get_tensor_by_name('batch_normalization_12/moving_variance:0')
    bn_12_beta = graph.get_tensor_by_name('batch_normalization_12/beta:0')
    bn_12_gamma = graph.get_tensor_by_name('batch_normalization_12/gamma:0')

    bn_13_moving_mean = graph.get_tensor_by_name('batch_normalization_13/moving_mean:0')
    bn_13_moving_variance = graph.get_tensor_by_name('batch_normalization_13/moving_variance:0')
    bn_13_beta = graph.get_tensor_by_name('batch_normalization_13/beta:0')
    bn_13_gamma = graph.get_tensor_by_name('batch_normalization_13/gamma:0')

    bn_14_moving_mean = graph.get_tensor_by_name('batch_normalization_14/moving_mean:0')
    bn_14_moving_variance = graph.get_tensor_by_name('batch_normalization_14/moving_variance:0')
    bn_14_beta = graph.get_tensor_by_name('batch_normalization_14/beta:0')
    bn_14_gamma = graph.get_tensor_by_name('batch_normalization_14/gamma:0')

    # input
    w_c1 = w_c1.eval()
    bn_1_moving_mean = bn_1_moving_mean.eval()
    bn_1_moving_variance = bn_1_moving_variance.eval()
    bn_1_beta = bn_1_beta.eval()
    bn_1_gamma = bn_1_gamma.eval()

    w_c2 = w_c2.eval()
    bn_2_moving_mean = bn_2_moving_mean.eval()
    bn_2_moving_variance = bn_2_moving_variance.eval()
    bn_2_beta = bn_2_beta.eval()
    bn_2_gamma = bn_2_gamma.eval()

    # module 1
    w_c3 = w_c3.eval()
    bn_3_moving_mean = bn_3_moving_mean.eval()
    bn_3_moving_variance = bn_3_moving_variance.eval()
    bn_3_beta = bn_3_beta.eval()
    bn_3_gamma = bn_3_gamma.eval()

    w_depthwise_kernel_c1 = w_depthwise_kernel_c1.eval()
    w_pointwise_kernel_c1 = w_pointwise_kernel_c1.eval()
    bn_4_moving_mean = bn_4_moving_mean.eval()
    bn_4_moving_variance = bn_4_moving_variance.eval()
    bn_4_beta = bn_4_beta.eval()
    bn_4_gamma = bn_4_gamma.eval()

    w_depthwise_kernel_c2 = w_depthwise_kernel_c2.eval()
    w_pointwise_kernel_c2 = w_pointwise_kernel_c2.eval()
    bn_5_moving_mean = bn_5_moving_mean.eval()
    bn_5_moving_variance = bn_5_moving_variance.eval()
    bn_5_beta = bn_5_beta.eval()
    bn_5_gamma = bn_5_gamma.eval()

    # module 2
    w_c4 = w_c4.eval()
    bn_6_moving_mean = bn_6_moving_mean.eval()
    bn_6_moving_variance = bn_6_moving_variance.eval()
    bn_6_beta = bn_6_beta.eval()
    bn_6_gamma = bn_6_gamma.eval()

    w_depthwise_kernel_c3 = w_depthwise_kernel_c3.eval()
    w_pointwise_kernel_c3 = w_pointwise_kernel_c3.eval()
    bn_7_moving_mean = bn_7_moving_mean.eval()
    bn_7_moving_variance = bn_7_moving_variance.eval()
    bn_7_beta = bn_7_beta.eval()
    bn_7_gamma = bn_7_gamma.eval()

    w_depthwise_kernel_c4 = w_depthwise_kernel_c4.eval()
    w_pointwise_kernel_c4 = w_pointwise_kernel_c4.eval()
    bn_8_moving_mean = bn_8_moving_mean.eval()
    bn_8_moving_variance = bn_8_moving_variance.eval()
    bn_8_beta = bn_8_beta.eval()
    bn_8_gamma = bn_8_gamma.eval()

    # module 3
    w_c5 = w_c5.eval()
    bn_9_moving_mean = bn_9_moving_mean.eval()
    bn_9_moving_variance = bn_9_moving_variance.eval()
    bn_9_beta = bn_9_beta.eval()
    bn_9_gamma = bn_9_gamma.eval()

    w_depthwise_kernel_c5 = w_depthwise_kernel_c5.eval()
    w_pointwise_kernel_c5 = w_pointwise_kernel_c5.eval()
    bn_10_moving_mean = bn_10_moving_mean.eval()
    bn_10_moving_variance = bn_10_moving_variance.eval()
    bn_10_beta = bn_10_beta.eval()
    bn_10_gamma = bn_10_gamma.eval()

    w_depthwise_kernel_c6 = w_depthwise_kernel_c6.eval()
    w_pointwise_kernel_c6 = w_pointwise_kernel_c6.eval()
    bn_11_moving_mean = bn_11_moving_mean.eval()
    bn_11_moving_variance = bn_11_moving_variance.eval()
    bn_11_beta = bn_11_beta.eval()
    bn_11_gamma = bn_11_gamma.eval()

    # module 4
    w_c6 = w_c6.eval()
    bn_12_moving_mean = bn_12_moving_mean.eval()
    bn_12_moving_variance = bn_12_moving_variance.eval()
    bn_12_beta = bn_12_beta.eval()
    bn_12_gamma = bn_12_gamma.eval()

    w_depthwise_kernel_c7 = w_depthwise_kernel_c7.eval()
    w_pointwise_kernel_c7 = w_pointwise_kernel_c7.eval()
    bn_13_moving_mean = bn_13_moving_mean.eval()
    bn_13_moving_variance = bn_13_moving_variance.eval()
    bn_13_beta = bn_13_beta.eval()
    bn_13_gamma = bn_13_gamma.eval()

    w_depthwise_kernel_c8 = w_depthwise_kernel_c8.eval()
    w_pointwise_kernel_c8 = w_pointwise_kernel_c8.eval()
    bn_14_moving_mean = bn_14_moving_mean.eval()
    bn_14_moving_variance = bn_14_moving_variance.eval()
    bn_14_beta = bn_14_beta.eval()
    bn_14_gamma = bn_14_gamma.eval()

    w_c7 = w_c7.eval()
    b_c7 = b_c7.eval()


with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[None,64,64,1], name="input")
    w_c1 = tf.constant(w_c1)
    bn_1_moving_mean = tf.constant(bn_1_moving_mean)
    bn_1_moving_variance = tf.constant(bn_1_moving_variance)
    bn_1_beta = tf.constant(bn_1_beta)
    bn_1_gamma = tf.constant(bn_1_gamma)

    w_c2 = tf.constant(w_c2)
    bn_2_moving_mean = tf.constant(bn_2_moving_mean)
    bn_2_moving_variance = tf.constant(bn_2_moving_variance)
    bn_2_beta = tf.constant(bn_2_beta)
    bn_2_gamma = tf.constant(bn_2_gamma)

    # module 1
    w_c3 = tf.constant(w_c3)
    bn_3_moving_mean = tf.constant(bn_3_moving_mean)
    bn_3_moving_variance = tf.constant(bn_3_moving_variance)
    bn_3_beta = tf.constant(bn_3_beta)
    bn_3_gamma = tf.constant(bn_3_gamma)

    w_depthwise_kernel_c1 = tf.constant(w_depthwise_kernel_c1)
    w_pointwise_kernel_c1 = tf.constant(w_pointwise_kernel_c1)
    bn_4_moving_mean = tf.constant(bn_4_moving_mean)
    bn_4_moving_variance = tf.constant(bn_4_moving_variance)
    bn_4_beta = tf.constant(bn_4_beta)
    bn_4_gamma = tf.constant(bn_4_gamma)

    w_depthwise_kernel_c2 = tf.constant(w_depthwise_kernel_c2)
    w_pointwise_kernel_c2 = tf.constant(w_pointwise_kernel_c2)
    bn_5_moving_mean = tf.constant(bn_5_moving_mean)
    bn_5_moving_variance = tf.constant(bn_5_moving_variance)
    bn_5_beta = tf.constant(bn_5_beta)
    bn_5_gamma = tf.constant(bn_5_gamma)

    # module 2
    w_c4 = tf.constant(w_c4)
    bn_6_moving_mean = tf.constant(bn_6_moving_mean)
    bn_6_moving_variance = tf.constant(bn_6_moving_variance)
    bn_6_beta = tf.constant(bn_6_beta)
    bn_6_gamma = tf.constant(bn_6_gamma)

    w_depthwise_kernel_c3 = tf.constant(w_depthwise_kernel_c3)
    w_pointwise_kernel_c3 = tf.constant(w_pointwise_kernel_c3)
    bn_7_moving_mean = tf.constant(bn_7_moving_mean)
    bn_7_moving_variance = tf.constant(bn_7_moving_variance)
    bn_7_beta = tf.constant(bn_7_beta)
    bn_7_gamma = tf.constant(bn_7_gamma)

    w_depthwise_kernel_c4 = tf.constant(w_depthwise_kernel_c4)
    w_pointwise_kernel_c4 = tf.constant(w_pointwise_kernel_c4)
    bn_8_moving_mean = tf.constant(bn_8_moving_mean)
    bn_8_moving_variance = tf.constant(bn_8_moving_variance)
    bn_8_beta = tf.constant(bn_8_beta)
    bn_8_gamma = tf.constant(bn_8_gamma)

    # module 3
    w_c5 = tf.constant(w_c5)
    bn_9_moving_mean = tf.constant(bn_9_moving_mean)
    bn_9_moving_variance = tf.constant(bn_9_moving_variance)
    bn_9_beta = tf.constant(bn_9_beta)
    bn_9_gamma = tf.constant(bn_9_gamma)

    w_depthwise_kernel_c5 = tf.constant(w_depthwise_kernel_c5)
    w_pointwise_kernel_c5 = tf.constant(w_pointwise_kernel_c5)
    bn_10_moving_mean = tf.constant(bn_10_moving_mean)
    bn_10_moving_variance = tf.constant(bn_10_moving_variance)
    bn_10_beta = tf.constant(bn_10_beta)
    bn_10_gamma = tf.constant(bn_10_gamma)

    w_depthwise_kernel_c6 = tf.constant(w_depthwise_kernel_c6)
    w_pointwise_kernel_c6 = tf.constant(w_pointwise_kernel_c6)
    bn_11_moving_mean = tf.constant(bn_11_moving_mean)
    bn_11_moving_variance = tf.constant(bn_11_moving_variance)
    bn_11_beta = tf.constant(bn_11_beta)
    bn_11_gamma = tf.constant(bn_11_gamma)

    # module 4
    w_c6 = tf.constant(w_c6)
    bn_12_moving_mean = tf.constant(bn_12_moving_mean)
    bn_12_moving_variance = tf.constant(bn_12_moving_variance)
    bn_12_beta = tf.constant(bn_12_beta)
    bn_12_gamma = tf.constant(bn_12_gamma)

    w_depthwise_kernel_c7 = tf.constant(w_depthwise_kernel_c7)
    w_pointwise_kernel_c7 = tf.constant(w_pointwise_kernel_c7)
    bn_13_moving_mean = tf.constant(bn_13_moving_mean)
    bn_13_moving_variance = tf.constant(bn_13_moving_variance)
    bn_13_beta = tf.constant(bn_13_beta)
    bn_13_gamma = tf.constant(bn_13_gamma)

    w_depthwise_kernel_c8 = tf.constant(w_depthwise_kernel_c8)
    print w_depthwise_kernel_c8
    w_pointwise_kernel_c8 = tf.constant(w_pointwise_kernel_c8)
    print w_depthwise_kernel_c8
    bn_14_moving_mean = tf.constant(bn_14_moving_mean)
    bn_14_moving_variance = tf.constant(bn_14_moving_variance)
    bn_14_beta = tf.constant(bn_14_beta)
    bn_14_gamma = tf.constant(bn_14_gamma)

    w_c7 = tf.constant(w_c7)
    b_c7 = tf.constant(b_c7)

    with tf.Session() as sess:
        with tf.device("device:XLA_CPU:0"):
            # input
            net = tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.batch_normalization(net, bn_1_moving_mean, bn_1_moving_variance, bn_1_beta, bn_1_gamma, 0.001)
            net = tf.nn.relu(net)

            net = tf.nn.conv2d(net, w_c2, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.batch_normalization(net, bn_2_moving_mean, bn_2_moving_variance, bn_2_beta, bn_2_gamma, 0.001)
            net = tf.nn.relu(net)
            
            # module 1
            residual = tf.nn.conv2d(net, w_c3, strides=[1, 2, 2, 1], padding='SAME')
            residual = tf.nn.batch_normalization(residual, bn_3_moving_mean, bn_3_moving_variance, bn_3_beta, bn_3_gamma, 0.001)
            
            #net = tf.nn.separable_conv2d(net, w_depthwise_kernel_c1, w_pointwise_kernel_c1, strides=[1,1,1,1], rate=[1,1], padding="SAME")
            net = separable_conv2d(net, w_depthwise_kernel_c1, w_pointwise_kernel_c1)
            '''
            net = tf.nn.batch_normalization(net, bn_4_moving_mean, bn_4_moving_variance, bn_4_beta, bn_4_gamma, 0.001)
            net = tf.nn.relu(net)
            
            net = tf.nn.separable_conv2d(net, w_depthwise_kernel_c2, w_pointwise_kernel_c2, strides=[1,1,1,1], rate=[1,1], padding="SAME")
            net = tf.nn.batch_normalization(net, bn_5_moving_mean, bn_5_moving_variance, bn_5_beta, bn_5_gamma, 0.001)
            net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            net = tf.add(net, residual)
            
            # module 2
            residual = tf.nn.conv2d(net, w_c4, strides=[1, 2, 2, 1], padding='SAME')
            residual = tf.nn.batch_normalization(residual, bn_6_moving_mean, bn_6_moving_variance, bn_6_beta, bn_6_gamma, 0.001)

            net = tf.nn.separable_conv2d(net, w_depthwise_kernel_c3, w_pointwise_kernel_c3, strides=[1,1,1,1], rate=[1,1], padding="SAME")
            net = tf.nn.batch_normalization(net, bn_7_moving_mean, bn_7_moving_variance, bn_7_beta, bn_7_gamma, 0.001)
            net = tf.nn.relu(net)

            net = tf.nn.separable_conv2d(net, w_depthwise_kernel_c4, w_pointwise_kernel_c4, strides=[1,1,1,1], rate=[1,1], padding="SAME")
            net = tf.nn.batch_normalization(net, bn_8_moving_mean, bn_8_moving_variance, bn_8_beta, bn_8_gamma, 0.001)
            net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            net = tf.add(net, residual)

            # module 3
            residual = tf.nn.conv2d(net, w_c5, strides=[1, 2, 2, 1], padding='SAME')
            residual = tf.nn.batch_normalization(residual, bn_9_moving_mean, bn_9_moving_variance, bn_9_beta, bn_9_gamma, 0.001)

            net = tf.nn.separable_conv2d(net, w_depthwise_kernel_c5, w_pointwise_kernel_c5, strides=[1,1,1,1], rate=[1,1], padding="SAME")
            net = tf.nn.batch_normalization(net, bn_10_moving_mean, bn_10_moving_variance, bn_10_beta, bn_10_gamma, 0.001)
            net = tf.nn.relu(net)

            net = tf.nn.separable_conv2d(net, w_depthwise_kernel_c6, w_pointwise_kernel_c6, strides=[1,1,1,1], rate=[1,1], padding="SAME")
            net = tf.nn.batch_normalization(net, bn_11_moving_mean, bn_11_moving_variance, bn_11_beta, bn_11_gamma, 0.001)
            net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            net = tf.add(net, residual)

            # module 4
            residual = tf.nn.conv2d(net, w_c6, strides=[1, 2, 2, 1], padding='SAME')
            residual = tf.nn.batch_normalization(residual, bn_12_moving_mean, bn_12_moving_variance, bn_12_beta, bn_12_gamma, 0.001)

            net = tf.nn.separable_conv2d(net, w_depthwise_kernel_c7, w_pointwise_kernel_c7, strides=[1,1,1,1], rate=[1,1], padding="SAME")
            net = tf.nn.batch_normalization(net, bn_13_moving_mean, bn_13_moving_variance, bn_13_beta, bn_13_gamma, 0.001)
            net = tf.nn.relu(net)

            net = tf.nn.separable_conv2d(net, w_depthwise_kernel_c8, w_pointwise_kernel_c8, strides=[1,1,1,1], rate=[1,1], padding="SAME")
            
            net = tf.nn.batch_normalization(net, bn_14_moving_mean, bn_14_moving_variance, bn_14_beta, bn_14_gamma, 0.001)
            net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            net = tf.add(net, residual)

            #output
            net = tf.add(tf.nn.conv2d(net, w_c7, strides=[1, 1, 1, 1], padding='SAME'), b_c7)
            net = tflearn.layers.conv.global_avg_pool(net, name='GlobalAvgPool')
            '''
            y = tf.nn.softmax(net)

        starttime = datetime.datetime.now()
        preds = sess.run(y, feed_dict={x: roi})
        endtime = datetime.datetime.now()
        delta = (endtime - starttime).microseconds/1000.0
        print preds
'''
        # input
        param1 = roi
        param0 = w_c1.eval()
        param2 = bn_1_moving_mean.eval()
        param3 = bn_1_moving_variance.eval()
        param4 = bn_1_beta.eval()
        param5 = bn_1_gamma.eval()

        param6 = w_c2.eval()
        param7 = bn_2_moving_mean.eval()
        param8 = bn_2_moving_variance.eval()
        param9 = bn_2_beta.eval()
        param10 = bn_2_gamma.eval()

        # module 1
        param11 = w_c3.eval()
        param12 = bn_3_moving_mean.eval()
        param13 = bn_3_moving_variance.eval()
        param14 = bn_3_beta.eval()
        param15 = bn_3_gamma.eval()

        param16 = w_depthwise_kernel_c1.eval()
        param17 = w_pointwise_kernel_c1.eval()
        param18 = bn_4_moving_mean.eval()
        param19 = bn_4_moving_variance.eval()
        param20 = bn_4_beta.eval()
        param21 = bn_4_gamma.eval()

        param22 = w_depthwise_kernel_c2.eval()
        param23 = w_pointwise_kernel_c2.eval()
        param24 = bn_5_moving_mean.eval()
        param25 = bn_5_moving_variance.eval()
        param26 = bn_5_beta.eval()
        param27 = bn_5_gamma.eval()

        # module 2
        param28 = w_c4.eval()
        param29 = bn_6_moving_mean.eval()
        param30 = bn_6_moving_variance.eval()
        param31 = bn_6_beta.eval()
        param32 = bn_6_gamma.eval()

        param33 = w_depthwise_kernel_c3.eval()
        param34 = w_pointwise_kernel_c3.eval()
        param35 = bn_7_moving_mean.eval()
        param36 = bn_7_moving_variance.eval()
        param37 = bn_7_beta.eval()
        param38 = bn_7_gamma.eval()

        param39 = w_depthwise_kernel_c4.eval()
        param40 = w_pointwise_kernel_c4.eval()
        param41 = bn_8_moving_mean.eval()
        param42 = bn_8_moving_variance.eval()
        param43 = bn_8_beta.eval()
        param44 = bn_8_gamma.eval()

        # module 3
        param45 = w_c5.eval()
        param46 = bn_9_moving_mean.eval()
        param47 = bn_9_moving_variance.eval()
        param48 = bn_9_beta.eval()
        param49 = bn_9_gamma.eval()

        param50 = w_depthwise_kernel_c5.eval()
        param51 = w_pointwise_kernel_c5.eval()
        param52 = bn_10_moving_mean.eval()
        param53 = bn_10_moving_variance.eval()
        param54 = bn_10_beta.eval()
        param55 = bn_10_gamma.eval()

        param56 = w_depthwise_kernel_c6.eval()
        param57 = w_pointwise_kernel_c6.eval()
        param58 = bn_11_moving_mean.eval()
        param59 = bn_11_moving_variance.eval()
        param60 = bn_11_beta.eval()
        param61 = bn_11_gamma.eval()

        # module 4
        param62 = w_c6.eval()
        param63 = bn_12_moving_mean.eval()
        param64 = bn_12_moving_variance.eval()
        param65 = bn_12_beta.eval()
        param66 = bn_12_gamma.eval()

        param67 = w_depthwise_kernel_c7.eval()
        param68 = w_pointwise_kernel_c7.eval()
        param69 = bn_13_moving_mean.eval()
        param70 = bn_13_moving_variance.eval()
        param71 = bn_13_beta.eval()
        param72 = bn_13_gamma.eval()

        param73 = w_depthwise_kernel_c8.eval()
        param74 = w_pointwise_kernel_c8.eval()
        param75 = bn_14_moving_mean.eval()
        param76 = bn_14_moving_variance.eval()
        param77 = bn_14_beta.eval()
        param78 = bn_14_gamma.eval()

        # output
        param79 = w_c7.eval()
        param80 = b_c7.eval()

mif.createMem([param0,param1,param2,param3,param4,param5,param6,param7,param8,param9,
                param10,param11,param12,param13,param14,param15,param16,param17,param18,param19,
                param20,param21,param22,param23,param24,param25,param26,param27,param28,param29,
                param30,param31,param32,param33,param34,param35,param36,param37,param38,param39,
                param40,param41,param42,param43,param44,param45,param46,param47,param48,param49,
                param50,param51,param52,param53,param54,param55,param56,param57,param58,param59,
                param60,param61,param62,param63,param64,param65,param66,param67,param68,param69,
                param70,param71,param72,param73,param74,param75,param76,param77,param78,param79,
                param80])
'''
emotion_probability = np.max(preds)
label = EMOTIONS[preds.argmax()]

print label
print "The emotion is %s, and prob is %s" % (label, emotion_probability)
print "The processing cost time: %s ms" % delta

print [w_depthwise_kernel_c1,w_pointwise_kernel_c1,w_depthwise_kernel_c2,w_pointwise_kernel_c2,
       w_depthwise_kernel_c3,w_pointwise_kernel_c3,w_depthwise_kernel_c4,w_pointwise_kernel_c4,
       w_depthwise_kernel_c5,w_pointwise_kernel_c5,w_depthwise_kernel_c6,w_pointwise_kernel_c6,
       w_depthwise_kernel_c7,w_pointwise_kernel_c7,w_depthwise_kernel_c8,w_pointwise_kernel_c8]



