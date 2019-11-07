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
graph = load_graph("../models/model-tiny-cnn-9753.pb")

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
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)


with tf.Session(graph=graph) as sess:
    pred = graph.get_tensor_by_name("predictions/Softmax:0")
    starttime = datetime.datetime.now()
    preds = sess.run(pred, feed_dict={"image_array_input:0": roi})
    endtime = datetime.datetime.now()
    delta = (endtime - starttime).microseconds/1000.0
    print preds

    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]

    print label
    print "The emotion is %s, and prob is %s" % (label, emotion_probability)
    print "The processing cost time: %s ms" % delta

    #print sess.run(graph.get_tensor_by_name("batch_normalization_1/batchnorm_1/add_1:0"), feed_dict={"image_array_input:0": roi})
    #duiying: net = tf.nn.batch_normalization(net, bn_1_moving_mean, bn_1_moving_variance, bn_1_beta, bn_1_gamma, 0.001)  
# =============convert to fpga verilog=================

x = graph.get_tensor_by_name('image_array_input:0')
w_img_c = graph.get_tensor_by_name('image_array/kernel:0')
w_c1 = graph.get_tensor_by_name('conv2d_1/kernel:0')
w_c2 = graph.get_tensor_by_name('conv2d_2/kernel:0')
w_c3 = graph.get_tensor_by_name('conv2d_3/kernel:0')
w_c4 = graph.get_tensor_by_name('conv2d_4/kernel:0')
w_c5 = graph.get_tensor_by_name('conv2d_5/kernel:0')
w_c6 = graph.get_tensor_by_name('conv2d_6/kernel:0')
w_c7 = graph.get_tensor_by_name('conv2d_7/kernel:0')
w_c8 = graph.get_tensor_by_name('conv2d_8/kernel:0')
w_c9 = graph.get_tensor_by_name('conv2d_9/kernel:0')

b_img_c = graph.get_tensor_by_name('image_array/bias:0')
b_c1 = graph.get_tensor_by_name('conv2d_1/bias:0')
b_c2 = graph.get_tensor_by_name('conv2d_2/bias:0')
b_c3 = graph.get_tensor_by_name('conv2d_3/bias:0')
b_c4 = graph.get_tensor_by_name('conv2d_4/bias:0')
b_c5 = graph.get_tensor_by_name('conv2d_5/bias:0')
b_c6 = graph.get_tensor_by_name('conv2d_6/bias:0')
b_c7 = graph.get_tensor_by_name('conv2d_7/bias:0')
b_c8 = graph.get_tensor_by_name('conv2d_8/bias:0')
b_c9 = graph.get_tensor_by_name('conv2d_9/bias:0')


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

epsilon = 0.00000001
with tf.Session(graph=graph) as sess:
    with tf.device("device:XLA_CPU:0"):
        # layer1
        net = tf.nn.conv2d(x, w_img_c, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_img_c)
        net = tf.nn.batch_normalization(net, bn_1_moving_mean, bn_1_moving_variance, bn_1_beta, bn_1_gamma, epsilon)      
        net = tf.nn.conv2d(net, w_c1, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c1)
        net = tf.nn.batch_normalization(net, bn_2_moving_mean, bn_2_moving_variance, bn_2_beta, bn_2_gamma, epsilon)
        net = tf.nn.relu(net)
        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.layers.average_pooling2d(net, [2,2], [2,2], padding='SAME')

        # layer2
        net = tf.nn.conv2d(net, w_c2, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c2)
        net = tf.nn.batch_normalization(net, bn_3_moving_mean, bn_3_moving_variance, bn_3_beta, bn_3_gamma, epsilon)      
        net = tf.nn.conv2d(net, w_c3, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c3)
        net = tf.nn.batch_normalization(net, bn_4_moving_mean, bn_4_moving_variance, bn_4_beta, bn_4_gamma, epsilon)
        net = tf.nn.relu(net)
        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.layers.average_pooling2d(net, [2,2], [2,2], padding='SAME')

        # layer3
        net = tf.nn.conv2d(net, w_c4, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c4)
        net = tf.nn.batch_normalization(net, bn_5_moving_mean, bn_5_moving_variance, bn_5_beta, bn_5_gamma, epsilon)      
        net = tf.nn.conv2d(net, w_c5, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c5)
        net = tf.nn.batch_normalization(net, bn_6_moving_mean, bn_6_moving_variance, bn_6_beta, bn_6_gamma, epsilon)
        net = tf.nn.relu(net)
        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.layers.average_pooling2d(net, [2,2], [2,2], padding='SAME')

        # layer4
        net = tf.nn.conv2d(net, w_c6, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c6)
        net = tf.nn.batch_normalization(net, bn_7_moving_mean, bn_7_moving_variance, bn_7_beta, bn_7_gamma, epsilon)      
        net = tf.nn.conv2d(net, w_c7, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c7)
        net = tf.nn.batch_normalization(net, bn_8_moving_mean, bn_8_moving_variance, bn_8_beta, bn_8_gamma, epsilon)
        net = tf.nn.relu(net)
        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.layers.average_pooling2d(net, [2,2], [2,2], padding='SAME')

        # layer5
        net = tf.nn.conv2d(net, w_c8, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c8)    
        net = tf.nn.conv2d(net, w_c9, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c9)

        # output
        net = tflearn.layers.conv.global_avg_pool(net, name='GlobalAvgPool')
        y = tf.nn.softmax(net)

    starttime = datetime.datetime.now()
    preds = sess.run(y, feed_dict={x: roi})
    endtime = datetime.datetime.now()
    delta = (endtime - starttime).microseconds/1000.0
    print preds

    '''
    # layer 1
    param1 = roi
    param0 = w_img_c.eval()
    param2 = b_img_c.eval()
    param3 = bn_1_moving_mean.eval()
    param4 = bn_1_moving_variance.eval()
    param5 = bn_1_beta.eval()
    param6 = bn_1_gamma.eval()

    param7 = w_c1.eval()#p3
    param8 = b_c1.eval()
    param9 = bn_2_moving_mean.eval()
    param10 = bn_2_moving_variance.eval()
    param11 = bn_2_beta.eval()
    param12 = bn_2_gamma.eval()

    # layer 2
    param13 = w_c2.eval()#p5
    param14 = b_c2.eval()
    param15 = bn_3_moving_mean.eval()
    param16 = bn_3_moving_variance.eval()
    param17 = bn_3_beta.eval()
    param18 = bn_3_gamma.eval()

    param19 = w_c3.eval()#p7
    param20 = b_c3.eval()
    param21 = bn_4_moving_mean.eval()
    param22 = bn_4_moving_variance.eval()
    param23 = bn_4_beta.eval()
    param24 = bn_4_gamma.eval()

    # layer 3
    param25 = w_c4.eval()#p9
    param26 = b_c4.eval()
    param27 = bn_5_moving_mean.eval()
    param28 = bn_5_moving_variance.eval()
    param29 = bn_5_beta.eval()
    param30 = bn_5_gamma.eval()

    param31 = w_c5.eval()#p11
    param32 = b_c5.eval()
    param33 = bn_6_moving_mean.eval()
    param34 = bn_6_moving_variance.eval()
    param35 = bn_6_beta.eval()
    param36 = bn_6_gamma.eval()

    # layer 4
    param37 = w_c6.eval()#p13
    param38 = b_c6.eval()
    param39 = bn_7_moving_mean.eval()
    param40 = bn_7_moving_variance.eval()
    param41 = bn_7_beta.eval()
    param42 = bn_7_gamma.eval()

    param43 = w_c7.eval()#p15
    param44 = b_c7.eval()
    param45 = bn_8_moving_mean.eval()
    param46 = bn_8_moving_variance.eval()
    param47 = bn_8_beta.eval()
    param48 = bn_8_gamma.eval()

    # layer 5
    param49 = w_c8.eval()#p17
    param50 = b_c8.eval()
    param51 = w_c9.eval()#p19
    param52 = b_c9.eval()

mif.createMem([param0,param1,param2,param3,param4,param5,param6,param7,param8,param9,
                param10,param11,param12,param13,param14,param15,param16,param17,param18,param19,
                param20,param21,param22,param23,param24,param25,param26,param27,param28,param29,
                param30,param31,param32,param33,param34,param35,param36,param37,param38,param39,
                param40,param41,param42,param43,param44,param45,param46,param47,param48,param49,
                param50,param51,param52])
    '''

    # layer 1
    param1 = roi
    param0 = w_img_c.eval()
    param2 = b_img_c.eval()


    param3 = w_c1.eval()#p3
    param4 = b_c1.eval()

    # layer 2
    param5 = w_c2.eval()#p5
    param6 = b_c2.eval()

    param7 = w_c3.eval()#p7
    param8 = b_c3.eval()

    # layer 3
    param9 = w_c4.eval()#p9
    param10 = b_c4.eval()


    param11 = w_c5.eval()#p11
    param12 = b_c5.eval()

    # layer 4
    param13 = w_c6.eval()#p13
    param14 = b_c6.eval()

    param15 = w_c7.eval()#p15
    param16 = b_c7.eval()

    # layer 5
    param17 = w_c8.eval()#p17
    param18 = b_c8.eval()
    param19 = w_c9.eval()#p19
    param20 = b_c9.eval()

mif.createMem([param0,param1,param2,param3,param4,param5,param6,param7,param8,param9,
                param10,param11,param12,param13,param14,param15,param16,param17,param18,param19,
                param20])


emotion_probability = np.max(preds)
label = EMOTIONS[preds.argmax()]

print label
print "The emotion is %s, and prob is %s" % (label, emotion_probability)
print "The processing cost time: %s ms" % delta

