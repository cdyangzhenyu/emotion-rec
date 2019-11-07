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
graph = load_graph("../models/model-my-cnn-3557.pb")

for op in graph.get_operations():
    print(op.name)  

detection_model_path = '../haarcascade_files/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]

imagepath = "../test_pic/happy.jpg"
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

b_img_c = graph.get_tensor_by_name('image_array/bias:0')
b_c1 = graph.get_tensor_by_name('conv2d_1/bias:0')
b_c2 = graph.get_tensor_by_name('conv2d_2/bias:0')
b_c3 = graph.get_tensor_by_name('conv2d_3/bias:0')

d_k1 = graph.get_tensor_by_name('dense_1/kernel:0')
d_b1 = graph.get_tensor_by_name('dense_1/bias:0')

d_k2 = graph.get_tensor_by_name('dense_2/kernel:0')
d_b2 = graph.get_tensor_by_name('dense_2/bias:0')

epsilon = 0.00000001
with tf.Session(graph=graph) as sess:
    with tf.device("device:XLA_CPU:0"):
        net = tf.nn.conv2d(x, w_img_c, strides=[1, 1, 1, 1], padding='SAME')
        
        net = tf.add(net, b_img_c)    
        '''
        net = tf.nn.relu(net)
        
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        net = tf.nn.conv2d(net, w_c1, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c1)    
        net = tf.nn.relu(net)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        net = tf.nn.conv2d(net, w_c2, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(net, b_c2)
        net = tf.nn.relu(net)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        net = tf.nn.conv2d(net, w_c3, strides=[1, 1, 1, 1], padding='VALID')
        net = tf.add(net, b_c3)
        net = tf.nn.relu(net)
        net = tf.reshape(net, [-1, 100])
        net = tf.matmul(net, d_k1)
        net = tf.add(net, d_b1)

        net = tf.matmul(net, d_k2)
        net = tf.add(net, d_b2)
        print net
        # output
        '''
        net = tflearn.layers.conv.global_avg_pool(net, name='GlobalAvgPool')
        y = tf.nn.softmax(net)

    starttime = datetime.datetime.now()
    preds = sess.run(y, feed_dict={x: roi})
    endtime = datetime.datetime.now()
    delta = (endtime - starttime).microseconds/1000.0
    print preds

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
    param9 = d_k1.eval()#p9
    param10 = d_b1.eval()

    param11 = d_k2.eval()#p11
    param12 = d_b2.eval()

mif.createMem([param0,param1,param2,param3,param4,param5,param6,param7,param8,param9,
                param10,param11,param12])


emotion_probability = np.max(preds)
label = EMOTIONS[preds.argmax()]

print label
print "The emotion is %s, and prob is %s" % (label, emotion_probability)
print "The processing cost time: %s ms" % delta

