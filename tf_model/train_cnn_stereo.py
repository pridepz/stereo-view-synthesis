# encoding: utf-8
'''
   tensorflow cnn-stereo

'''
# import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python import debug as tf_debug
from scipy import misc
import threading
from util import *
from corr import *
# from resnet_model import *
import time

# training parameters
reg_lambda = 1e-3
learning_rate = 1e-4
training_iters = 200000 #200000
batch_size = 16
display_step = 5
snapshot = 5000
width = 448
height = 448
width_res = 480  #LS:resize to this size
height_res = 480
width_orig = 640
height_orig = 640
path_to_model_save = './model/Dapangzi6_t1' # save the model weights
path_to_model_load = './model_init/Dapangzi6/train/dapangzi6_model' # load the model weights
path_to_log = './log' # for tensorboard logs
path_to_data = '/home/nerv/Research/cnn-stereo/data/hybrid_dataset_v3/'
interm = 'th70_'
file_train = path_to_data + 'hybrid_dataset_v2/hybrid_dataset_v5_' + interm + 'largeimage_TRAIN.list'
file_test = path_to_data + 'hybrid_dataset_v2/hybrid_dataset_v2_' + interm + 'TEST.list'
list_train_file = open(file_train, 'rb')
list_train = [[],[],[]]

# load the training data
with open(file_train, 'r') as openedfile:
    for line in openedfile:
        s = line.split()
        list_train[0].append(s[0])
        list_train[1].append(s[1])
        list_train[2].append(s[2])
n_train = len(list_train[0])
perm_train = np.random.permutation(n_train)
list_train_file.close()

# definition of negative relu
def relu_parameter(_input, negative_slope=0.1):
    return _input * negative_slope + tf.nn.relu(_input) * (1 - negative_slope)

# definition of DispNet
def disp_net_dapangzi6(_L, _R, _D_aug, _weights, _biases):

    with tf.variable_scope('qian'):
        #img_L &R --> conv0 --> con0 --> conv1 ReLU1 --> conv2ReLU3 --> corr --> conv_redir --> concat2 --> blob20
        #imag_L&R / 255.0
        Elt_L = tf.multiply(_L, 1 / 255.0)
        Elt_R = tf.multiply(_R, 1 / 255.0)

        #imag_L&R through conv0, -->
        with tf.variable_scope('conv0',):
            conv0_L = tf.nn.conv2d(tf.pad(_L, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv0-k_'], strides=[1, 1, 1, 1], padding='VALID')
            conv0_R = tf.nn.conv2d(tf.pad(_R, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv0-k_'], strides=[1, 1, 1, 1], padding='VALID')
#            tf.summary.image('conv0_L', conv0_L)
            con0_L = tf.concat([Elt_L, conv0_L], 3)
            con0_R = tf.concat([Elt_R, conv0_R], 3)

        with tf.variable_scope('conv1'):
            conv1_L = tf.nn.bias_add(tf.nn.conv2d(tf.pad(con0_L, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT"), _weights['conv1_'], strides=[1, 2, 2, 1], padding='VALID'), _biases['conv1_'])
            conv1_R = tf.nn.bias_add(tf.nn.conv2d(tf.pad(con0_R, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT"), _weights['conv1_'], strides=[1, 2, 2, 1], padding='VALID'), _biases['conv1_'])
            conv1_L_relu = relu_parameter(conv1_L, 0.1)
            conv1_R_relu = relu_parameter(conv1_R, 0.1)

        with tf.variable_scope('conv2'):
            conv2_L = tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv1_L_relu, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv2'], strides=[1, 2, 2, 1], padding='VALID'), _biases['conv2'])
            conv2_R = tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv1_R_relu, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv2'], strides=[1, 2, 2, 1], padding='VALID'), _biases['conv2'])
            conv2_L_relu = relu_parameter(conv2_L, 0.1)
            conv2_R_relu = relu_parameter(conv2_R, 0.1)  # 64 channels

        with tf.variable_scope('corr_ori'):
            #corr = corr1d(normalize_c(conv2_L_relu, epsilon = 1e-4), normalize_c(conv2_R_relu,  epsilon = 1e-4), l_disp=16, r_disp=10)
            corr = corr1d(conv2_L_relu, conv2_R_relu,l_disp=23, r_disp=3) #27 channels

        with tf.variable_scope('redir'):
            #redir = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(conv2_L_relu, _weights['redir'], strides=[1, 1, 1, 1], padding='SAME'), _biases['redir'])) #original one
            redir = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv2_L_relu, [[0, 0], [0, 0], [0, 0], [0, 0]], "CONSTANT"), _weights['redir'], strides=[1, 1, 1, 1], padding='VALID'), _biases['redir'])) # 48 channels

        with tf.variable_scope('blob20'):
            blob20 = tf.concat([corr, redir], 3)  #75 channels
    #blob20 --> conv3&_1 --> conv4&_1 --> conv5&_1 --> conv6&_1
    with tf.variable_scope('zhong'):
        with tf.variable_scope('conv3'):
            conv3 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(blob20, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv3'], strides=[1, 2, 2, 1], padding='VALID'), _biases['conv3']), 0.1) #128 channels
            conv3_1 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv3, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv3_1'], strides=[1, 1, 1, 1], padding='VALID'), _biases['conv3_1']), 0.1) #128 channels
        with tf.variable_scope('conv4'):
            conv4 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv3_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv4'], strides=[1, 2, 2, 1], padding='VALID'), _biases['conv4']), 0.1) # 256 channels
            conv4_1 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv4, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv4_1'], strides=[1, 1, 1, 1], padding='VALID'), _biases['conv4_1']), 0.1) # 256 channels
        with tf.variable_scope('conv5'):
            conv5 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv4_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv5'], strides=[1, 2, 2, 1], padding='VALID'), _biases['conv5']), 0.1) # 155 channels
            conv5_1 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv5, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv5_1'], strides=[1, 1, 1, 1], padding='VALID'), _biases['conv5_1']), 0.1) # 176 channels
        with tf.variable_scope('conv6'):
            conv6 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv5_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv6'], strides=[1, 2, 2, 1], padding='VALID'), _biases['conv6']), 0.1) #266 channels
            conv6_1 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv6, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['conv6_1'], strides=[1, 1, 1, 1], padding='VALID'), _biases['conv6_1']), 0.1) #260 channels

    with tf.variable_scope('hou'):
        with tf.variable_scope('predict_flow6'):
            predict_flow6 = tf.nn.bias_add(tf.nn.conv2d(tf.pad(conv6_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['predict_flow6'], strides=[1, 1, 1, 1], padding='VALID'), _biases['predict_flow6'])
            predict_flow6_relu = -tf.nn.relu(-predict_flow6)
            D6 = tf.image.resize_images(_D_aug, [predict_flow6.get_shape().as_list()[1],
                                            predict_flow6.get_shape().as_list()[2]],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            loss6 = tf.reduce_mean(tf.abs(D6 - predict_flow6_relu))
        with tf.variable_scope('nnup5'):
            nnup5 = tf.image.resize_images(conv6_1, [conv6_1.get_shape().as_list()[1]*4, conv6_1.get_shape().as_list()[2]*4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            cdeconv5 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(nnup5, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['cdeconv5'], strides=[1, 1, 1, 1], padding='VALID'), _biases['cdeconv5']), 0.1) # 128 channels
            interp3 = tf.nn.avg_pool(conv0_L, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding='SAME')
            blob41 = tf.concat([conv4_1, cdeconv5, interp3], 3) # 256 + 128 + 1 =385
            Convolution4 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(blob41, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['Convolution4'], strides=[1, 1, 1, 1], padding='VALID'), _biases['Convolution4']), 0.1)
        with tf.variable_scope('predict_flow4'):
            predict_flow4 = tf.nn.bias_add(tf.nn.conv2d(tf.pad(Convolution4, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['predict_flow4'], strides=[1, 1, 1, 1], padding='VALID'), _biases['predict_flow4'])
            predict_flow4_relu = -tf.nn.relu(-predict_flow4)
            D4 = tf.image.resize_images(_D_aug, [predict_flow4_relu.get_shape().as_list()[1],
                                            predict_flow4_relu.get_shape().as_list()[2]],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            loss4 = tf.reduce_mean(tf.abs(D4 - predict_flow4_relu))

        with tf.variable_scope('nnup3'):
            nnup3 = tf.image.resize_images(Convolution4, [Convolution4.get_shape().as_list()[1]*4, Convolution4.get_shape().as_list()[2]*4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            cdeconv3 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(nnup3,  [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['cdeconv3'], strides=[1, 1, 1, 1], padding='VALID'), _biases['cdeconv3']), 0.1) # 48 channels
            interp5 = tf.nn.avg_pool(conv0_L, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
            blob55 = tf.concat([blob20, cdeconv3, interp5], 3) # 75+48+1=124
            Convolution8 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(blob55, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['Convolution8'], strides=[1, 1, 1, 1], padding='VALID'), _biases['Convolution8']), 0.1)
        with tf.variable_scope('predict_flow2'):
            predict_flow2 = tf.nn.bias_add(tf.nn.conv2d(tf.pad(Convolution8, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['predict_flow2'], strides=[1, 1, 1, 1], padding='VALID'), _biases['predict_flow2'])
            predict_flow2_relu = -tf.nn.relu(-predict_flow2)
            D2 = tf.image.resize_images(_D_aug, [predict_flow2_relu.get_shape().as_list()[1],
                                            predict_flow2_relu.get_shape().as_list()[2]],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            loss2 = tf.reduce_mean(tf.abs(D2 - predict_flow2_relu))
        with tf.variable_scope('nnup1'):
            nnup1 = tf.image.resize_images(Convolution8, [Convolution8.get_shape().as_list()[1]*2, Convolution8.get_shape().as_list()[2]*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            cdeconv1 = relu_parameter(tf.nn.bias_add(tf.nn.conv2d(tf.pad(nnup1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['cdeconv1'], strides=[1, 1, 1, 1], padding='VALID'), _biases['cdeconv1']), 0.1)
            cde_interp = tf.image.resize_images(cdeconv1,[height, width], method=tf.image.ResizeMethod.BILINEAR)

            #interp7 = tf.nn.avg_pool(conv0_L, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #blob62 = tf.concat([conv1_L_relu, cdeconv1, interp7], 3)
            # conf_fin = tf.nn.bias_add(tf.nn.conv2d(cdeconv1, _weights['conf_fin'], strides=[1, 1, 1, 1], padding='SAME'), _biases['conf_fin'])

        with tf.variable_scope('predict_flow1'):
            predict_flow1 = tf.nn.bias_add(tf.nn.conv2d(tf.pad(cde_interp, [[0, 0], [0, 0], [0, 0], [0, 0]], "CONSTANT"), _weights['predict_flow1'], strides=[1, 1, 1, 1], padding='VALID'), _biases['predict_flow1'])
            predict_flow1_relu = -tf.nn.relu(-predict_flow1)
            D1 = tf.image.resize_images(_D_aug, [predict_flow1_relu.get_shape().as_list()[1],
                                            predict_flow1_relu.get_shape().as_list()[2]],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            loss1 = tf.reduce_mean(tf.abs(D1 - predict_flow1_relu))

        with tf.variable_scope('loss_total'):
            loss_all = 0.2 * loss6 + 0.6 * loss4 + loss2 + loss1
    return predict_flow1_relu, loss_all, loss1, loss2, loss4, loss6  #, loss6

# dictionary definition of the trainable parameter
graph = tf.Graph()
with graph.as_default():
    with tf.variable_scope('canshupf1'):
        weights = {
            'conv0-k_': tf.Variable(tf.random_normal([3, 3, 1, 1], dtype=tf.float32), dtype=tf.float32),
            'conv1_': tf.Variable(tf.random_normal([5, 5, 2, 48], dtype=tf.float32), dtype=tf.float32),
            'conv2': tf.Variable(tf.random_normal([3, 3, 48, 64], dtype=tf.float32), dtype=tf.float32),
            'redir': tf.Variable(tf.random_normal([1, 1, 64, 48], dtype=tf.float32), dtype=tf.float32),
            'conv3': tf.Variable(tf.random_normal([3, 3, 75, 128], dtype=tf.float32), dtype=tf.float32),
            # 64+corr output channels 27 =???
            'conv3_1': tf.Variable(tf.random_normal([3, 3, 128, 128], dtype=tf.float32), dtype=tf.float32),
            'conv4': tf.Variable(tf.random_normal([3, 3, 128, 256], dtype=tf.float32), dtype=tf.float32),
            'conv4_1': tf.Variable(tf.random_normal([3, 3, 256, 256], dtype=tf.float32), dtype=tf.float32),
            'conv5': tf.Variable(tf.random_normal([3, 3, 256, 155], dtype=tf.float32), dtype=tf.float32),
            'conv5_1': tf.Variable(tf.random_normal([3, 3, 155, 176], dtype=tf.float32), dtype=tf.float32),
            'conv6': tf.Variable(tf.random_normal([3, 3, 176, 266], dtype=tf.float32), dtype=tf.float32),
            'conv6_1': tf.Variable(tf.random_normal([3, 3, 266, 260], dtype=tf.float32), dtype=tf.float32),
            'predict_flow6': tf.Variable(tf.random_normal([3, 3, 260, 1], dtype=tf.float32), dtype=tf.float32),
            'cdeconv5': tf.Variable(tf.random_normal([3, 3, 260, 128], dtype=tf.float32), dtype=tf.float32),
            'Convolution4': tf.Variable(tf.random_normal([3, 3, 385, 128], dtype=tf.float32), dtype=tf.float32),
            'predict_flow4': tf.Variable(tf.random_normal([3, 3, 128, 1], dtype=tf.float32), dtype=tf.float32),
            'cdeconv3': tf.Variable(tf.random_normal([3, 3, 128, 48], dtype=tf.float32), dtype=tf.float32),
            'Convolution8': tf.Variable(tf.random_normal([3, 3, 124, 48], dtype=tf.float32), dtype=tf.float32),
            # 1+48+conv3 input channels
            'predict_flow2': tf.Variable(tf.random_normal([3, 3, 48, 1], dtype=tf.float32), dtype=tf.float32),
            'cdeconv1': tf.Variable(tf.random_normal([3, 3, 48, 16], dtype=tf.float32), dtype=tf.float32),
            # 'conf_fin': tf.Variable(tf.random_normal([3, 3, 16, 1], dtype=tf.float32), dtype=tf.float32),
            'predict_flow1': tf.Variable(tf.random_normal([1, 1, 16, 1], dtype=tf.float32), dtype=tf.float32)
        }
        bias = {
            'conv1_': tf.Variable(tf.random_normal([48], dtype=tf.float32), dtype=tf.float32),
            'conv2': tf.Variable(tf.random_normal([64], dtype=tf.float32), dtype=tf.float32),
            'redir': tf.Variable(tf.random_normal([48], dtype=tf.float32), dtype=tf.float32),
            'conv3': tf.Variable(tf.random_normal([128], dtype=tf.float32), dtype=tf.float32),
            'conv3_1': tf.Variable(tf.random_normal([128], dtype=tf.float32), dtype=tf.float32),
            'conv4': tf.Variable(tf.random_normal([256], dtype=tf.float32), dtype=tf.float32),
            'conv4_1': tf.Variable(tf.random_normal([256], dtype=tf.float32), dtype=tf.float32),
            'conv5': tf.Variable(tf.random_normal([155], dtype=tf.float32), dtype=tf.float32),
            'conv5_1': tf.Variable(tf.random_normal([176], dtype=tf.float32), dtype=tf.float32),
            'conv6': tf.Variable(tf.random_normal([266], dtype=tf.float32), dtype=tf.float32),
            'conv6_1': tf.Variable(tf.random_normal([260], dtype=tf.float32), dtype=tf.float32),
            'predict_flow6': tf.Variable(tf.random_normal([1], dtype=tf.float32), dtype=tf.float32),
            'cdeconv5': tf.Variable(tf.random_normal([128], dtype=tf.float32), dtype=tf.float32),
            'Convolution4': tf.Variable(tf.random_normal([128], dtype=tf.float32), dtype=tf.float32),
            'predict_flow4': tf.Variable(tf.random_normal([1], dtype=tf.float32), dtype=tf.float32),
            'cdeconv3': tf.Variable(tf.random_normal([48], dtype=tf.float32), dtype=tf.float32),
            'Convolution8': tf.Variable(tf.random_normal([48], dtype=tf.float32), dtype=tf.float32),
            'predict_flow2': tf.Variable(tf.random_normal([1], dtype=tf.float32), dtype=tf.float32),
            'cdeconv1': tf.Variable(tf.random_normal([16], dtype=tf.float32), dtype=tf.float32),
            # 'conf_fin': tf.Variable(tf.random_normal([1], dtype=tf.float32), dtype=tf.float32),
            'predict_flow1': tf.Variable(tf.random_normal([1], dtype=tf.float32), dtype=tf.float32)
        }

    # miscellaneous data loading and logging
    i_L = tf.placeholder(tf.float32, shape=[ height, width, 1])
    i_R = tf.placeholder(tf.float32, shape=[height, width, 1])
    i_D = tf.placeholder(tf.float32, shape=[ height, width, 1])
    q = tf.FIFOQueue(80, [tf.float32, tf.float32, tf.float32], shapes=[[height, width, 1], [height, width, 1], [height, width, 1]])
    enqueue_op = q.enqueue([i_L, i_R, i_D])
    L, R, D = q.dequeue_many(batch_size)
    predict_flow1, loss_disp, flow_loss1, flow_loss2, flow_loss4, flow_loss6 = disp_net_dapangzi6(L, R, D, weights, bias)

    # all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='canshupf1')
    trainable_v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    tf.summary.image('L', L)
    tf.summary.image('D', D)
    tf.summary.image('predict_flow1', predict_flow1)

    with tf.variable_scope('loss'):
        tf.summary.scalar('loss disp', loss_disp)
        tf.summary.scalar('flow_loss1', flow_loss1)
        tf.summary.scalar('flow_loss2', flow_loss2)
        tf.summary.scalar('flow_loss4', flow_loss4)
        tf.summary.scalar('flow_loss6', flow_loss6)

    with tf.variable_scope('summary'):
        summary_op = tf.summary.merge_all()
    l_rate_decay = tf.train.exponential_decay(learning_rate, training_iters, 4000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate_decay, epsilon=1e-4).minimize(loss_disp)

writer = tf.summary.FileWriter(path_to_log, graph)
saver = tf.train.Saver(all_variables) # all_variables)

# the real training session
with tf.Session(graph = graph) as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, path_to_model_load)
    print('Loaded variables:')
    print(all_variables)
    step = 0
    print('Trainable variables:')
    print(trainable_v)

    # data loader: a FIFO queue
    def load_and_enqueue():
        step = 0
        while True: #step < 10:
            idx = perm_train[(step - 1) % n_train]
            disp_gt, s, ttype = load_pfm(path_to_data + list_train[2][idx])
            disp_gt = -disp_gt * 32.
            x = np.random.random_integers(0, width_res - width - 1 )
            y = np.random.random_integers(0, height_res - height - 1)
            imgL = rgb2gray(misc.imread(path_to_data + list_train[0][idx]))
            imgR = rgb2gray(misc.imread(path_to_data + list_train[1][idx]))
            imgL = misc.imresize(imgL, (height_res, width_res), interp='bilinear')
            imgR = misc.imresize(imgR, (height_res, width_res), interp='bilinear')
            disp_gt = misc.imresize(disp_gt, (height_res, width_res), interp='nearest', mode='F')
            disp_gt = disp_gt * (width_res * 1. / width_orig )
            imgL = np.expand_dims(imgL[y : y + height, x : x + width], axis=2)
            imgR = np.expand_dims(imgR[y : y + height, x : x + width], axis=2)
            disp_gt = np.expand_dims(disp_gt[y: y + height, x: x + width], axis=2)
            sess.run(enqueue_op, feed_dict={i_L: imgL, i_R: imgR, i_D: disp_gt})
            step += 1

    t = threading.Thread(target=load_and_enqueue)
    t.start()
    t_start = time.time()

    # perform standard training
    while step < training_iters:
        if (step % display_step == 0 or step == 0):

            l1_all, summary_, flow1, loss1, loss2, loss4, loss6 = sess.run([loss_disp, summary_op, predict_flow1, flow_loss1, flow_loss2, flow_loss4, flow_loss6])
            # l1, l2, summary_, conv0_results, conv1_results, r_flow, flow1, pr_flow_SPN_, pr_SPN_, W_t_= sess.run([loss_disp, loss6, summary_op, predict_flow1, pr_flow_SPN, pr_SPN, W_t])

            writer.add_summary(summary_, step)
            print("Iteration: " + str(step) + "\nLoss 1: "  + str(loss1) +"    Loss 2: " + str(loss2) + "    Loss 4: " + str(loss4) + "    Loss 6: " + str(loss6))
            print("Loss total: " + str(l1_all) + "\n")

        # if step % 10 == 1:
        #     t_start = time.time()
        sess.run(optimizer)
        # if step % 10 ==0:
        #     print("10 iterations need time: %4.4f" % (time.time() - t_start))
        #     t_start = time.time()

        if step % snapshot == 0:
            saver.save(sess, path_to_model_save, global_step=step)
        step += 1
