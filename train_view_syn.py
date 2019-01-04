from __future__ import division
import tensorflow as tf

from scipy import misc
from scipy import interpolate
import numpy as np
from numpy.linalg import inv
import cv2
import os
import threading
import time
import re
from train_cnn_stereo import *

flags = tf.app.flags
#Input flags
flags.DEFINE_string('path_to_pose', '/mnt/lustre/panzheng/dataset/stereo_dataset/valpos_train_i0', 'input pose directory')
flags.DEFINE_string('path_to_image', '/mnt/lustre/panzheng/dataset/stereo_dataset/train_image', 'input image directory')
flags.DEFINE_integer('num_epoch', 10, 'number of epochs')
flags.DEFINE_integer('training_steps', 200000, 'num of training steps')
flags.DEFINE_integer('display_C_step', 10, 'show every num step')
flags.DEFINE_integer('C_snapshot', 1000, 'snapshot to save')
flags.DEFINE_integer('batch_size', 8, 'batch size') #2 for testing
flags.DEFINE_integer('height', 448, 'height')
flags.DEFINE_integer('width', 448, 'width')
flags.DEFINE_integer('height_re', 480, 'height resize')
flags.DEFINE_integer('width_re', 480, 'width resize')
flags.DEFINE_integer('num_pair', 100, 'number of pairs of images')
flags.DEFINE_float('lr', 1e-4, 'learning rate')

FLAGS = flags.FLAGS

#compute L2 distance between a vector pair
def L2dist(V1, V2):
    dist = np.sqrt(np.sum(np.square(V1 - V2)))
    return dist

def convert_depth(C1, C2, disp, fx):
    '''convert disparity map into depth map
    para:
        C1, C2: camera coordinate in world frame
        disp: disparity map
        fx: focal length fx
    return:
        depth map
    '''
    h, w = disp.shape
    depth = np.zeros(shape=(h, w))
    baseline = L2dist(np.array(C1), np.array(C2))
    # print 'fx:', fx
    # print 'baseline:', baseline
    for i in range(h):
        for j in range(w):
            depth[i][j] = (1.0 * fx * baseline) / (abs(disp[i][j]) + 1e-10)
    return depth

def warp_depth(dep, pos1, pos2):
    '''warp a depth map into a novel view
    para:
        dep: depth map
        pos1, pos2: camera pose. Format check for dataset camera parameter format
    return:
        depth map in novel view
    '''
    h, w = dep.shape
    cam1 = np.array([[float(pos1[7]), float(pos1[8]), float(pos1[9]), float(pos1[10])], [float(pos1[11]), float(pos1[12]), float(pos1[13]), float(pos1[14])], [float(pos1[15]), float(pos1[16]), float(pos1[17]), float(pos1[18])]])
    K1 = np.array([[w*float(pos1[1]), 0, w*float(pos1[3])], [0, h*float(pos1[2]), h*float(pos1[4])], [0, 0, 1]])
    R1 = np.array([[float(pos1[7]), float(pos1[8]), float(pos1[9])], [float(pos1[11]), float(pos1[12]), float(pos1[13])], [float(pos1[15]), float(pos1[16]), float(pos1[17])]])
    t1 = np.array([[float(pos1[10])], [float(pos1[14])], [float(pos1[18])]])

    cam2 = np.array([[float(pos2[7]), float(pos2[8]), float(pos2[9]), float(pos2[10])], [float(pos2[11]), float(pos2[12]), float(pos2[13]), float(pos2[14])], [float(pos2[15]), float(pos2[16]), float(pos2[17]), float(pos2[18])]])
    K2 = np.array([[w*float(pos2[1]), 0, w*float(pos2[3])], [0, h*float(pos2[2]), h*float(pos2[4])], [0, 0, 1]])
    R2 = np.array([[float(pos2[7]), float(pos2[8]), float(pos2[9])], [float(pos2[11]), float(pos2[12]), float(pos2[13])], [float(pos2[15]), float(pos2[16]), float(pos2[17])]])
    t2 = np.array([[float(pos2[10])], [float(pos2[14])], [float(pos2[18])]])

    R_12 = np.dot(inv(R1), R2)
    t_12 = np.dot(inv(R1), t2 - t1)
    filler = np.array([[0.0, 0.0, 0.0, 1.0]])
    pose = np.concatenate((R_12, t_12), axis=1)
    pose = np.concatenate((pose, filler), axis=0)

    x_row = np.arange(w)
    y_col = np.arange(h)
    X, Y = np.meshgrid(x_row, y_col)
    ones = np.full((h, w), 1)
    pixel_coord = np.stack((X, Y, ones), axis=0)
    depth = dep.reshape((1, -1))
    pixel_coord = pixel_coord.reshape((3, -1))
    cam_coord = np.matmul(inv(K1), pixel_coord) * depth
    ones1 = np.full((1, h*w), 1)
    cam_coord = np.concatenate((cam_coord, ones1), axis=0)
    cam_coord = cam_coord.reshape((-1, h, w))
    K = np.concatenate((K1, np.zeros(shape=(3,1))), axis=1)
    K = np.concatenate((K, filler), axis=0)
    proj = np.matmul(K, pose)
    cam_coord = cam_coord.reshape((4, -1))
    un_pixel_cor = np.matmul(proj, cam_coord)
    x_u = un_pixel_cor[0]
    y_u = un_pixel_cor[1]
    z_u = un_pixel_cor[2]
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    # print 'x_n:', x_n
    # print 'y_n:', y_n
    src_pix_cor = np.concatenate((x_n, y_n), axis=0)
    src_pix_cor = src_pix_cor.reshape((2, h, w))
    xcor = src_pix_cor[0]
    ycor = src_pix_cor[1]
    xcor = np.clip(xcor, 0, w - 1 - 1e-5)
    ycor = np.clip(ycor, 0, h - 1 - 1e-5)
    tgt_dep = np.zeros(dep.shape) + 1e-5
    tgt_weight = np.zeros(dep.shape) + 1e-5
    #bilinear
    for j in range(h):
        for k in range(w):
            x = xcor[j][k]
            y = ycor[j][k]
            x0 = np.floor(x).astype(int)
            y0 = np.floor(y).astype(int)
            x1 = np.floor(x).astype(int) + 1
            y1 = np.floor(y).astype(int) + 1
            del_x = x - x0
            del_y = y - y0

            val = dep[j][k]
            tgt_dep[y0][x0] += (1 - del_x) * (1 - del_y) * val
            tgt_dep[y0][x1] += del_x * (1 - del_y) * val
            tgt_dep[y1][x0] += (1 - del_x) * del_y * val
            tgt_dep[y1][x1] += del_x * del_y * val

            tgt_weight[y0][x0] += (1 - del_x) * (1 - del_y)
            tgt_weight[y0][x1] += del_x * (1 - del_y)
            tgt_weight[y1][x0] += (1 - del_x) * del_y
            tgt_weight[y1][x1] += del_x * del_y
    tgt_dep = tgt_dep / tgt_weight
    tgt_dep = np.clip(tgt_dep, 0, 255)

    # #nearest
    # for j in range(h):
    #     for k in range(w):
    #         x = xcor[j][k]
    #         y = ycor[j][k]
    #         x0 = np.floor(x).astype(int)
    #         y0 = np.floor(y).astype(int)
    #         x1 = np.floor(x).astype(int) + 1
    #         y1 = np.floor(y).astype(int) + 1
    #         del_x = x - x0
    #         del_y = y - y0
    #         val = dep[j][k]
    #         if del_x < 0.5 and del_y < 0.5:
    #             tgt_dep[y0][x0] = val
    #         if del_x < 0.5 and del_y >= 0.5:
    #             tgt_dep[y0][x1] = val
    #         if del_x >= 0.5 and del_y < 0.5:
    #             tgt_dep[y1][x0] = val
    #         if del_x >= 0.5 and del_y >= 0.5:
    #             tgt_dep[y1][x1] = val
    return tgt_dep

def dep_warp_for(img, dep, pos1, pos2):
    '''forward warp using depth
    para:
        img: input image
        dep: depth map correspond to input image
        pos1, pos2: camera pose. Format check for dataset camera parameter format
    return:
        warped image, relative camera pose between pos1 and pos2 in 4x4 matrix
    '''
    h, w, c = img.shape
    cam1 = np.array([[float(pos1[7]), float(pos1[8]), float(pos1[9]), float(pos1[10])], [float(pos1[11]), float(pos1[12]), float(pos1[13]), float(pos1[14])], [float(pos1[15]), float(pos1[16]), float(pos1[17]), float(pos1[18])]])
    K1 = np.array([[w*float(pos1[1]), 0, w*float(pos1[3])], [0, h*float(pos1[2]), h*float(pos1[4])], [0, 0, 1]])
    R1 = np.array([[float(pos1[7]), float(pos1[8]), float(pos1[9])], [float(pos1[11]), float(pos1[12]), float(pos1[13])], [float(pos1[15]), float(pos1[16]), float(pos1[17])]])
    t1 = np.array([[float(pos1[10])], [float(pos1[14])], [float(pos1[18])]])

    cam2 = np.array([[float(pos2[7]), float(pos2[8]), float(pos2[9]), float(pos2[10])], [float(pos2[11]), float(pos2[12]), float(pos2[13]), float(pos2[14])], [float(pos2[15]), float(pos2[16]), float(pos2[17]), float(pos2[18])]])
    K2 = np.array([[w*float(pos2[1]), 0, w*float(pos2[3])], [0, h*float(pos2[2]), h*float(pos2[4])], [0, 0, 1]])
    R2 = np.array([[float(pos2[7]), float(pos2[8]), float(pos2[9])], [float(pos2[11]), float(pos2[12]), float(pos2[13])], [float(pos2[15]), float(pos2[16]), float(pos2[17])]])
    t2 = np.array([[float(pos2[10])], [float(pos2[14])], [float(pos2[18])]])

    # R_21 = np.dot(inv(R2), R1)
    # t_21 = np.dot(inv(R2), t1 - t2)
    # filler = np.array([[0.0, 0.0, 0.0, 1.0]])
    # pose = np.concatenate((R_21, t_21), axis=1)
    # pose = np.concatenate((pose, filler), axis=0)
    R_12 = np.dot(inv(R1), R2)
    t_12 = np.dot(inv(R1), t2 - t1)
    filler = np.array([[0.0, 0.0, 0.0, 1.0]])
    pose = np.concatenate((R_12, t_12), axis=1)
    pose = np.concatenate((pose, filler), axis=0)

    x_row = np.arange(w)
    y_col = np.arange(h)
    X, Y = np.meshgrid(x_row, y_col)
    ones = np.full((h, w), 1)
    pixel_coord = np.stack((X, Y, ones), axis=0)
    depth = dep.reshape((1, -1))
    pixel_coord = pixel_coord.reshape((3, -1))
    cam_coord = np.matmul(inv(K1), pixel_coord) * depth
    ones1 = np.full((1, h*w), 1)
    cam_coord = np.concatenate((cam_coord, ones1), axis=0)
    cam_coord = cam_coord.reshape((-1, h, w))
    K = np.concatenate((K1, np.zeros(shape=(3,1))), axis=1)
    K = np.concatenate((K, filler), axis=0)
    proj = np.matmul(K, pose)
    cam_coord = cam_coord.reshape((4, -1))
    un_pixel_cor = np.matmul(proj, cam_coord)
    x_u = un_pixel_cor[0]
    y_u = un_pixel_cor[1]
    z_u = un_pixel_cor[2]
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    # print 'x_n:', x_n
    # print 'y_n:', y_n
    src_pix_cor = np.concatenate((x_n, y_n), axis=0)
    src_pix_cor = src_pix_cor.reshape((2, h, w))
    # test_src_pix_cor = np.transpose(src_pix_cor, [1,2,0])
    # print 'test_src_pix_cor:', test_src_pix_cor
    xcor = src_pix_cor[0]
    ycor = src_pix_cor[1]
    xcor = np.clip(xcor, 0, w - 1 - 1e-5)
    ycor = np.clip(ycor, 0, h - 1 - 1e-5)
    warped_img = (img - img).astype(float)
    warp_weight = np.zeros(img.shape) + 1e-5
    #bilinear
    for j in range(h):
        for k in range(w):
            x = xcor[j][k]
            y = ycor[j][k]
            x0 = np.floor(x).astype(int)
            y0 = np.floor(y).astype(int)
            x1 = np.floor(x).astype(int) + 1
            y1 = np.floor(y).astype(int) + 1
            del_x = x - x0
            del_y = y - y0
            val = img[j][k]

            warped_img[y0][x0] += (1 - del_x) * (1 - del_y) * val
            warped_img[y0][x1] += del_x * (1 - del_y) * val
            warped_img[y1][x0] += (1 - del_x) * del_y * val
            warped_img[y1][x1] += del_x * del_y * val\

            warp_weight[y0][x0] += (1 - del_x) * (1 - del_y)
            warp_weight[y0][x1] += del_x * (1 - del_y)
            warp_weight[y1][x0] += (1 - del_x) * del_y
            warp_weight[y1][x1] += del_x * del_y
    warped_img = warped_img / warp_weight
    warped_img = np.clip(warped_img, 0, 255)

    # #nearest
    # for j in range(h):
    #     for k in range(w):
    #         x = xcor[j][k]
    #         y = ycor[j][k]
    #         x0 = np.floor(x).astype(int)
    #         y0 = np.floor(y).astype(int)
    #         x1 = np.floor(x).astype(int) + 1
    #         y1 = np.floor(y).astype(int) + 1
    #         del_x = x - x0
    #         del_y = y - y0
    #         val = img[j][k]
    #         if del_x < 0.5 and del_y < 0.5:
    #             warped_img[y0][x0] = val
    #         if del_x < 0.5 and del_y >= 0.5:
    #             warped_img[y0][x1] = val
    #         if del_x >= 0.5 and del_y < 0.5:
    #             warped_img[y1][x0] = val
    #         if del_x >= 0.5 and del_y >= 0.5:
    #             warped_img[y1][x1] = val
    # print 'xcor:', xcor
    # print 'ycor:', ycor
    #warped_img = bilinear_interp(img, xcor, ycor)

    return warped_img, pose

def dep_warp_inv(img, dep, pos1, pos2):
    '''inverse warp using depth
    para:
        img: input image
        dep: depth map correspond to input image
        pos1, pos2: camera pose. Format check for dataset camera parameter format
    return:
        warped image, relative camera pose between pos2 and pos1 in 4x4 matrix
    '''
    h, w, c = img.shape
    cam1 = np.array([[float(pos1[7]), float(pos1[8]), float(pos1[9]), float(pos1[10])], [float(pos1[11]), float(pos1[12]), float(pos1[13]), float(pos1[14])], [float(pos1[15]), float(pos1[16]), float(pos1[17]), float(pos1[18])]])
    K1 = np.array([[w*float(pos1[1]), 0, w*float(pos1[3])], [0, h*float(pos1[2]), h*float(pos1[4])], [0, 0, 1]])
    R1 = np.array([[float(pos1[7]), float(pos1[8]), float(pos1[9])], [float(pos1[11]), float(pos1[12]), float(pos1[13])], [float(pos1[15]), float(pos1[16]), float(pos1[17])]])
    t1 = np.array([[float(pos1[10])], [float(pos1[14])], [float(pos1[18])]])

    cam2 = np.array([[float(pos2[7]), float(pos2[8]), float(pos2[9]), float(pos2[10])], [float(pos2[11]), float(pos2[12]), float(pos2[13]), float(pos2[14])], [float(pos2[15]), float(pos2[16]), float(pos2[17]), float(pos2[18])]])
    K2 = np.array([[w*float(pos2[1]), 0, w*float(pos2[3])], [0, h*float(pos2[2]), h*float(pos2[4])], [0, 0, 1]])
    R2 = np.array([[float(pos2[7]), float(pos2[8]), float(pos2[9])], [float(pos2[11]), float(pos2[12]), float(pos2[13])], [float(pos2[15]), float(pos2[16]), float(pos2[17])]])
    t2 = np.array([[float(pos2[10])], [float(pos2[14])], [float(pos2[18])]])

    R_21 = np.dot(inv(R2), R1)
    t_21 = np.dot(inv(R2), t1 - t2)
    filler = np.array([[0.0, 0.0, 0.0, 1.0]])
    pose = np.concatenate((R_21, t_21), axis=1)
    pose = np.concatenate((pose, filler), axis=0)

    x_row = np.arange(w)
    y_col = np.arange(h)
    X, Y = np.meshgrid(x_row, y_col)
    ones = np.full((h, w), 1)
    pixel_coord = np.stack((X, Y, ones), axis=0)
    depth = dep.reshape((1, -1))
    pixel_coord = pixel_coord.reshape((3, -1))
    cam_coord = np.matmul(inv(K1), pixel_coord) * depth
    ones1 = np.full((1, h*w), 1)
    cam_coord = np.concatenate((cam_coord, ones1), axis=0)
    cam_coord = cam_coord.reshape((-1, h, w))
    K = np.concatenate((K1, np.zeros(shape=(3,1))), axis=1)
    K = np.concatenate((K, filler), axis=0)
    proj = np.matmul(K, pose)
    cam_coord = cam_coord.reshape((4, -1))
    un_pixel_cor = np.matmul(proj, cam_coord)
    x_u = un_pixel_cor[0]
    y_u = un_pixel_cor[1]
    z_u = un_pixel_cor[2]
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    # print 'x_n:', x_n
    # print 'y_n:', y_n
    src_pix_cor = np.concatenate((x_n, y_n), axis=0)
    src_pix_cor = src_pix_cor.reshape((2, h, w))
    src_pix_cor = np.transpose(src_pix_cor, [1,2,0])
    #print 'src_pix_cor:', src_pix_cor
    F = []
    for i in range(c):
        #f = interpolate.interp2d(x_row, y_col, img[:, :, i]) #bilinear interpolation
        f = interpolate.RectBivariateSpline(x_row, y_col, img[:, :, i]) #cubic interpolation
        F.append(f)

    for i in range(c):
        f = F[i]
        #print(f(1.5,2.5))
        if i == 0:
            warped_img = img[:, :, 0:1] - img[:, :, 0:1] #initialize to zero
            for j in range(len(y_col)):
                for k in range(len(x_row)):
                    #print('warpwarp')
                    #warped_img[j][k][0] = f(src_pix_cor[j][k][0], src_pix_cor[j][k][1])[0] #bilinear interpolation
                    warped_img[j][k][0] = f(src_pix_cor[j][k][1], src_pix_cor[j][k][0])[0][0] #cubic interpolation
        else:
            tmp_img = img[:, :, 0:1] - img[:, :, 0:1] #initialize to zero
            for j in range(len(y_col)):
                for k in range(len(x_row)):
                    #tmp_img[j][k][0] = f(src_pix_cor[j][k][0], src_pix_cor[j][k][1])[0] #bilinear interpolation
                    tmp_img[j][k][0] = f(src_pix_cor[j][k][1], src_pix_cor[j][k][0])[0][0] #cubic interpolation
            warped_img = np.concatenate([warped_img, tmp_img], axis=2)
    return warped_img, pose

def leaky_relu(_input, negative_slope=0.1):
    return _input * negative_slope + tf.nn.relu(_input) * (1 - negative_slope)

def infer_net(concat_img, GT, _weights, _biases):
    '''network
    para:
        concat_img: concatenated channels of inputs
        GT: ground truth image
        _weights, _biases: weights and biases for network
    return:
        output image, l1 loss
    '''

    input_shape = concat_img.get_shape().as_list()
    input_flat = tf.reshape(concat_img, [input_shape[0], -1])
    dim_in_flat = input_flat.get_shape()[1].value

    rgb_shape = [input_shape[0], input_shape[1], input_shape[2], 3]
    dim_rgb_flat = input_shape[1] * input_shape[2] * 3

    #conv_C1
    with tf.variable_scope('conv_C1') as scope:
        conv = tf.nn.bias_add(tf.nn.conv2d(concat_img, _weights['conv_filter_C1_w1'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C1_b1'])
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C1_w2'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C1_b2'])
        conv_C1 = leaky_relu(conv, 0.1)

    #conv_C2
    with tf.variable_scope('conv_C2') as scope:
        conv = tf.nn.max_pool(conv_C1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C2_w1'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C2_b1'])
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C2_w2'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C2_b2'])
        conv_C2 = leaky_relu(conv, 0.1)

    #conv_C3
    with tf.variable_scope('conv_C3') as scope:
        conv = tf.nn.max_pool(conv_C2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C3_w1'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C3_b1'])
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C3_w2'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C3_b2'])
        conv_C3 = leaky_relu(conv, 0.1)

    #conv_C4
    with tf.variable_scope('conv_C4') as scope:
        conv = tf.nn.max_pool(conv_C3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C4_w1'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C4_b1'])
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C4_w2'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C4_b2'])
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C4_w3'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C4_b3'])
        conv_C4 = leaky_relu(conv, 0.1)

    #conv_C5
    with tf.variable_scope('conv_C5') as scope:
        conv = tf.nn.conv2d_transpose(conv_C4, _weights['conv_filter_C5_w1'], [input_shape[0], 112, 112, 112], [1, 2, 2, 1], padding='SAME')
        skip = tf.concat([conv, conv_C3], axis=3)
        conv = tf.nn.bias_add(tf.nn.conv2d(skip, _weights['conv_filter_C5_w2'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C5_b2'])
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C5_w3'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C5_b3'])
        conv_C5 = leaky_relu(conv, 0.1)

    #conv_C6
    with tf.variable_scope('conv_C6') as scope:
        conv = tf.nn.conv2d_transpose(conv_C5, _weights['conv_filter_C6_w1'], [input_shape[0], 224, 224, 112], [1, 2, 2, 1], padding='SAME')
        skip = tf.concat([conv, conv_C2], axis=3)
        conv = tf.nn.bias_add(tf.nn.conv2d(skip, _weights['conv_filter_C6_w2'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C6_b2'])
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C6_w3'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C6_b3'])
        conv_C6 = leaky_relu(conv, 0.1)

    with tf.variable_scope('conv_C7') as scope:
        conv = tf.nn.conv2d_transpose(conv_C6, _weights['conv_filter_C7_w1'], [input_shape[0], 448, 448, 56], [1, 2, 2, 1], padding='SAME')
        skip = tf.concat([conv, conv_C1], axis=3)
        conv = tf.nn.bias_add(tf.nn.conv2d(skip, _weights['conv_filter_C7_w2'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C7_b2'])
        conv = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C7_w3'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C7_b3'])
        conv_C7 = leaky_relu(conv, 0.1)

    with tf.variable_scope('conv_C8') as scope:
        conv = tf.nn.bias_add(tf.nn.conv2d(conv_C7, _weights['conv_filter_C8_w1'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C8_b1'])
        conv_C8 = tf.nn.bias_add(tf.nn.conv2d(conv, _weights['conv_filter_C8_w2'], [1, 1, 1, 1], padding='SAME'), _biases['conv_filter_C8_b2'])

    out_img = tf.reshape(conv_C8, rgb_shape)

    with tf.variable_scope('loss') as scope:
        num_ele = tf.shape(out_img)[0] * tf.shape(out_img)[1] * tf.shape(out_img)[2] * tf.shape(out_img)[3]
        num_ele = tf.to_float(num_ele)
        # l2_loss = tf.nn.l2_loss(tf.abs(GT - out_img))
        # l2_loss = tf.div(l2_loss, num_ele)
        l1_loss = tf.reduce_sum(tf.abs(GT - out_img))
        l1_loss = tf.div(l1_loss, num_ele)
    return out_img, l1_loss

def main(_):

    #tf_model part.
    reg_lambda = 1e-3
    learning_rate = 1e-4
    training_iters = 200000  # 200000
    #batch_size = 2  # 16
    batch_size = 1 #for result
    display_step = 50
    snapshot = 5000
    width = 448
    height = 448
    width_res = 480  # LS:resize to this size
    height_res = 480
    width_orig = 640
    height_orig = 640
    #path_to_model_save = './models/Dapangzi6_deploy_t1'  # save the model weights
    path_to_model_save = '/mnt/lustre/panzheng/pyrun/tf_model/savemodel'
    #path_to_model_save = '/home/SENSETIME/panzheng/Documents/Run/tf_model/savemodel'
    #patn_to_model_load = '/mnt/lustre/sunliusheng/SPN_train/tf_base_model/dapangzi6_deploy_model'  # load the model weights
    patn_to_model_load = '/mnt/lustre/panzheng/pyrun/tf_model/train/dapangzi6_model'
    #patn_to_model_load = '/home/SENSETIME/panzheng/Documents/Run/tf_model/test/dapangzi6_deploy_model'
    #patn_to_log = './logs_SPN/Dapangzi6_deploy_t1/'  # for tensorboard logs
    patn_to_log = '/mnt/lustre/panzheng/pyrun/tf_model/log/'
    #patn_to_log = '/home/SENSETIME/panzheng/Documents/Run/tf_model/log/'
    # path_to_data = '/mnt/lustre/share/disp_data/hybrid_dataset_v3/'
    # interm = 'th70_'
    # file_train = path_to_data + 'hybrid_dataset_v2/hybrid_dataset_v5_' + interm + 'largeimage_TRAIN.list'
    # file_test = path_to_data + 'hybrid_dataset_v2/hybrid_dataset_v2_' + interm + 'TEST.list'
    # list_train_file = open(file_train, 'rb')
    # list_train = [[], [], []]
    # with open(file_train, 'r') as openedfile:
    #     for line in openedfile:
    #         s = line.split()
    #         list_train[0].append(s[0])
    #         list_train[1].append(s[1])
    #         list_train[2].append(s[2])
    # n_train = len(list_train[0])
    # perm_train = np.random.permutation(n_train)
    # list_train_file.close()

    print('main started...')

    #read pose data
    path_to_pose = FLAGS.path_to_pose

    #functions for naturally sorting file names
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(t) for t in re.split('(\d+)', text)]

    image_count = [] #count images in each corresponding txt/video
    path_to_pose = FLAGS.path_to_pose
    pose_file = os.listdir(path_to_pose)
    pose_file.sort(key=natural_keys)
    #pose_file = pose_file[:10]
    tmp_cam_cor = [[] for i in range(len(pose_file))] #camera coordinate in world frame
    tmp_cam_mat = [[] for i in range(len(pose_file))] #camera matrix data
    #load camera data
    count = 0
    for file in pose_file:
        data = []
        file_path = path_to_pose + '/' + file
        with open(file_path, 'r') as f:
            for line in f:
                data.append(line)
        for i in range(len(data)):
            tmp_list = data[i].split()
            tmp_cam_mat[count].append(tmp_list)
            R = np.array([[float(tmp_list[7]), float(tmp_list[8]), float(tmp_list[9])], [float(tmp_list[11]), float(tmp_list[12]), float(tmp_list[13])], [float(tmp_list[15]), float(tmp_list[16]), float(tmp_list[17])]])
            t = np.array([[float(tmp_list[10])], [float(tmp_list[14])], [float(tmp_list[18])]])
            C = - np.dot(R.T, t)
            tmp_cam_cor[count].append([C[0][0], C[1][0], C[2][0]])
        count += 1
        image_count.append(len(data) - 1)

    print('pose data created...')

    #read image data
    #load image file name [[src1.jpg], [src2.jpg], [gt.jpg]]

    path_to_image = FLAGS.path_to_image
    image_file = os.listdir(path_to_image)
    image_file.sort(key=natural_keys)
    tmp_doc = []
    for imf in image_file:
        imfile = os.listdir(path_to_image + '/' + imf)
        imfile.sort(key=natural_keys)
        for imn in imfile:
            tmp_doc.append(path_to_image + '/' + imf + '/' + imn)
    tmp_image_doc = []
    c = 0
    for i in range(len(tmp_cam_cor)):
        tmp_image_doc.append(tmp_doc[c:c + len(tmp_cam_cor[i])])
        c = c + len(tmp_cam_cor[i])

    cam_cor = []
    image_doc = []
    cam_mat = []
    for i in range(len(tmp_cam_cor)):
        if len(tmp_cam_cor[i]) >= 3:
            cam_cor.append(tmp_cam_cor[i])
            cam_mat.append(tmp_cam_mat[i])
        if len(tmp_image_doc[i]) >= 3:
            image_doc.append(tmp_image_doc[i])

    print('image data created...')
    print(len(cam_cor))
    #load sample images & poses
    list_train = [[], [], []] #for disparity net
    #image_list = [[], [], []]
    sample_cam_pose = [[], [], []]
    sample_cam_mat = [[], [], []]
    for i in range(len(cam_cor)):
        sign = True #ensure that 3 chosen images are different
        while sign:
            index = np.random.randint(0, len(cam_cor[i]), size=3)
            s_ind = []
            for j in range(3):
                s_ind.append(index[j])
            s_ind.sort()
            if (index[0] != index[1]) & (index[0] != index[2]) & (index[1] != index[2]):
                if (s_ind[2] - s_ind[0] <= 40): #ensure that sample frames are not too far away from each other
                    sign = False
        tmpi1 = image_doc[i][index[0]]
        tmpi2 = image_doc[i][index[1]]
        tmpp1 = cam_cor[i][index[0]]
        tmpp2 = cam_cor[i][index[1]]
        tmpm1 = cam_mat[i][index[0]]
        tmpm2 = cam_mat[i][index[1]]
        if tmpp1[0] < tmpp2[0]: #ensure for left and right view point
            img1 = tmpi1
            img2 = tmpi2
            list_train[0].append(img1)
            list_train[1].append(img2)

            pose1 = tmpp1
            pose2 = tmpp2
            sample_cam_pose[0].append(pose1)
            sample_cam_pose[1].append(pose2)
            mat1 = tmpm1
            mat2 = tmpm2
            sample_cam_mat[0].append(mat1)
            sample_cam_mat[1].append(mat2)
        else:
            img1 = tmpi2
            img2 = tmpi1
            list_train[0].append(img1)
            list_train[1].append(img2)

            pose1 = tmpp2
            pose2 = tmpp1
            sample_cam_pose[0].append(pose1)
            sample_cam_pose[1].append(pose2)
            mat1 = tmpm2
            mat2 = tmpm1
            sample_cam_mat[0].append(mat1)
            sample_cam_mat[1].append(mat2)
        tgt_img = image_doc[i][index[2]]
        list_train[2].append(tgt_img)
        tgt_pose = cam_cor[i][index[2]]
        sample_cam_pose[2].append(tgt_pose)
        tgt_mat = cam_mat[i][index[2]]
        sample_cam_mat[2].append(tgt_mat)

    n_train = len(list_train[0])
    seq_train = np.array([i for i in range(n_train)])
    perm_train = np.random.permutation(n_train)

    print('data list created...')
    print('n_train = ', n_train)

    disp = []
    disparity = []

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

        i_L = tf.placeholder(tf.float32, shape=[height, width, 1])
        i_R = tf.placeholder(tf.float32, shape=[height, width, 1])
        i_D = tf.placeholder(tf.float32, shape=[height, width, 1])
        q = tf.FIFOQueue(80, [tf.float32, tf.float32, tf.float32],
                         shapes=[[height, width, 1], [height, width, 1], [height, width, 1]])
        enqueue_op = q.enqueue([i_L, i_R, i_D])
        L, R, D = q.dequeue_many(batch_size)
        predict_flow1, loss_disp = disp_net_dapangzi6(L, R, D, weights, bias)

        # all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='canshupf1')
        trainable_v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tf.summary.image('L', L)
        tf.summary.image('D', D)
        tf.summary.image('predict_flow1', predict_flow1)

        # with tf.variable_scope('loss'):
        #     tf.summary.scalar('loss disp', loss_disp)

        with tf.variable_scope('summary'):
            summary_op = tf.summary.merge_all()
        #l_rate_decay = tf.train.exponential_decay(learning_rate, training_iters, 4000, 0.96, staircase=True)
        #optimizer = tf.train.AdamOptimizer(learning_rate=l_rate_decay, epsilon=1e-4).minimize(loss_disp)

    writer = tf.summary.FileWriter(patn_to_log, graph)
    saver = tf.train.Saver(all_variables)  # all_variables)

    with tf.Session(graph=graph) as sess:
        #    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        tf.global_variables_initializer().run()
        saver.restore(sess, patn_to_model_load)
        print('Loaded variables:')
        print(all_variables)
        step = 0
        print('Trainable variables:')
        print(trainable_v)

        def load_and_enqueue():
            step = 0
            while True:  # step < 10:
                idx = seq_train[step % n_train]
                #disp_gt, s, ttype = load_pfm(path_to_data + list_train[2][idx])
                #disp_gt = -disp_gt * 32.
                # x = np.random.random_integers(0, width_res - width - 1)
                # y = np.random.random_integers(0, height_res - height - 1)
                #imgL = rgb2gray(misc.imread(path_to_data + list_train[0][idx]))
                #imgR = rgb2gray(misc.imread(path_to_data + list_train[1][idx]))
                imgL = rgb2gray(misc.imread(list_train[0][idx]))
                imgR = rgb2gray(misc.imread(list_train[1][idx]))
                imgL = misc.imresize(imgL, (height, width), interp='bilinear')
                imgR = misc.imresize(imgR, (height, width), interp='bilinear')
                #disp_gt = misc.imresize(disp_gt, (height_res, width_res), interp='nearest', mode='F')
                #disp_gt = disp_gt * (width_res * 1. / width_orig)
                imgL = np.expand_dims(imgL, axis=2)
                imgR = np.expand_dims(imgR, axis=2)
                #disp_gt = np.expand_dims(disp_gt[y: y + height, x: x + width], axis=2)
                zero = imgR - imgR
                sess.run(enqueue_op, feed_dict={i_L: imgL, i_R: imgR, i_D: zero})
                step += 1

        t = threading.Thread(target=load_and_enqueue)
        t.start()
        t_start = time.time()

        print('\nstep:' + '  ' + str(step))
        #while step < training_iters:
        while step < n_train: #pz
            conv0_re = tf.get_default_graph().get_tensor_by_name('canshupf1/Variable:0')
            l1, summary_, flow1, conv0_ = sess.run([loss_disp, summary_op, predict_flow1, conv0_re])
            flow1 = flow1 / 32.0
            disp.append(flow1.reshape((height, width)))  # load disparity for warping pz
            disparity.append(flow1.reshape((height, width, 1)))

            step += 1

    # print('len(disp) = ', len(disp))
    # print(disp[0])

    #color predicting part
    src_image1_pos = sample_cam_pose[0]
    src_image2_pos = sample_cam_pose[1]
    tgt_image_pos = sample_cam_pose[2]

    print('Warping images...')
    concat_image_list = []
    for i in range(n_train):
        img1 = misc.imread(list_train[0][i])
        img1 = misc.imresize(img1, (FLAGS.height, FLAGS.width), interp='bilinear')
        img2 = misc.imread(list_train[1][i])
        img2 = misc.imresize(img2, (FLAGS.height, FLAGS.width), interp='bilinear')
        # warp1 = warp_image(img1, disp[i], delta_x1, delta_y1, delta_x2)
        # warp2 = warp_image(img2, disp[i], delta_x2, delta_y2, delta_x1)
        depth1 = convert_depth(src_image1_pos[i], src_image2_pos[i], disp[i], FLAGS.width * float(sample_cam_mat[0][i][1]))
        depth2 = warp_depth(depth1, sample_cam_mat[0][i], sample_cam_mat[1][i])
        #print 'three:', depth2
        #depth_tgt = warp_depth(depth1, sample_cam_mat[0][i], sample_cam_mat[2][i])
        print('warping image pair ', i)
        warp1, tempos1 = dep_warp_for(img1, depth1, sample_cam_mat[0][i], sample_cam_mat[2][i])
        warp2, tempos2 = dep_warp_for(img2, depth2, sample_cam_mat[1][i], sample_cam_mat[2][i])
        # warp1, tempos1 = dep_warp_inv(img1, depth_tgt, sample_cam_mat[0][i], sample_cam_mat[2][i])
        # warp2, tempos2 = dep_warp_inv(img2, depth_tgt, sample_cam_mat[1][i], sample_cam_mat[2][i])
        pos1 = np.zeros((FLAGS.height, FLAGS.width)) + 1.0
        pos2 = np.zeros((FLAGS.height, FLAGS.width)) + 1.0
        pos1[:tempos1.shape[0], :tempos1.shape[1]] = tempos1
        pos2[:tempos2.shape[0], :tempos2.shape[1]] = tempos2
        pos1 = np.expand_dims(pos1, axis=2)
        pos2 = np.expand_dims(pos2, axis=2)
        # sobelx1 = cv2.Sobel(img1[:,:,0], cv2.CV_64F, 1, 0, ksize=5)      #may use image gradients to learn occluded structure
        # sobely1 = cv2.Sobel(img1[:,:,0], cv2.CV_64F, 0, 1, ksize=5)
        # sobelx2 = cv2.Sobel(img2[:,:,0], cv2.CV_64F, 1, 0, ksize=5)
        # sobely2 = cv2.Sobel(img2[:,:,0], cv2.CV_64F, 0, 1, ksize=5)
        # sobelx1 = np.expand_dims(sobelx1, axis=2)
        # sobely1 = np.expand_dims(sobely1, axis=2)
        # sobelx2 = np.expand_dims(sobelx2, axis=2)
        # sobely2 = np.expand_dims(sobely2, axis=2)
        # tgt_i = misc.imread(list_train[2][i])
        # tgt_i = misc.imresize(tgt_i, (FLAGS.height, FLAGS.width), interp='bilinear')
        tmp_pair = np.concatenate((img1, img2, warp1, warp2, disparity[i], pos1, pos2), axis=2)
        #tmp_pair = np.concatenate((img1, img2, warp1, warp2, disparity[i], pos1, pos2, sobelx1, sobely1, sobelx2, sobely2), axis=2)
        concat_image_list.append(tmp_pair)

    print('Finished concatenating...')

    tgt_image_list = []
    for i in range(len(list_train[2])):
        tmpi = misc.imread(list_train[2][i])
        tgt_image_list.append(misc.imresize(tmpi, (FLAGS.height, FLAGS.width), interp='bilinear'))

    print('Entering color predict part...')

    #color predict part

    path_to_C_log = '/mnt/lustre/panzheng/pyrun/view_syn/log_v5/'
    #path_to_C_log = '/home/SENSETIME/panzheng/Documents/pz/view_syn/log_v2/'
    #path_to_C_model_load = 'need_fill'
    path_to_C_model_save = '/mnt/lustre/panzheng/pyrun/view_syn/save_v5/'
    #path_to_C_model_save = '/home/SENSETIME/panzheng/Documents/pz/view_syn/save'

    C_graph = tf.Graph()
    with C_graph.as_default():
        with tf.variable_scope('parameters'):
            weights = {
                'conv_filter_C1_w1': tf.get_variable('conv_filter_C1_w1', shape=[5, 5, 15, 28], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C1_w2': tf.get_variable('conv_filter_C1_w2', shape=[5, 5, 28, 28], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C2_w1': tf.get_variable('conv_filter_C2_w1', shape=[5, 5, 28, 56], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C2_w2': tf.get_variable('conv_filter_C2_w2', shape=[5, 5, 56, 56], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C3_w1': tf.get_variable('conv_filter_C3_w1', shape=[3, 3, 56, 112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C3_w2': tf.get_variable('conv_filter_C3_w2', shape=[3, 3, 112, 112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C4_w1': tf.get_variable('conv_filter_C4_w1', shape=[3, 3, 112, 112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C4_w2': tf.get_variable('conv_filter_C4_w2', shape=[3, 3, 112, 112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C4_w3': tf.get_variable('conv_filter_C4_w3', shape=[3, 3, 112, 112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C5_w1': tf.get_variable('conv_filter_C5_w1', shape=[3, 3, 112, 112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C5_w2': tf.get_variable('conv_filter_C5_w2', shape=[3, 3, 224, 112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C5_w3': tf.get_variable('conv_filter_C5_w3', shape=[3, 3, 112, 112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C6_w1': tf.get_variable('conv_filter_C6_w1', shape=[3, 3, 112, 112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C6_w2': tf.get_variable('conv_filter_C6_w2', shape=[5, 5, 168, 56], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C6_w3': tf.get_variable('conv_filter_C6_w3', shape=[5, 5, 56, 56], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C7_w1': tf.get_variable('conv_filter_C7_w1', shape=[5, 5, 56, 56], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C7_w2': tf.get_variable('conv_filter_C7_w2', shape=[5, 5, 84, 28], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C7_w3': tf.get_variable('conv_filter_C7_w3', shape=[5, 5, 28, 28], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C8_w1': tf.get_variable('conv_filter_C8_w1', shape=[5, 5, 28, 28], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C8_w2': tf.get_variable('conv_filter_C8_w2', shape=[5, 5, 28, 3], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            }
            biases = {
                'conv_filter_C1_b1': tf.get_variable('conv_filter_C1_b1', shape=[28], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C1_b2': tf.get_variable('conv_filter_C1_b2', shape=[28], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C2_b1': tf.get_variable('conv_filter_C2_b1', shape=[56], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C2_b2': tf.get_variable('conv_filter_C2_b2', shape=[56], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C3_b1': tf.get_variable('conv_filter_C3_b1', shape=[112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C3_b2': tf.get_variable('conv_filter_C3_b2', shape=[112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C4_b1': tf.get_variable('conv_filter_C4_b1', shape=[112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C4_b2': tf.get_variable('conv_filter_C4_b2', shape=[112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C4_b3': tf.get_variable('conv_filter_C4_b3', shape=[112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C5_b2': tf.get_variable('conv_filter_C5_b2', shape=[112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C5_b3': tf.get_variable('conv_filter_C5_b3', shape=[112], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C6_b2': tf.get_variable('conv_filter_C6_b2', shape=[56], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C6_b3': tf.get_variable('conv_filter_C6_b3', shape=[56], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C7_b2': tf.get_variable('conv_filter_C7_b2', shape=[28], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C7_b3': tf.get_variable('conv_filter_C7_b3', shape=[28], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C8_b1': tf.get_variable('conv_filter_C8_b1', shape=[28], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'conv_filter_C8_b2': tf.get_variable('conv_filter_C8_b2', shape=[3], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            }

        tmp_input = tf.placeholder(tf.float32, shape=[FLAGS.height, FLAGS.width, 15])
        tmp_gt = tf.placeholder(tf.float32, shape=[FLAGS.height, FLAGS.width, 3])
        C_q = tf.FIFOQueue(80, [tf.float32, tf.float32], shapes=[[FLAGS.height, FLAGS.width, 15], [FLAGS.height, FLAGS.width, 3]])
        enq_op = C_q.enqueue([tmp_input, tmp_gt])
        concat_input, syn_gt = C_q.dequeue_many(FLAGS.batch_size)
        syn_image, loss = infer_net(concat_input, syn_gt, weights, biases)

        all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='parameters')
        trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tf.summary.image('syn_image', syn_image)

        with tf.variable_scope('C_loss'):
            tf.summary.scalar('loss', loss)

        with tf.variable_scope('C_summary'):
            C_summary_op = tf.summary.merge_all()

        C_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(loss)

    C_writer = tf.summary.FileWriter(path_to_C_log, C_graph)
    C_saver = tf.train.Saver(all_var)

    with tf.Session(graph = C_graph) as sess:
        tf.global_variables_initializer().run()
        #C_saver.restore(sess, path_to_C_model_load)
        print 'test:', n_train, len(sample_cam_mat[0])
        print('Loaded C_variables:')
        print(all_var)
        print('Trainable C_variables:')
        print(trainable_var)

        #dataloader
        def C_load_and_enqueue():
            step = 0
            while True:  # step < 10:
                idx = perm_train[(step - 1) % n_train]
                cil = np.array(concat_image_list)
                til = np.array(tgt_image_list)
                sess.run(enq_op, feed_dict={tmp_input: cil[idx], tmp_gt: til[idx]})
                step += 1

        t = threading.Thread(target=C_load_and_enqueue)
        t.start()
        #time_start = time.time()

        print('ready to train...')
        C_step = 0
        time_start = time.time()
        while C_step < FLAGS.training_steps:
            if (C_step == 0 or C_step % FLAGS.display_C_step == 0):
                l, C_summary_, syn_img = sess.run([loss, C_summary_op, syn_image])
                print('syn_img:')
                print syn_image

                C_writer.add_summary(C_summary_, C_step)
                print('Iter ' + str(C_step) + '\nL: ' + str(l) + '\n')

            # if C_step % 10 == 1:
            #     time_start = time.time()
            sess.run(C_optimizer)
            # if C_step % 10 == 0:
            #     print('10 iterations need time: %4.4f' % (time.time() - time_start))
            #     time_start = time.time()
            if C_step % FLAGS.C_snapshot == 0:
                C_saver.save(sess, path_to_C_model_save, global_step=C_step)
            	print('saved parameters...')
            C_step += 1

        # training_op = one_step_train(next_concat, next_tgt, weights, biases)
        # with tf.train.MonitoredTrainingSession() as mon_sess:
        #     tf.global_variables_initializer().run()
        #     while not mon_sess.should_stop():
        #         mon_sess.run(training_op)

    print('Finished!')

if __name__ == '__main__':
    tf.app.run()
