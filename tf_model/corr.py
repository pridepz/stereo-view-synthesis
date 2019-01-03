import tensorflow as tf
import tensorflow.contrib.keras as kr

def normalize_c(arr, epsilon = 1e-9):
    with tf.variable_scope('mvn'):
        mean, var = tf.nn.moments(arr, -1, keep_dims=True)
        return ( ( arr - mean ) / ( tf.sqrt(var) + epsilon ) )

def corr1d(left, right, l_disp=70, r_disp=10):
    with tf.variable_scope('correlation'):
        n, h, w, c = left.get_shape()

        corr_list = []
        for i in xrange(-l_disp, r_disp+1):
            offset, target = max(0, i), w-abs(i)
            crop = tf.image.crop_to_bounding_box(right, 0, offset, h, target)

            p_offset = -min(0, i)
            padding = tf.image.pad_to_bounding_box(crop, 0, p_offset, h, w)

            corr_single = tf.reduce_mean( left * padding, axis=3, keep_dims=True )
            corr_list.append(corr_single)
        corr = kr.layers.Concatenate()(corr_list)
    return corr

def _test():
    n, h, w, c = 1, 96, 96, 32
    left = tf.placeholder(tf.float16, name='left', shape=(n,h,w,c))
    right = tf.placeholder(tf.float16, name='right', shape=(n,h,w,c))

    config = tf.ConfigProto(device_count={'GPU': 0})

    left_n = normalize_c(left, 1e-9)
    right_n = normalize_c(right, 1e-9)

    corr = corr1d(left_n, right_n)

    # mvn
    left_feed = np.reshape(np.transpose( sio.loadmat('in.mat')['in'], [1,2,0] ), [n,h,w,c])
    o_ans = np.reshape(np.transpose( sio.loadmat('out.mat')['out'], [1,2,0] ), [n,h,w,c])
    with tf.Session(config=config) as sess:
        norm_out = sess.run(left_n, {left: left_feed, right: left_feed})
    print np.sum(np.abs(norm_out - o_ans))

    # corr

    # left_feed = np.array( [ [ [ [1,2,3] for i in xrange(1, 11) ] ]*10 ], dtype=np.float16 )
    # right_feed = np.array( [ [ [ [1,2,3] for i in xrange(1, 11) ] ]*10 ], dtype=np.float16 )
    left_feed = np.transpose(np.expand_dims(sio.loadmat('in_l.mat')['in_l'], axis=0), [0,2,1,3])
    right_feed = np.transpose(np.expand_dims(sio.loadmat('in_r.mat')['in_r'], axis=0), [0,2,1,3])
    o_ans = np.transpose(np.expand_dims(sio.loadmat('out_corr.mat')['out'], axis=0), [0,2,1,3])

    with tf.Session(config=config) as sess:
        with tf.variable_scope('correlation'):
            corr_out = sess.run([corr], {left: left_feed, right: right_feed})

    # print np.transpose(corr_out, [0, 3, 1, 2])
    print o_ans
    print corr_out
    print np.sum(np.abs(o_ans - corr_out))

if __name__ == '__main__':
    import scipy.io as sio
    import numpy as np
    _test()
