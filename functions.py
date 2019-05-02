import tensorflow as tf
import numpy as np
from sklearn import preprocessing

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def extract(input, n_fea, time_window, moving, n_classes):
    xx = input[:, :n_fea]
    xx = preprocessing.scale(xx)  # z-score normalization
    yy = input[:, n_fea:n_fea+1]
    new_x = []
    new_y = []
    number = int((xx.shape[0]/moving)-1)
    for i in range(number):
        ave_y = np.average(yy[(i * moving):(i * moving + time_window)])
        if ave_y in range(n_classes+1):
            new_x.append(xx[(i * moving):(i * moving + time_window), :])
            new_y.append(ave_y)
        else:
            new_x.append(xx[(i * moving):(i * moving + time_window), :])
            new_y.append(0)

    new_x = np.array(new_x)
    new_x = new_x.reshape([-1, n_fea * time_window])
    new_y = np.array(new_y)
    new_y.shape =[new_y.shape[0], 1]
    data = np.hstack((new_x, new_y))
    # data = np.vstack((data[0], data))
    # print new_y.shape
    return new_x, new_y, data

# def batch_split(feature, label, n_group, batch_size):
#     a = feature
#     train_fea = []
#     train_label = []
#
#     for i in range(n_group):
#         f = a[(0 + batch_size * i):(batch_size + batch_size * i)]
#         train_fea.append(f)
#
#     for i in range(n_group):
#         f = label[(0 + batch_size * i):(batch_size + batch_size * i), :]
#         train_label.append(f)
#     return train_fea, train_label

# the CNN code
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    print type(x), type(W)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_1x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1, 2,1], padding='SAME')

def conv(x_image, depth_1):
    W_conv1 = weight_variable([2, 2, 1, depth_1]) # patch 5x5, in size is 1, out size is 8
    b_conv1 = bias_variable([depth_1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 1*64*2
    # h_pool1 = max_pool_1x2(h_conv1)                          # output size 1*32x2
    return h_conv1

def fc(data, input_shape, output_size):
    size2 = output_size
    W_fc1 = weight_variable([input_shape, size2])
    b_fc1 = bias_variable([size2])
    h_pool2_flat = tf.reshape(data, [-1, input_shape])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return h_fc1
def fc_nosig(data, input_shape, output_size):
    size2 = output_size
    W_fc1 = weight_variable([input_shape, size2])
    b_fc1 = bias_variable([size2])
    h_pool2_flat = tf.reshape(data, [-1, input_shape])
    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return h_fc1

def fc_relu(data, input_shape, output_size):
    size2 = output_size
    W_fc1 = weight_variable([input_shape, size2])
    b_fc1 = bias_variable([size2])
    h_pool2_flat = tf.reshape(data, [-1, input_shape])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return h_fc1
def fc_tanh(data, input_shape, output_size):
    size2 = output_size
    W_fc1 = weight_variable([input_shape, size2])
    b_fc1 = bias_variable([size2])
    h_pool2_flat = tf.reshape(data, [-1, input_shape])
    h_fc1 = tf.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return h_fc1


