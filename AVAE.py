import tensorflow as tf
import scipy.io as sc
import numpy as np
import pickle
import random
import time
from functions import *
from VAE_plus_plus import *
from sklearn.metrics import classification_report
from scipy import stats
from sklearn.metrics import accuracy_score
import os


"""Dataset loading. All the datasets are public available.
PAMAP2: http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
TUH: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
MNIST: http://yann.lecun.com/exdb/mnist/
Yelp: https://www.yelp.com/dataset
Here we take the PAMAP2 dataset as an example.
"""

""""PAMAP2: Here we use a subset of PAMAP2, totally contains 120,000 samples for 5 subjects (the more subjects,
the better but more computational. We select 5 most commonly used activities (Cycling, standing, walking, 
lying, and running, labelled from 0 to4) as a subset for evaluation.12,000 samples for each sub, 51 features.
The dataset is huge, using small subset for debugging is strongly recommended."""

feature = sc.loadmat("/home/xiangzhang/scratch/AR_6p_8c.mat")
data = feature['AR_6p_8c']
data = np.concatenate((data[0:120000], data[200000*1:200000*1+120000],  data[200000*2:200000*2+120000],
                      data[200000*3:200000*3+120000], data[200000*4:200000*4+120000]
                      , ), axis=0)
print data.shape
n_classes = 5  # number of classes
keep_rate = 0.5  # kep rate for dropout
len_block = 10  # the window size of each segment
overlap = 0.5  # the overlapping
# data_size_1 = data.shape[0]
no_fea = data.shape[1] - 1  # the number of features
# data segmentation
new_x, new_y, data = extract(data, no_fea, len_block, int(overlap * len_block), n_classes)
print new_x.shape, new_y.shape, data.shape

# make the data size as a multiple of 5. This operation is prepare for batch splitting.
data_size = data.shape[0]  # data size
dot = data_size % 5
data = data[0:data_size - dot]
data_size = data.shape[0]  # update data_size
no_fea_long = no_fea * len_block  # the feature dimension after segmentation

# split data into 80% and 20% for training and testing
train_data = data[int(data_size*0.2):]
test_data = data[:int(data_size*0.2)]
np.random.shuffle(train_data)
np.random.shuffle(test_data)
supervision_rate = 0.6  # key parameter: supervision rate
n_labeled = int(train_data.shape[0]*supervision_rate)  # the number of labelled training samples
# create training/testing features and labels
train_x = train_data[:, :no_fea_long]
train_y = one_hot(train_data[:, no_fea_long:no_fea_long + 1])
test_x = test_data[:, :no_fea_long]
test_y = one_hot(test_data[:, no_fea_long:no_fea_long + 1])
print 'shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape

# split semi supervised dataset into labelled and unlabelled
x_l = train_x[0:n_labeled]
y_l = train_y[0:n_labeled]
x_u = train_x[n_labeled:train_data.shape[0]]
y_u = train_y[n_labeled:train_data.shape[0]]
print 'shape_', train_data.shape, np.array(x_l).shape, np.array(x_u).shape

# dimensions: n_fea_long, n_classes, 1
label_data = np.hstack([x_l, y_l, np.ones([x_l.shape[0], 1])])
unlabelled_data = np.hstack([x_u, y_u, np.zeros([x_u.shape[0], 1])])
joint_data = np.vstack([label_data, unlabelled_data])
# joint_data = label_data  # only use labelled data for GAN

np.random.shuffle(joint_data)  # shuffle the labeled data and unlabelled data
train_x_semi = joint_data[:, 0: no_fea_long]
train_y_semi = joint_data[:, no_fea_long: no_fea_long+n_classes]
# flag_ is a mark for labelled or unlabelled, which decide
# use which loss function (L_{label} or L_{unlabel}) to calculate the loss
flag_ = joint_data[:, no_fea_long+n_classes: no_fea_long+n_classes+1]
print 'flag_ shape', flag_.shape
# for the test data, we hope to recognize the specific class from the K classes, thus, we set the flag_test as 1
flag_test = np.ones([test_x.shape[0], 1])

dim_img = no_fea_long
dim_z = int(no_fea_long)
n_hidden = int(no_fea_long/4)
global n_classes, dim_z
print 'code neurons, hidden neurons', dim_z, n_hidden
learn_rate = 0.0005

lr_gan = 0.0001
# train
batch_size = test_data.shape[0]
n_samples = train_data.shape[0]
total_batch = int(n_samples / batch_size)
# total_batch = 4  # n_groups
n_epochs = 2000
ADD_NOISE = False

# input placeholders
# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
"""VAE++"""
y, z_s, loss, neg_marginal_likelihood, KL_divergence, z_I, mu, sigma = autoencoder(x_hat, x, dim_img, dim_z, n_hidden,
                                                                                 keep_prob)

""" Prepare for the semi-supervised GAN, i.e., the discriminator"""
# data preparation for discriminator
n_classes_d = n_classes + 1  # the number of classes in semi-supervised GAN
label = tf.placeholder(tf.float32, shape=[None, n_classes], name='inputlabel')  # feed the label in, (n_classes)
flag_in = tf.placeholder(tf.float32, shape=[None, 1], name='flag')  # labeled: 1, unlabelled : 0

final_z = tf.placeholder(tf.float32, shape=[None, dim_z])
label_z = tf.concat([label, tf.zeros(shape=(batch_size, 1))], axis=1)
label_z_s = tf.concat([label, tf.ones(shape=(batch_size, 1))], axis=1)


un_label_z = tf.concat([tf.ones(shape=(batch_size, 1)), tf.zeros(shape=(batch_size, 1))], axis=1)
un_label_z_s = tf.concat([tf.zeros(shape=(batch_size, 1)), tf.ones(shape=(batch_size, 1))], axis=1)

# following with dim: dim_z, n_clsses_d, 2, 1, 1
"""Create False/True data for the discriminator"""
z_data = tf.concat([z_I, label_z, un_label_z, flag_in], axis=1)
z_s_data = tf.concat([z_s, label_z_s, un_label_z_s, flag_in], axis=1)
Data = tf.concat([z_data, z_s_data], axis=0)
# np.random.shuffle(Data)
dim_d = dim_z + n_classes_d + 2 + 1 + 1
print 'dim_z', dim_z
# dim_z = 210
d_data = Data[:, 0:dim_z]
super_label = Data[:, dim_z:dim_z+n_classes_d]
unsuper_label = Data[:, dim_d-4: dim_d-2]
flag = Data[:, dim_d-2: dim_d-1]


# just a very simple CNN classifier, used to compare the performance with KNN.
# Please set your own parameters based on your situation
def CNN(input):
    z_image = tf.reshape(input, [-1, 1, dim_z, 1])
    depth_1 = 10
    h_conv1 = tf.contrib.layers.convolution2d(z_image, depth_1, [3, 3], activation_fn=tf.nn.relu,  padding='SAME')
    # fc1 layer
    input_size = dim_z*depth_1
    size3 = dim_z
    fc2_p = fc_relu(data=h_conv1, input_shape=input_size, output_size=size3)
    prediction = fc_nosig(data=fc2_p, input_shape=size3, output_size=n_classes_d)

    # change the unlabelled prediction to [1, 0] or [0, 1]
    aa = tf.reduce_max(prediction[:, 0:n_classes_d-1], axis=1, keep_dims=True)
    un_prediction = tf.concat([aa, prediction[:, n_classes_d-1:n_classes_d]], axis=1)
    return prediction, un_prediction, fc2_p

super_prediction, unsuper_prediction, middle_feature = CNN(d_data)
super_prediction_, _, __ = CNN(final_z)

""" loss calculation"""
# unsupervised loss
unsuper_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=unsuper_label*0.9,
                                                                      logits=unsuper_prediction))
# supervised loss
super_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=super_label*0.9, logits=super_prediction))
loss_featurematch = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=d_data, logits=middle_feature))
d_loss = 0.9*flag * super_loss + 0.1*(1-flag)*unsuper_loss  # this weights of two loss: 0.9, 0.1

loss_gan_supervised = super_loss
loss_gan_unsupervised = unsuper_loss
d_loss_plot = loss_gan_supervised + loss_gan_unsupervised
# acc
acc_gan_unsuper = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(unsuper_prediction, 1),
                                                  tf.argmax(unsuper_label, 1)), tf.float32))
acc_gan_super = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(super_prediction, 1), tf.argmax(super_label, 1)), tf.float32))

"""Build a CNN classifier"""
acc_final = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(super_prediction_[:, :n_classes], 1),
                                            tf.argmax(label, 1)), tf.float32))
final_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label * 0.9,
                                                                    logits=super_prediction_[:, :n_classes]))
train_cnn = tf.train.AdamOptimizer(0.0004).minimize(final_loss)
train_d = tf.train.AdamOptimizer(lr_gan).minimize(d_loss)
# train VAE, add feature matching loss, but it seems the feature matching is not effective
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss + loss_featurematch)

""" training """
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # used to control the GPU device
config = tf.ConfigProto()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print 'Start training'
time_s = time.clock()
for epoch in range(n_epochs):
    # Loop over data batches
    for i in range(total_batch):
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (n_samples)
        batch_xs_input = train_x_semi[offset:(offset + batch_size), :]  # semi supervised as input
        batch_y_input = train_y_semi[offset:(offset + batch_size), :]
        flag_input = flag_[offset:(offset + batch_size), :]

        batch_xs_target = batch_xs_input
        test_x = test_x

        train_feed = {x_hat: batch_xs_input, x: batch_xs_target, label: batch_y_input,
                      flag_in: flag_input, keep_prob: keep_rate}
        test_feed = {x_hat: test_x, x: test_x, label: test_y, flag_in: flag_test, keep_prob: keep_rate}
        # train VAE++
        _, tot_loss, loss_likelihood, loss_divergence = sess.run(
            (train_op, loss, neg_marginal_likelihood, KL_divergence),
            feed_dict={x_hat: batch_xs_input,
                       x: batch_xs_target,
                       label: batch_y_input,
                       flag_in:flag_input,
                       keep_prob: keep_rate})
        # train GAN
        sess.run(train_d,  feed_dict=train_feed)

    # print cost every epoch
    if epoch% 40 == 0:
        test_feed = {x_hat: test_x, x: test_x, label: test_y, flag_in: flag_test, keep_prob: keep_rate}
        tot_loss_t, loss_likelihood_t, loss_divergence_t, d_loss_t, acc_super, acc_unsuper = sess.run(
            (loss, neg_marginal_likelihood, KL_divergence, d_loss, acc_gan_super, acc_gan_unsuper),
            feed_dict=test_feed)
        print 'epoch', epoch, 'testing loss:', tot_loss_t, np.mean(loss_likelihood_t), np.mean(loss_divergence_t)
        print 'epoch', epoch, 'GAN loss, super acc, unsuper_acc', np.mean(d_loss_t), acc_super, acc_unsuper
        time_e = time.clock()
        test_z = sess.run(mu, feed_dict={x_hat: test_x, x: test_x, keep_prob: keep_rate})
        # creat semi_labeled_z based on x_l, the corresponding label is y_l
        # the semi_labeled_z is the exclusive code produced by our AVAE.
        semi_labeled_z = sess.run(z_I, feed_dict={x_hat: x_l, x: x_l, keep_prob: keep_rate})

        """Classification"""
        from sklearn.metrics import classification_report
        time1 = time.clock()
        print "This is the results:----------------------"

        # Prepare training and testing data for classifier
        feature_train = semi_labeled_z
        label_train = y_l
        feature_test = test_z
        label_test = test_y
        # KNN classifier. Personally, I prefer to KNN for the lightweight
        from sklearn.neighbors import KNeighborsClassifier
        time1 = time.clock()
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(feature_train, np.argmax(label_train, 1))
        time2 = time.clock()
        knn_acc = neigh.score(feature_test, np.argmax(label_test, 1))
        knn_prob = neigh.predict_proba(feature_test)
        time3 = time.clock()
        print classification_report(np.argmax(label_test, 1), np.argmax(knn_prob,1))
        print "KNN Accuracy, epoch:", knn_acc, epoch
        print "training time", time2 - time1, 'testing time', time3 - time2

        # CNN classifier, in some situations, CNN outperforms KNN
        # for i in range(1000):
        #     train_acc, _, cnn_cost, pred = sess.run([acc_final, train_cnn, final_loss, super_prediction_],
        #                                 feed_dict={final_z: feature_train, label: label_train, keep_prob: keep_rate})
        #     if i % 100 == 0:  # test
        #         test_acc = sess.run(acc_final,
        #                             feed_dict={final_z: feature_test, label: label_test,keep_prob: keep_rate})
        #         print 'iteration, CNN final acc', i, train_acc, test_acc, cnn_cost

