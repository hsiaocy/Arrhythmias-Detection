"""
Author:
https://github.com/y0ast/VAE-TensorFlow/blob/master/experiment4.py
"""
from __future__ import division
from __future__ import print_function
import os.path

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# FLAGS = None
# mnist = input_data.read_data_sets('MNIST_data')

# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=self.stddev)
#   return tf.Variable(initial)
#
# def bias_variable(shape):
#   initial = tf.constant(0., shape=shape)
#   return tf.Variable(initial)

class Autoencoder(object):
    def __init__(self, data_set):

        self.data_set = data_set
        if len(np.shape(data_set)) == 2:
            (self.data_num, self.input_dim) = np.shape(data_set)
        if len(np.shape(data_set)) == 3:
            (self.data_with, self.height) = np.shape(data_set[0])
            pass

        self.learning_rate = 0.001
        self.batch_size = 64
        self.hidden_encoder_dim = 512
        self.hidden_decoder_dim = 512
        self.latent_dim = 64
        self.lam = 0
        self.stddev = 0.001
        self.wta_rate = None

    def _init(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.l2_loss = tf.constant(0.0, dtype=tf.float32)

    def encoder(self):
        W_encoder_input_hidden = tf.Variable(tf.truncated_normal([self.input_dim, self.hidden_encoder_dim], stddev=self.stddev))
        b_encoder_input_hidden = tf.Variable(tf.constant(0., shape=[self.hidden_encoder_dim]))
        self.l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

        # Hidden layer encoder
        hidden_encoder = tf.nn.relu(tf.matmul(self.x, W_encoder_input_hidden) + b_encoder_input_hidden)

        W_encoder_hidden_mu = tf.Variable(tf.truncated_normal([self.hidden_encoder_dim, self.latent_dim], stddev=self.stddev))
        b_encoder_hidden_mu = tf.Variable(tf.constant(0., shape=[self.latent_dim]))
        self.l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

        # Mu encoder
        self.mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

        W_encoder_hidden_logvar = tf.Variable(tf.truncated_normal([self.hidden_encoder_dim, self.latent_dim], stddev=self.stddev))
        b_encoder_hidden_logvar = tf.Variable(tf.constant(0., shape=[self.latent_dim]))
        self.l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

        # Sigma encoder
        self.logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(self.logvar_encoder), name='epsilon')

        # Sample latent variable
        self.std_encoder = tf.exp(0.5 * self.logvar_encoder)
        self.z = self.mu_encoder + tf.multiply(self.std_encoder, epsilon)

    def decoder(self):
        self.W_decoder_z_hidden = tf.Variable(tf.truncated_normal([self.latent_dim, self.hidden_decoder_dim], stddev=self.stddev))
        self.b_decoder_z_hidden = tf.Variable(tf.constant(0., shape=[self.hidden_decoder_dim]))
        self.l2_loss += tf.nn.l2_loss(self.W_decoder_z_hidden)

        # Hidden layer decoder
        hidden_decoder = tf.nn.relu(tf.matmul(self.z, self.W_decoder_z_hidden) + self.b_decoder_z_hidden)

        W_decoder_hidden_reconstruction = tf.Variable(tf.truncated_normal([self.hidden_decoder_dim, self.input_dim], stddev=self.stddev))
        b_decoder_hidden_reconstruction = tf.Variable(tf.constant(0., shape=[self.input_dim]))
        self.l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

        self.x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction

    def model(self):

        self.encoder()
        self.decoder()

        KLD = -0.5 * tf.reduce_sum(1 + self.logvar_encoder - tf.pow(self.mu_encoder, 2) - tf.exp(self.logvar_encoder),
                                   reduction_indices=1)

        BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_hat, labels=self.x),
                            reduction_indices=1)

        self.loss = tf.reduce_mean(BCE + KLD)

        regularized_loss = self.loss + self.lam * self.l2_loss

        self.loss_summ = tf.summary.scalar("lowerbound", self.loss)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(regularized_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        # saver = tf.train.Saver()

    def train(self, train_x, test_x, epoch=300, itr=None):

        self._init()
        self.model()

        init = tf.global_variables_initializer()
        iteration = len(train_x) // self.batch_size if itr is None else itr
        with tf.Session() as sess:
            sess.run(init)
            # summary_writer = tf.summary.FileWriter('experiment', graph=sess.graph)
            # # if os.path.isfile("save/model.ckpt"):
            # #     print("Restoring saved parameters")
            # #     saver.restore(sess, "save/model.ckpt")
            # # else:
            print("Initializing parameters")

            for j in range(epoch):
                for i in range(iteration):
                    loss = sess.run([self.loss], feed_dict={
                        self.x: train_x[i * self.batch_size:(i + 1) * self.batch_size]})

                    if i % 1 == 0:
                        # save_path = saver.save(sess, "save/model.ckpt")
                        print("Step {0} | Loss: {1}".format(i, loss))
            train_z, train_mn, train_sd, train_wav = sess.run([self.z, self.mu_encoder, self.std_encoder, self.x_hat], feed_dict={self.x: train_x})
            tset_z, test_mn, test_sd, test_wav = sess.run([self.z, self.mu_encoder, self.std_encoder, self.x_hat], feed_dict={self.x: test_x})
            train_feat = {'z': train_z, 'mn': train_mn, 'sd': train_sd, 'dec': train_wav}
            test_feat = {'z': tset_z, 'mn': test_mn, 'sd': test_sd, 'dec': test_wav}
        return train_feat, test_feat

if __name__ == '__main__':
    # ae = Autoencoder()
    # ae._init()
    # ae.encoder()
    # ae.decoder()
    # ae.train()
    pass