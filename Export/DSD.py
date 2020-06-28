import numpy as np
import tensorflow as tf

class DSD:
    """
    Proposed method
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.para = dict()
        for key, item in kwargs.items():
            self.para[key] = item

        self.optimizer = None
        self.loss = None
        self.enc_layer_1 = None
        self.dec_layer_1 = None
        self.enc_output = None

    def tf_init(self, ):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.para['feat_len']])

    def model(self, ):
        pass

    def train_feat(self, ):
        pass

    # TODO: high variance win
    def wta(self, _input, wta_rate=0.9):
        """
        Winner Take All method

        :param _input: parameters
        :param wta_rate: % to win
        :return: Winner weights
        """
        batch_size, feat_size = tf.shape(_input)[0], tf.shape(_input)[1]

        # Rvery value defaults tf.float32 in tensorflow, so change it to tf.int32 to get k
        k = tf.cast(wta_rate * tf.cast(feat_size, tf.float32), tf.int32)

        # Find the position k
        th_k, _ = tf.nn.top_k(_input, k)

        # minimum winner of parameters
        winner_min = th_k[:, k - 1: k]

        # this is a mask like [0,0,0,1,1,0, ..., 1], the same size as parameters
        drop = tf.where(_input < winner_min, tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))

        # input data wear drop mask
        _input *= drop
        return _input


class DenseLayer(DSD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tf_init(self, ):
        """
        Weights initialization

        """
        super().tf_init()
        self.w_enc_h1 = tf.Variable(tf.random_normal([self.para['feat_len'], self.para['num_hidden']]))
        self.w_dec_h1 = tf.Variable(tf.random_normal([self.para['num_hidden'], self.para['feat_len']]))
        self.b_enc_h1 = tf.Variable(tf.random_normal([self.para['num_hidden']]))
        self.b_dec_h1 = tf.Variable(tf.random_normal([self.para['feat_len']]))

    def model(self, ):
        """
        Construct model
        """

        # Layers
        self.enc_layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.w_enc_h1), self.b_enc_h1))
        self.enc_layer_1 = tf.layers.batch_normalization(inputs=self.enc_layer_1)
        self.dec_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.enc_layer_1, self.w_dec_h1), self.b_dec_h1))
        self.dec_layer_1 = tf.layers.batch_normalization(inputs=self.dec_layer_1)

        # Prediction & Targets (Labels) are the input data.
        y_pred, y_true = self.dec_layer_1, self.x

        # Define loss and optimizer, minimize the squared error
        self.loss = tf.reduce_mean(tf.square(y_true - y_pred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.para['learning_rate']).minimize(self.loss)

    def train_feat(self, train_x, itr=None, display_step=10):

        self.tf_init()
        self.model()
        init = tf.global_variables_initializer()

        iteration = len(train_x) // self.para['batch_size'] if itr is None else itr

        with tf.Session() as sess:
            sess.run(init)
            for j in range(self.para['epoch']):
                for i in range(iteration):

                    loss, w_enc, b_enc, w_dec, b_dec = \
                        sess.run([self.loss, self.w_enc_h1, self.b_enc_h1, self.w_dec_h1, self.b_dec_h1],
                                 feed_dict={self.x: train_x[i * self.para['batch_size']:(i + 1) * self.para['batch_size']]})

                    print('Iteration %i: Optimization: %f' % (i + 1, loss))
                print('Finished Epoch {0}\n'.format(j+1))
        para_dict = {'w_enc': w_enc, 'w_dec': w_dec, 'b_enc': b_enc, 'b_dec': b_dec}
        return para_dict


class SparseLayer(DSD):
    def __init__(self, para_dict, mask_dict, **kwargs):
        super().__init__(**kwargs)
        self.para_dict = para_dict
        self.mask_dict = mask_dict

    def tf_init(self):
        super().tf_init()
        self.w_enc_h1 = tf.get_variable(name='w_prune_enc_h1', initializer=self.para_dict['w_enc'])
        self.w_dec_h1 = tf.get_variable(name='w_brune_dec_h1', initializer=self.para_dict['w_dec'])
        self.b_enc_h1 = tf.get_variable(name='p_brune_enc_h1', initializer=self.para_dict['b_enc'])
        self.b_dec_h1 = tf.get_variable(name='p_brune_dec_h1', initializer=self.para_dict['b_dec'])

    def mask_func(self, w_enc, w_dec, b_enc, b_dec, sparsity=0.5):
        mask_w_enc = tf.where(w_enc > self._top_k(_input=w_enc, sparsity=sparsity), x=tf.ones(shape=tf.shape(w_enc), dtype=tf.float32), y=tf.zeros(shape=tf.shape(w_enc), dtype=tf.float32))
        mask_w_dec = tf.where(w_dec > self._top_k(_input=w_dec, sparsity=sparsity), x=tf.ones(shape=tf.shape(w_dec), dtype=tf.float32), y=tf.zeros(shape=tf.shape(w_dec), dtype=tf.float32))
        mask_b_enc = tf.where(b_enc > self._top_k(_input=b_enc, sparsity=sparsity), x=tf.ones(shape=tf.shape(b_enc), dtype=tf.float32), y=tf.zeros(shape=tf.shape(b_enc), dtype=tf.float32))
        mask_b_dec = tf.where(b_dec > self._top_k(_input=b_dec, sparsity=sparsity), x=tf.ones(shape=tf.shape(b_dec), dtype=tf.float32), y=tf.zeros(shape=tf.shape(b_dec), dtype=tf.float32))
        return mask_w_enc, mask_b_enc, mask_w_dec, mask_b_dec
        pass

    @staticmethod
    def _top_k(_input, sparsity):
        para_len = tf.shape(_input)[-1]
        k = tf.cast(x=(1 - sparsity) * para_len, dtype=tf.int16, name='top_k')
        top_k, _ = tf.nn.top_k(input=_input, k=k)  # top_k.shape: (-1, k)
        thres_k = top_k[:, -1] if tf.rank(_input) > 2 else top_k[k]
        return thres_k
        pass

    def model(self):
        # keep sparse weight
        self.w_enc_h1 *= self.mask_dict['mask_w_enc']
        self.b_enc_h1 *= self.mask_dict['mask_b_enc']
        self.w_dec_h1 *= self.mask_dict['mask_w_dec']
        self.b_dec_h1 *= self.mask_dict['mask_b_dec']

        # Construct model
        self.enc_layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.w_enc_h1), self.b_enc_h1))
        self.enc_layer_1 = tf.layers.batch_normalization(inputs=self.enc_layer_1)
        self.dec_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.enc_layer_1, self.w_dec_h1), self.b_dec_h1))
        self.dec_layer_1 = tf.layers.batch_normalization(inputs=self.dec_layer_1)

        # Prediction & Targets (Labels) are the input data.
        y_pred, y_true = self.dec_layer_1, self.x

        # Define loss and optimizer, minimize the squared error
        self.loss = tf.reduce_mean(tf.square(y_true - y_pred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.para['learning_rate']).minimize(self.loss)

    def train_feat(self, train_x, itr=None, display_step=10):

        self.tf_init()
        self.model()
        init = tf.global_variables_initializer()

        iteration = len(train_x) // self.para['batch_size'] if itr is None else itr
        with tf.Session() as sess:
            sess.run(init)
            for j in range(self.para['epoch']):
                for i in range(iteration):

                    loss, w_enc, b_enc, w_dec, b_dec = \
                        sess.run([self.loss, self.w_enc_h1, self.b_enc_h1, self.w_dec_h1, self.b_dec_h1],
                                 feed_dict={self.x: train_x[i * self.para['batch_size']:(i + 1) * self.para['batch_size']]})

                    print('Iteration %i: Optimization: %f' % (i + 1, loss))
                print('Finished Epoch {0}\n'.format(j))

        para_dict = {'w_enc': w_enc, 'b_enc': b_enc, 'w_dec': w_dec, 'b_dec': b_dec}
        return para_dict
        pass
    pass


class ReDense(DSD):
    def __init__(self, para_dict, **kwargs):
        super().__init__(**kwargs)
        self.para_dict = para_dict

    def tf_init(self):
        super().tf_init()
        self.w_enc_h1 = tf.get_variable(name='w_restore_enc_h1', initializer=self.para_dict['w_enc'])
        self.w_dec_h1 = tf.get_variable(name='w_restore_dec_h1', initializer=self.para_dict['w_dec'])
        self.b_enc_h1 = tf.get_variable(name='b_restore_enc_h1', initializer=self.para_dict['b_enc'])
        self.b_dec_h1 = tf.get_variable(name='b_restore_dec_h1', initializer=self.para_dict['b_dec'])

    def model(self):
        # Construct model
        self.enc_layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.w_enc_h1), self.b_enc_h1))
        self.enc_layer_1 = tf.layers.batch_normalization(inputs=self.enc_layer_1)
        self.dec_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.enc_layer_1, self.w_dec_h1), self.b_dec_h1))
        self.dec_layer_1 = tf.layers.batch_normalization(inputs=self.dec_layer_1)

        # Prediction & Targets (Labels) are the input data.
        y_pred, y_true = self.dec_layer_1, self.x

        # Define loss and optimizer, minimize the squared error
        self.loss = tf.reduce_mean(tf.square(y_true - y_pred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.para['learning_rate']).minimize(self.loss)

    def train_feat(self, train_x, test_x, itr=None, display_step=10, ):

        self.tf_init()
        self.model()
        init = tf.global_variables_initializer()

        iteration = len(train_x) // self.para['batch_size'] if itr is None else itr
        with tf.Session() as sess:
            sess.run(init)
            for j in range(self.para['epoch']):
                for i in range(iteration):

                    loss, w_enc, b_enc, w_dec, b_dec = \
                        sess.run([self.loss, self.w_enc_h1, self.b_enc_h1, self.w_dec_h1, self.b_dec_h1],
                                 feed_dict={self.x: train_x[i * self.para['batch_size']:(i + 1) * self.para['batch_size']]})

                    print('Iteration %i: Optimization: %f' % (i + 1, loss))
                print('Finished Epoch {0}\n'.format(j))

            train_feat = sess.run(([self.enc_layer_1]), feed_dict={self.x: train_x})
            test_feat = sess.run([self.enc_layer_1], feed_dict={self.x: test_x})
            para_dict = {'w_enc': w_enc, 'b_enc': b_enc, 'w_dec': w_dec, 'b_dec': b_dec}
        return train_feat, test_feat, para_dict
        pass