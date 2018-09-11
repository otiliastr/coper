import tensorflow as tf
import numpy as np
import os
import sys
import pickle

def variable_summaries(var):
    # Attach a lot of summaries to a tensor (for TensorBoard visualizations).
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class DistMultReg(object):
    def __init__(self, Config):
        self.num_ent = Config['num_ent']
        self.num_rel = Config['num_rel']
        self.emb_dim = Config['emb_dim']
        self.lr = Config['lr']
        self.batch_size = Config['batch_size']
        self.max_seq_len = Config['max_seq_len']

        self.weights = self.get_weights()

        self.reg_weight = tf.placeholder(tf.float32)
        self.baseline_weight = tf.placeholder(tf.float32)

        self.distmult_iterator_handle = tf.placeholder(tf.string, shape=[])
        self.distmult_iterator = tf.data.Iterator.from_string_handle(
            self.distmult_iterator_handle,
            output_types={
                'e1': tf.int64,
                'e2': tf.int64,
                'rel': tf.int64,
                'e2_multi1': tf.float32},
            output_shapes={
                'e1': [self.batch_size],
                'e2': [self.batch_size],
                'rel': [self.batch_size],
                'e2_multi1': [self.batch_size, self.num_ent]})

        self.relreg_iterator_handle = tf.placeholder_with_default("", shape=[])
        self.relreg_iterator = tf.data.Iterator.from_string_handle(
            self.relreg_iterator_handle,
            output_types={
                'seq': tf.int64,
                'seq_multi': tf.float32,
                'seq_len': tf.int64,
                'seq_mask': tf.float32
            },
            output_shapes={
                'seq': [self.batch_size, self.max_seq_len],
                'seq_multi': [self.batch_size],
                'seq_len': [self.batch_size],
                'seq_mask': [self.batch_size, self.max_seq_len]})

        self.next_input_sample = self.distmult_iterator.get_next()
        self.next_relreg_sample = self.relreg_iterator_handle.get_next()

        self.is_train = tf.placeholder_with_default(False, shape=[])
        self.e1 = self.next_input_sample['e1']
        self.rel = self.next_input_sample['rel']
        self.e2_multi = self.next_input_sample['e2_multi1']
        self.seq = self.next_relreg_sample['seq']
        self.seq_multi = self.next_relreg_sample['seq_multi']
        self.seq_len = self.next_relreg_sample['seq_len']
        self.seq_mask = tf.reshape(self.next_relreg_sample['seq_mask'], [self.batch_size, self.max_seq_len, 1])


        # DistMult Side
        # self.e1 = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 1])
        # self.rel = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 1])
        # self.e2_multi = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_ent])
        # Rel Regularization Side
        # self.seq = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_seq_len])
        # self.seq_lens = tf.placeholder(dtype=tf.float32, shape=[self.batch_size])
        # self.output_mask = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_seq_len, 1])
        # self.rel_multi = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_rel])

        # DistMult Prediction Pipeline
        self.e1_emb = tf.nn.embedding_lookup(self.weights['ent_emb'], self.e1)
        self.rel_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.rel)
        # DistMult Prediction
        self.distmult_raw_pred = tf.matmul(self.e1_emb * self.rel_emb, self.weights['ent_emb'], transpose_b=True)
        # self.distmult_pred = tf.nn.sigmoid(self.distmult_raw_pred)

        self.distmult_elem_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.e2_multi,
                                                                  logits=self.distmult_raw_pred)
        self.distmult_loss = tf.reduce_mean(self.distmult_elem_loss)

        # Rel Regularization Pipeline
        self.sequences = tf.nn.embedding_lookup(self.weights['rel_emb'], self.seq)
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.emb_dim, forget_bias=1.0, state_is_tuple=True)
        self.init_state = self.cell.zero_state(self.batch_size, tf.float32)
        self.outputs, self.state = tf.nn.dynamic_rnn(
            self.cell,
            self.sequences,
            sequence_length=self.seq_len,
            initial_state=self.init_state,
            dtype=tf.float32,
            parallel_iterations=None,
            swap_memory=False,
            time_major=False,
            scope=None
        )

        self.output_filtered = self.outputs * self.seq_mask
        # turns 3D matrix of one non-zero row per 2D matrix to 2D dense matrix
        self.outputs_collapsed = tf.reduce_sum(self.output_filtered, 1)

        self.relreg_raw = tf.matmul(self.outputs_collapsed, self.weights['rel_emb'], transpose_b=True)
        self.relreg_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.seq_multi,
                                                           logits=self.relreg_raw)

        # Optimization
        self.collective_loss = self.baseline_weight * self.distmult_loss + self.reg_weight * self.relreg_loss
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.collective_loss)

    def get_weights(self):
        weights = dict()
        weights['ent_emb'] = tf.get_variable('ent_embeddings',
                                             [self.num_ent, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
        weights['rel_emb'] = tf.get_variable('rel_embeddings',
                                             [self.num_rel, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
        return weights
