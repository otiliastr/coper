import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import math


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


class BiLinearPerturbation(object):
    def __init__(self, Config):
        self.num_ent = Config['num_ent']
        self.num_rel = Config['num_rel']
        self.emb_dim = Config['emb_dim']
        self.lr = Config['lr']
        self.summaries = None
        self.weights = self.get_weights()
        self.reg_weight = tf.placeholder(tf.float32)
        self.baseline_weight = tf.placeholder(tf.float32)

        self.sim_threshold = tf.placeholder(tf.float32)

        self.relreg_iterator_handle = tf.placeholder_with_default("", shape=[])
        self.relreg_iterator = tf.data.Iterator.from_string_handle(
            self.relreg_iterator_handle,
            output_types={
                's_ent': tf.int64,
                'rel': tf.int64,
                'seq_0': tf.int64,
                'seq_1': tf.int64,
                'seq_2': tf.int64,
                'sim': tf.float32,
                'seq_mask': tf.int64,
                'e2_multi': tf.float32
            },
            output_shapes={
                's_ent': [None],
                'rel': [None],
                'seq_0': [None],
                'seq_1': [None],
                'seq_2': [None],
                'sim': [None],
                'seq_mask': [None, 3],
                'e2_multi': [None, self.num_ent]})

        self.next_relreg_sample = self.relreg_iterator.get_next()

        self.is_train = tf.placeholder_with_default(False, shape=[])
        
        self.reg_e1 = self.next_relreg_sample['s_ent']
        self.reg_rel = self.next_relreg_sample['rel']
        self.reg_seq_0 = self.next_relreg_sample['seq_0']
        self.reg_seq_1 = self.next_relreg_sample['seq_1']
        self.reg_seq_2 = self.next_relreg_sample['seq_2']
        self.reg_sim = self.next_relreg_sample['sim']
        self.reg_seq_mask = self.next_relreg_sample['seq_mask']
        self.e2_multi = self.next_relreg_sample['e2_multi']
        batch_size = tf.shape(self.reg_e1)[0]
        # Sim Thresholding
        self.reg_sim = tf.reshape(self.reg_sim, [batch_size, 1])
        self.reg_sim = tf.maximum(self.reg_sim - self.sim_threshold, 0.0)

        # DistMult Prediction Pipeline 
        self.e1_emb = tf.nn.embedding_lookup(self.weights['ent_emb'], self.reg_e1)
        self.rel_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_rel)
        # DistMult Prediction
        self.pair_e2_pred = tf.einsum('bn,bnm->bm', self.e1_emb, self.rel_emb)
        self.pair_e2_pred = tf.nn.tanh(self.pair_e2_pred)
        # Rel Regularization Pipeline
        self.reg_seq_0_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_seq_0)
        self.reg_seq_1_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_seq_1)
        self.reg_seq_2_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_seq_2)

        batch_size = tf.shape(self.reg_seq_0_emb)[0]
        # unroll sequence through graph
        seq_rel_1_e2 = tf.nn.tanh(tf.einsum('bn,bnm->bm', self.e1_emb, self.reg_seq_0_emb))
        seq_rel_2_e2 = tf.nn.tanh(tf.einsum('bn,bnm->bm', seq_rel_1_e2, self.reg_seq_1_emb))
        seq_rel_3_e2 = tf.nn.tanh(tf.einsum('bn,bnm->bm', seq_rel_2_e2, self.reg_seq_2_emb))
        # prepare predicted e2s for masking
        seq_rel_1_e2 = tf.reshape(seq_rel_1_e2, [batch_size, 1, self.emb_dim])
        seq_rel_2_e2 = tf.reshape(seq_rel_2_e2, [batch_size, 1, self.emb_dim])
        seq_rel_3_e2 = tf.reshape(seq_rel_3_e2, [batch_size, 1, self.emb_dim])
        # extract correct unroll amount
        seq_e2_pred = tf.boolean_mask(tf.concat([seq_rel_1_e2, seq_rel_2_e2, seq_rel_3_e2], 1), self.reg_seq_mask)
        # ensure data is of same shape
        self.rel_e2_pred = tf.reshape(self.pair_e2_pred, [batch_size, self.emb_dim])
        self.seq_e2_pred = tf.reshape(seq_e2_pred, [batch_size, self.emb_dim])
        # perturb training data
        self.reg_sim = tf.reshape(self.reg_sim, [batch_size, 1])
        self.perturbed_e2_pred = self.rel_e2_pred * (1. - self.reg_sim) + self.seq_e2_pred * (self.reg_sim)
        # compute prediction probabilities
        self.e2s = self.weights['ent_emb']
        self.e2_pred = tf.matmul(self.perturbed_e2_pred, self.e2s, transpose_b=True)

        self.bilinear_elem_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.e2_multi,
                                                                  logits=self.e2_pred)
        self.loss = tf.reduce_sum(self.bilinear_elem_loss)
 
        # Optimization
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.loss)

        # Validation Implementation
        self.input_iterator_handle = tf.placeholder(tf.string, shape=[])
        self.input_iterator = tf.data.Iterator.from_string_handle(
            self.input_iterator_handle,
            output_types={
                'e1': tf.int64,
                'e2': tf.int64,
                'rel': tf.int64,
                'e2_multi1': tf.float32},
            output_shapes={
                'e1': [None],
                'e2': [None],
                'rel': [None],
                'e2_multi1': [None, self.num_ent]})

        self.next_input_sample = self.input_iterator.get_next()
        self.valid_e1 = self.next_input_sample['e1']
        self.valid_rel = self.next_input_sample['rel']
        self.valid_e2_multi = self.next_input_sample['e2_multi1']
        # Validation Pipeline
        self.valid_e1_emb = tf.nn.embedding_lookup(self.weights['ent_emb'], self.valid_e1)
        self.valid_rel_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.valid_rel)
        
        self.valid_pair_e2_pred = tf.einsum('bn,bnm->bm', self.valid_e1_emb, self.valid_rel_emb)
        self.valid_pair_e2_pred = tf.nn.tanh(self.valid_pair_e2_pred)
        # normalize predictions
        self.valid_raw_pred = tf.matmul(self.valid_pair_e2_pred, self.e2s, transpose_b=True)

        self.valid_raw_pred = self.valid_raw_pred
        self.valid_elem_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.valid_e2_multi,
                                                                  logits=self.valid_raw_pred)
        self.valid_loss = tf.reduce_sum(self.valid_elem_loss)


    def get_weights(self):
        weights = dict()
        weights['ent_emb'] = tf.get_variable('ent_embeddings',
                                             [self.num_ent, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
        """
        weights['rel_emb_1'] = tf.get_variable('rel_embeddings_1',
                                             [self.num_rel, self.emb_dim, 8],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
        weights['rel_emb_2'] = tf.get_variable('rel_embeddings_2',
                                             [self.num_rel, 8, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
        weights['rel_emb'] = tf.matmul(weights['rel_emb_1'], weights['rel_emb_2'])
        """
        weights['rel_emb'] = tf.get_variable('rel_embeddings',
                                             [self.num_rel, self.emb_dim, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
         #weights['rel_bias'] = tf.get_variable('rel_bias',
         #                                    [self.num_rel, self.emb_dim, self.emb_dim],
          #                                   dtype=tf.float32,
           #                                  initializer=tf.contrib.layers.xavier_initializer())

        return weights

