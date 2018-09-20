import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import math

from eunn import EUNNCell

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


class BiLinearReg(object):
    def __init__(self, Config):
        self.num_ent = Config['num_ent']
        self.num_rel = Config['num_rel']
        self.emb_dim = Config['emb_dim']
        self.lr = Config['lr']
        self.summaries = None
        self.batch_size = 128
        self.weights = self.get_weights()

        self.reg_weight = tf.placeholder(tf.float32)
        self.baseline_weight = tf.placeholder(tf.float32)
   
        self.sim_threshold = tf.placeholder(tf.float32)
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

        self.relreg_iterator_handle = tf.placeholder_with_default("", shape=[])
        self.relreg_iterator = tf.data.Iterator.from_string_handle(
            self.relreg_iterator_handle,
            output_types={
                'rel': tf.int64,
                #'seq': tf.int64,
                'seq_0': tf.int64,
                'seq_1': tf.int64,
                'seq_2': tf.int64,
                'sim': tf.float32,
                'seq_mask': tf.int64
                #'seq_mask_0': tf.int64,
                #'seq_mask_1': tf.int64,
                #'seq_mask_2': tf.int64,
            },
            output_shapes={
                'rel': [None],
                'seq_0': [None],
                'seq_1': [None],
                'seq_2': [None],
                #'seq': [None, 3],
                'sim': [None],
                #'seq_mask_0': [None],
                #'seq_mask_1': [None],
                #'seq_mask_2': [None]})
                'seq_mask': [None, 3]})
        
        #printed_input_iterator = tf.Print(self.input_iterator, [self.input_iterator])
        #printed_relreg_sample = tf.Print(self.relreg_iterator, [self.relreg_iterator])

        #self.next_input_sample = self.input_iterator.get_next()
        #self.next_relreg_sample = self.relreg_sample.get_next()

        self.next_input_sample = self.input_iterator.get_next()
        self.next_relreg_sample = self.relreg_iterator.get_next()

        self.is_train = tf.placeholder_with_default(False, shape=[])
        self.e1 = self.next_input_sample['e1']
        self.rel = self.next_input_sample['rel']
        self.e2_multi = self.next_input_sample['e2_multi1']
        
        self.reg_rel = self.next_relreg_sample['rel']
        self.reg_seq_0 = self.next_relreg_sample['seq_0']
        self.reg_seq_1 = self.next_relreg_sample['seq_1']
        self.reg_seq_2 = self.next_relreg_sample['seq_2']
        #self.reg_seq = self.next_relreg_sample['seq']
        self.reg_sim = self.next_relreg_sample['sim']
        #self.reg_seq_mask_0 = self.next_relreg_sample['seq_mask_0']
        #self.reg_seq_mask_1 = self.next_relreg_sample['seq_mask_1']
        #self.reg_seq_mask_2 = self.next_relreg_sample['seq_mask_2']
        self.reg_seq_mask = self.next_relreg_sample['seq_mask']

        #self.reg_rel_printed = tf.Print(self.reg_rel, [self.reg_rel])

        # DistMult Prediction Pipeline
        # max-margin loss. Shift according to piece-wise tanh
        self.pos_sim_shift = 1./(1. - self.sim_threshold) * math.pi * (self.reg_sim - self.sim_threshold)
        self.neg_sim_shift = 1./(self.sim_threshold) * math.pi * (self.reg_sim - self.sim_threshold)
        self.sim_transform = -tf.tanh(self.pos_sim_shift)
        self.not_sim_transform = -tf.tanh(self.neg_sim_shift)
        self.reg_loss_weight = tf.minimum(self.sim_transform, self.not_sim_transform)


        self.e1_emb = tf.nn.embedding_lookup(self.weights['ent_emb'], self.e1)
        self.rel_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.rel)
        # DistMult Prediction
        self.bilinear_raw_pred = tf.matmul(tf.einsum('bn,bnm->bm', self.e1_emb, self.rel_emb),
                                           self.weights['ent_emb'], transpose_b=True)

        self.bilinear_elem_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.e2_multi,
                                                                  logits=self.bilinear_raw_pred)
        self.bilinear_loss = tf.reduce_sum(self.bilinear_elem_loss)

        # Rel Regularization Pipeline
        self.reg_rel_embs = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_rel)# self.reg_rel)
        self.reg_seq_0_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_seq_0)
        self.reg_seq_1_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_seq_1)
        self.reg_seq_2_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_seq_2)

        #self.reg_seq_0_bias = tf.nn.embedding_lookup(self.weights['rel_bias'], self.reg_seq_0)
        #self.reg_seq_1_bias = tf.nn.embedding_lookup(self.weights['rel_bias'], self.reg_seq_1)
        #self.reg_seq_2_bias = tf.nn.embedding_lookup(self.weights['rel_bias'], self.reg_seq_2)
        
        #self.sequences_unpacked = tf.unstack(self.sequences)
        #tensor_maps = []
        #for batch_idx in range(self.sequences.shape[0]):
         #   scanned_prod = tf.scan(lambda a, b: tf.matmul(a, b), self.sequences_unpacked[batch_idx])
          #  valid_prod = tf.reshape(scanned_prod[self.seq_idx[batch_idx]], [1, self.emb_dim, self.emb_dim])
           # tensor_maps.append(valid_prod)
        
        #self.scanned_tensors = tf.map_fn(lambda x: tf.scan(lambda a, b: tf.matmul(a, b), x), self.reg_seq_embs)
        #self.reg_predictions = tf.boolean_mask(self.scanned_tensors, self.reg_seq_mask)
        batch_size = tf.shape(self.reg_seq_0_emb)[0]
        self.reg_seq_0_map = self.reg_seq_0_emb #+ self.reg_seq_0_bias
        self.reg_seq_1_map = tf.matmul(self.reg_seq_0_map, self.reg_seq_1_emb) #+ self.reg_seq_1_bias
        self.reg_seq_2_map = tf.matmul(self.reg_seq_1_map, self.reg_seq_2_emb) #+ self.reg_seq_2_bias

        self.is_seq_1 = tf.reshape(self.reg_seq_0_map, [batch_size, 1, self.emb_dim, self.emb_dim])
        self.is_seq_2 = tf.reshape(self.reg_seq_1_map, [batch_size, 1, self.emb_dim, self.emb_dim])
        self.is_seq_3 = tf.reshape(self.reg_seq_2_map, [batch_size, 1, self.emb_dim, self.emb_dim])
        
        self.reg_predictions = tf.boolean_mask(tf.concat([self.is_seq_1, self.is_seq_2, self.is_seq_3], 1), self.reg_seq_mask)
       # self.batch_maps = tf.concat(tensor_maps, 0)
        #self.reg_diffs_flat = tf.reshape(self.reg_rel_embs - self.reg_predictions, [batch_size, self.emb_dim*self.emb_dim])
        # make sure shapes match
        self.reg_predictions = tf.reshape(self.reg_predictions, [batch_size, self.emb_dim, self.emb_dim])
        self.reg_rel_embs = tf.reshape(self.reg_rel_embs, [batch_size, self.emb_dim, self.emb_dim])
        # normalize row bases
        self.reg_pred_normed = tf.nn.l2_normalize(self.reg_predictions, 2)
        self.reg_rel_embs_normed = tf.nn.l2_normalize(self.reg_rel_embs, 2)
        # compute row-wise cosine similarity
        self.cosine_similarity = tf.reduce_sum(tf.reduce_sum(tf.multiply(self.reg_pred_normed, self.reg_rel_embs_normed), 2), 1)
        # compute loss. Loss is defined as 1 + A' - A component-wise, where A' is dis-similar row-vector to a respective row-vector 
        # in B, and A is similar vector to that in B. Additionally, we add 1 * batch_size*self.emb_dim to bring range to 
        # [0, 2*batch_size*self.emb_dim
        self.offset = 1. * tf.to_float(batch_size * self.emb_dim)
        self.reg_loss = tf.reduce_sum(self.cosine_similarity * self.reg_loss_weight) + self.offset
        
        #self.reg_batch_loss = tf.reduce_sum(tf.square(self.reg_diffs_flat), 1)
        #self.reg_batch_loss = tf.map_fn(lambda x: tf.reduce_sum(x), tf.square(self.reg_rel_embs - self.reg_predictions))
        #self.reg_loss = tf.reduce_sum(self.reg_batch_loss *  self.reg_loss_weight)

        # Optimization
        #self.collective_loss = self.bilinear_loss
        self.collective_loss = self.baseline_weight * self.bilinear_loss + self.reg_weight * self.reg_loss
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.collective_loss)

    def get_weights(self):
        weights = dict()
        weights['ent_emb'] = tf.get_variable('ent_embeddings',
                                             [self.num_ent, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
        weights['rel_emb_1'] = tf.get_variable('rel_embeddings_1',
                                             [self.num_rel, self.emb_dim, 8],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
        weights['rel_emb_2'] = tf.get_variable('rel_embeddings_2',
                                             [self.num_rel, 8, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
        weights['rel_emb'] = tf.matmul(weights['rel_emb_1'], weights['rel_emb_2'])
        #weights['rel_bias'] = tf.get_variable('rel_bias',
         #                                    [self.num_rel, self.emb_dim, self.emb_dim],
          #                                   dtype=tf.float32,
           #                                  initializer=tf.contrib.layers.xavier_initializer())
        return weights

