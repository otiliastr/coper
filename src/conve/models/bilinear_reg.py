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


class BiLinearReg(object):
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
                's_ent': tf.int64,
                'rel': tf.int64,
                #'seq': tf.int64,
                'seq_0': tf.int64,
                'seq_1': tf.int64,
                'seq_2': tf.int64,
                'sim': tf.float32,
                #'agg_sim': tf.float32,
                'seq_mask': tf.int64
                #'seq_mask_0': tf.int64,
                #'seq_mask_1': tf.int64,
                #'seq_mask_2': tf.int64,
            },
            output_shapes={
                's_ent': [None],
                'rel': [None],
                'seq_0': [None],
                'seq_1': [None],
                'seq_2': [None],
                #'seq': [None, 3],
                'sim': [None],
                #'agg_sim': [None],
                #'seq_mask_0': [None],
                #'seq_mask_1': [None],
                #'seq_mask_2': [None]})
                'seq_mask': [None, 3]})
        

        self.next_input_sample = self.input_iterator.get_next()
        self.next_relreg_sample = self.relreg_iterator.get_next()

        self.is_train = tf.placeholder_with_default(False, shape=[])
        self.e1 = self.next_input_sample['e1']
        self.rel = self.next_input_sample['rel']
        self.e2_multi = self.next_input_sample['e2_multi1']
        
        self.reg_e1 = self.next_relreg_sample['s_ent']
        self.reg_rel = self.next_relreg_sample['rel']
        self.reg_seq_0 = self.next_relreg_sample['seq_0']
        self.reg_seq_1 = self.next_relreg_sample['seq_1']
        self.reg_seq_2 = self.next_relreg_sample['seq_2']
        self.reg_sim = self.next_relreg_sample['sim']
        #self.reg_agg_sim = self.next_relreg_sample['agg_sim']
        self.reg_seq_mask = self.next_relreg_sample['seq_mask']

        # DistMult Prediction Pipeline
        self.e1_emb = tf.nn.embedding_lookup(self.weights['ent_emb'], self.e1)
        self.rel_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.rel)
        #self.e1_scaled = self.e1_emb / tf.reduce_maximum(self.e1_emb)
        #self.rel_scaled = self.
        # DistMult Prediction
        self.pair_e2_pred = tf.einsum('bn,bnm->bm', self.e1_emb, self.rel_emb)
        self.pair_e2_pred = tf.nn.tanh(self.pair_e2_pred)
        # normalize predictions
        #self.pair_e2_pred = tf.nn.l2_normalize(self.pair_e2_pred, 1)
        #self.e2s = tf.nn.l2_normalize(self.weights['ent_emb'], 1)
        self.e2s = self.weights['ent_emb']
        self.bilinear_raw_pred = tf.matmul(self.pair_e2_pred, self.e2s, transpose_b=True)

        self.bilinear_raw_pred = self.bilinear_raw_pred
        self.bilinear_elem_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.e2_multi,
                                                                  logits=self.bilinear_raw_pred)
        self.bilinear_loss = tf.reduce_sum(self.bilinear_elem_loss)

        # Rel Regularization Pipeline
        self.reg_e1_embs = tf.nn.embedding_lookup(self.weights['ent_emb'], self.reg_e1)
        self.reg_rel_embs = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_rel) # self.reg_rel)
        self.reg_seq_0_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_seq_0)
        self.reg_seq_1_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_seq_1)
        self.reg_seq_2_emb = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_seq_2)

        #self.reg_seq_0_bias = tf.nn.embedding_lookup(self.weights['rel_bias'], self.reg_seq_0)
        #self.reg_seq_1_bias = tf.nn.embedding_lookup(self.weights['rel_bias'], self.reg_seq_1)
        #self.reg_seq_2_bias = tf.nn.embedding_lookup(self.weights['rel_bias'], self.reg_seq_2)
        
        #self.reg_loss = self.compute_non_linear_reg_l2()
        self.reg_loss = self.compute_non_linear_bce_reg()
        
        # Optimization
        #self.collective_loss = self.bilinear_loss
        self.bilinear_weighted_loss = self.baseline_weight * self.bilinear_loss
        self.reg_weighted_loss = self.reg_weight * self.reg_loss
        self.collective_loss = self.bilinear_weighted_loss + self.reg_weighted_loss
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.collective_loss)

    # model 2
    def cosine_similarity_regularization(self):
        batch_size = tf.shape(self.reg_seq_0_emb)[0]
        # map similarity to max margin score
        pos_sim_shift = 1./(1. - self.sim_threshold) * math.pi * (self.reg_sim - self.sim_threshold)
        neg_sim_shift = 1./(self.sim_threshold) * math.pi * (self.reg_sim - self.sim_threshold)
        sim_transform = -tf.tanh(pos_sim_shift)
        not_sim_transform = -tf.tanh(neg_sim_shift)
        reg_loss_weight = tf.minimum(sim_transform, not_sim_transform)
        # hard thresholding
        #condition = tf.greater_equal(self.reg_sim, self.sim_threshold)
        #reg_loss_weight = tf.where(condition, tf.ones_like(self.reg_sim)*-1., tf.ones_like(self.reg_sim))
        
        # extract sequences
        reg_seq_0_map = self.reg_seq_0_emb 
        reg_seq_1_map = tf.matmul(reg_seq_0_map, self.reg_seq_1_emb)
        reg_seq_2_map = tf.matmul(reg_seq_1_map, self.reg_seq_2_emb) 

        is_seq_1 = tf.reshape(reg_seq_0_map, [batch_size, 1, self.emb_dim, self.emb_dim])
        is_seq_2 = tf.reshape(reg_seq_1_map, [batch_size, 1, self.emb_dim, self.emb_dim])
        is_seq_3 = tf.reshape(reg_seq_2_map, [batch_size, 1, self.emb_dim, self.emb_dim])
        # filter out dummy sequences
        reg_sequences = tf.boolean_mask(tf.concat([is_seq_1, is_seq_2, is_seq_3], 1), self.reg_seq_mask)
        # normalzie rows for cosine similarity
        reg_sequences = tf.reshape(reg_sequences, [batch_size, self.emb_dim, self.emb_dim])
        reg_rel_embs = tf.reshape(self.reg_rel_embs, [batch_size, self.emb_dim, self.emb_dim])
        reg_pred_normed = tf.nn.l2_normalize(reg_sequences, 2)
        reg_rel_embs_normed = tf.nn.l2_normalize(reg_rel_embs, 2)
        # compute row-wise cosine similarity
        cosine_similarity = tf.reduce_mean(tf.reduce_sum(tf.multiply(reg_pred_normed, reg_rel_embs_normed), 2), 1)
        cosine_similarity = tf.reshape(self.cosine_similarity, [batch_size])
        reg_loss_weight = tf.reshape(self.reg_loss_weight, [batch_size])
        
        reg_loss = tf.reduce_sum(cosine_similarity * reg_loss_weight)
        return reg_loss

    #model 7
    def compute_non_linear_reg_l2(self):
        batch_size = tf.shape(self.reg_seq_0_emb)[0]
        # predict target entity from relation
        rel_e2_pred_raw = tf.einsum('bn,bnm->bm', self.reg_e1_embs, self.reg_rel_embs)
        rel_e2_pred = tf.nn.tanh(rel_e2_pred_raw)
        # unroll sequence through graph
        seq_rel_1_e2 = tf.nn.tanh(tf.einsum('bn,bnm->bm', self.reg_e1_embs, self.reg_seq_0_emb))
        seq_rel_2_e2 = tf.nn.tanh(tf.einsum('bn,bnm->bm', seq_rel_1_e2, self.reg_seq_1_emb))
        seq_rel_3_e2 = tf.nn.tanh(tf.einsum('bn,bnm->bm', seq_rel_2_e2, self.reg_seq_2_emb))
        # prepare predicted e2s for masking
        seq_rel_1_e2 = tf.reshape(seq_rel_1_e2, [batch_size, 1, self.emb_dim])
        seq_rel_2_e2 = tf.reshape(seq_rel_2_e2, [batch_size, 1, self.emb_dim])
        seq_rel_3_e2 = tf.reshape(seq_rel_3_e2, [batch_size, 1, self.emb_dim])
        # extract correct unroll amount
        seq_e2_pred = tf.boolean_mask(tf.concat([seq_rel_1_e2, seq_rel_2_e2, seq_rel_3_e2], 1), self.reg_seq_mask)
        # ensure data is of same shape
        rel_e2_pred = tf.reshape(rel_e2_pred, [batch_size, self.emb_dim])
        seq_2_pred = tf.reshape(seq_e2_pred, [batch_size, self.emb_dim])
        # normalize predictions for cosine similarity
        rel_e2_pred_normed = tf.nn.l2_normalize(rel_e2_pred, 1)
        seq_e2_pred_normed = tf.nn.l2_normalize(seq_e2_pred, 1)
        # compute row-wise cosine similarity
        e2_pred_sim = tf.reduce_sum(tf.multiply(rel_e2_pred_normed, seq_e2_pred_normed), 1)
        # ensure comparison vectors are same shape
        e2_pred_sim = tf.reshape(e2_pred_sim, [batch_size])
        reg_sim = tf.reshape(self.reg_sim, [batch_size])
        # compare similarity against preprocessing similarity
        residual_sim = e2_pred_sim - reg_sim
        reg_loss = tf.reduce_sum(tf.square(residual_sim))
        return reg_loss

    #model 8
    def compute_non_linear_bce_reg(self):
        batch_size = tf.shape(self.reg_seq_0_emb)[0]
        # predict target entity from relation
        rel_e2_pred_raw = tf.einsum('bn,bnm->bm', self.reg_e1_embs, self.reg_rel_embs)
        rel_e2_pred = tf.nn.tanh(rel_e2_pred_raw)
        # unroll sequence through graph
        seq_rel_1_e2 = tf.nn.tanh(tf.einsum('bn,bnm->bm', self.reg_e1_embs, self.reg_seq_0_emb))
        seq_rel_2_e2 = tf.nn.tanh(tf.einsum('bn,bnm->bm', seq_rel_1_e2, self.reg_seq_1_emb))
        seq_rel_3_e2 = tf.nn.tanh(tf.einsum('bn,bnm->bm', seq_rel_2_e2, self.reg_seq_2_emb))
        # prepare predicted e2s for masking
        seq_rel_1_e2 = tf.reshape(seq_rel_1_e2, [batch_size, 1, self.emb_dim])
        seq_rel_2_e2 = tf.reshape(seq_rel_2_e2, [batch_size, 1, self.emb_dim])
        seq_rel_3_e2 = tf.reshape(seq_rel_3_e2, [batch_size, 1, self.emb_dim])
        # extract correct unroll amount
        seq_e2_pred = tf.boolean_mask(tf.concat([seq_rel_1_e2, seq_rel_2_e2, seq_rel_3_e2], 1), self.reg_seq_mask)
        # ensure data is of same shape
        rel_e2_pred = tf.reshape(rel_e2_pred, [batch_size, self.emb_dim])
        seq_2_pred = tf.reshape(seq_e2_pred, [batch_size, self.emb_dim])
        # compute row-wise cosine similarity
        e2_pred_sim = tf.reduce_sum(tf.multiply(rel_e2_pred, seq_e2_pred), 1)
        # ensure comparison vectors are same shape
        e2_pred_sim = tf.reshape(e2_pred_sim, [batch_size])
        reg_sim = tf.reshape(self.reg_sim, [batch_size])
        # compare similarity by BCELoss (Same Scale as with Objective)
        reg_loss = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(multi_class_labels=reg_sim,
                                                                  logits=e2_pred_sim))
        return reg_loss

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

