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
                'seq': tf.int64,
                'sim': tf.float32,
                'seq_mask': tf.int64
            },
            output_shapes={
                'rel': [None],
                'seq': [None, 3],
                'sim': [None],
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
        self.reg_seq = self.next_relreg_sample['seq']
        self.reg_sim = self.next_relreg_sample['sim']
        self.reg_seq_mask = self.next_relreg_sample['seq_mask']

        #self.reg_rel_printed = tf.Print(self.reg_rel, [self.reg_rel])

        # DistMult Prediction Pipeline
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
        self.reg_seq_embs = tf.nn.embedding_lookup(self.weights['rel_emb'], self.reg_seq)
        #self.sequences_unpacked = tf.unstack(self.sequences)
        #tensor_maps = []
        #for batch_idx in range(self.sequences.shape[0]):
         #   scanned_prod = tf.scan(lambda a, b: tf.matmul(a, b), self.sequences_unpacked[batch_idx])
          #  valid_prod = tf.reshape(scanned_prod[self.seq_idx[batch_idx]], [1, self.emb_dim, self.emb_dim])
           # tensor_maps.append(valid_prod)
        
        self.scanned_tensors = tf.map_fn(lambda x: tf.scan(lambda a, b: tf.matmul(a, b), x), self.reg_seq_embs)
        self.reg_predictions = tf.boolean_mask(self.scanned_tensors, self.reg_seq_mask)
        
        
       # self.batch_maps = tf.concat(tensor_maps, 0)
        self.reg_batch_loss = tf.map_fn(lambda x: tf.reduce_sum(x), tf.square(self.reg_rel_embs - self.reg_predictions))
        self.reg_loss = tf.reduce_sum(self.reg_batch_loss * self.reg_sim)

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
        weights['rel_emb'] = tf.get_variable('rel_embeddings',
                                             [self.num_rel, self.emb_dim, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
        return weights

