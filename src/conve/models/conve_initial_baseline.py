from __future__ import absolute_import, division, print_function

import logging
#import math

import tensorflow as tf
import functools
import math

from ..utilities.amsgrad import AMSGradOptimizer


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

class ConvE(object):

    def __init__(self, model_descriptors):
        self.summaries = None
        self.label_smoothing_epsilon = model_descriptors['label_smoothing_epsilon']

        self.num_ent = model_descriptors['num_ent']
        self.num_rel = model_descriptors['num_rel']
        self.emb_size = model_descriptors['emb_size']
        self.input_dropout = model_descriptors['input_dropout']
        self.hidden_dropout = model_descriptors['hidden_dropout']
        self.output_dropout = model_descriptors['output_dropout']

        self._loss_summaries = model_descriptors['add_loss_summaries']
        self._variable_summaries = model_descriptors['add_variable_summaries']
        self._tensor_summaries = model_descriptors['add_tensor_summaries']

        learning_rate = model_descriptors['learning_rate']
        #optimizer = AMSGradOptimizer(learning_rate)

        # Build the graph.
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

        self.struc_iterator_handle = tf.placeholder_with_default("", shape=[])
        self.struc_iterator = tf.data.Iterator.from_string_handle(
            self.struc_iterator_handle,
            output_types={
                'source': tf.int64,
                'target': tf.int64,
                'weight': tf.float32},
            output_shapes={
                'source': [None],
                'target': [None],
                'weight': [None]})

        self.next_input_sample = self.input_iterator.get_next()
        self.next_struc_sample = self.struc_iterator.get_next()

        self.is_train = tf.placeholder_with_default(False, shape=[])
        self.e1 = self.next_input_sample['e1']
        self.rel = self.next_input_sample['rel']
        self.e2_multi = self.next_input_sample['e2_multi1']
        self.source_struc = self.next_struc_sample['source']
        self.target_struc = self.next_struc_sample['target']
        self.weight_struc = self.next_struc_sample['weight']

        self.semant_loss_weight = tf.placeholder(tf.float32)
        self.struct_loss_weight = tf.placeholder(tf.float32)

        #self.variables = self._create_variables()

        self.weights = self.construct_weights()
        #self.optimizer = AMSGradOptimizer(learning_rate) #tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.predictions = self.prediction(self.e1, self.rel)
        self.semantics_loss = self.loss(self.predictions, self.e2_multi)
        # build structure graph
        self.struct_loss = 0#self.structure_loss(self.source_struc, self.target_struc)
        # combine losses according to weights
        #print("here")
        self.loss = ((self.semantics_loss * self.semant_loss_weight) +
                              (self.struct_loss * self.struct_loss_weight))

        self.train_op = self.optimizer.minimize(self.loss)

    # initalize the network weights
    def construct_weights(self):
        weights = {}
        weights['entity_embeddings'] = tf.get_variable(name = 'entity_embeddings',
                                                       dtype=tf.float32,
                                                       shape=[self.num_ent, self.emb_size],
                                                       initializer=tf.contrib.layers.xavier_initializer())
        
        weights['relation_embeddings'] = tf.get_variable("relation_embeddings",
                                  [self.num_rel, self.emb_size],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())

        weights['structure_weights'] = tf.get_variable('structure_weights',
                                                 dtype=tf.float32,
                                                 shape=[self.num_ent, self.emb_size],
                                                 initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(self.emb_size)))

        weights['structure_bias'] = tf.get_variable('structure_bias',
                                                    dtype=tf.float32,
                                                    shape=[self.num_ent],
                                                    initializer=tf.zeros_initializer())

        weights['conv1_weights'] = tf.get_variable("conv1_weights",
                                          [3, 3, 1, 32],
                                          dtype = tf.float32,
                                          initializer = tf.contrib.layers.xavier_initializer())
        weights['conv1_bias'] = tf.get_variable("conv1_bias",
                                                [32],
                                                dtype = tf.float32,
                                                initializer=tf.zeros_initializer())
        weights['fc_weights'] = tf.get_variable("fc_weights",
                                                [10368, self.emb_size],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights['fc_bias'] = tf.get_variable("fc_bias",
                                             [self.emb_size],
                                             dtype=tf.float32,
                                             initializer=tf.zeros_initializer())
        weights['output_bias'] = tf.get_variable("output_bias",
                                   [self.num_ent],
                                   dtype=tf.float32,
                                   initializer=tf.zeros_initializer())
        return weights

    def prediction(self, entities, relations):
        # build graph
        with tf.name_scope('embeddings'):
            variable_summaries(self.weights['entity_embeddings'])
            variable_summaries(self.weights['relation_embeddings'])

        self.e1_embedded = tf.reshape(tf.nn.embedding_lookup(self.weights['entity_embeddings'],
                                                        entities,
                                                        name="e1_embedding"),
                                      [-1, 10, 20, 1])

        self.rel_embedded = tf.reshape(tf.nn.embedding_lookup(self.weights['relation_embeddings'],
                                                         relations,
                                                         name="rel_embedding"),
                                       [-1, 10, 20, 1])

        self.stacked_inputs = tf.concat([self.e1_embedded, self.rel_embedded], 1)
        self.stacked_inputs = tf.contrib.layers.batch_norm(self.stacked_inputs)
        self.stacked_dropout = tf.nn.dropout(self.stacked_inputs, 1 - self.input_dropout)
        # conv1 = tf.layers.conv2d(inputs=stacked_dropout,
        #                          filters=32,
        #                          kernel_size=[3, 3],
        #                          padding='valid')

        with tf.name_scope('convolution1'):
            variable_summaries(self.weights['conv1_weights'])
            variable_summaries(self.weights['conv1_bias'])
            self.conv1 = tf.nn.conv2d(input = self.stacked_dropout,
                                 filter = self.weights['conv1_weights'],
                                 strides = [1, 1, 1, 1],
                                 padding = 'VALID')
            tf.summary.histogram('conv1_transformation', self.conv1)
            self.conv1_plus_bias = self.conv1 + self.weights['conv1_bias']
            tf.summary.histogram('conv1_transformation_plus_bias', self.conv1_plus_bias)
            self.conv1_bn = tf.contrib.layers.batch_norm(self.conv1_plus_bias)
            tf.summary.histogram('conv1_bn', self.conv1_bn)
            self.conv1_relu = tf.nn.relu(self.conv1_bn)
            tf.summary.histogram('conv1_activation', self.conv1_relu)
            self.conv1_dropout = tf.nn.dropout(self.conv1_relu, 1 - self.hidden_dropout)
            tf.summary.histogram('conv1_with_dropout', self.conv1_dropout)
                

        batch_size = batch_size = tf.shape(self.conv1_dropout)[0]
        self.flat_tensor = tf.reshape(self.conv1_dropout, [batch_size, 10368])
        # fc = tf.contrib.layers.fully_connected(flat_tensor, self.emb_size)
        with tf.name_scope('fully_connected_layer'):
            variable_summaries(self.weights['fc_weights'])
            variable_summaries(self.weights['fc_bias'])

            self.fc = tf.matmul(self.flat_tensor, self.weights['fc_weights']) + self.weights['fc_bias']
            tf.summary.histogram('fully_connected_result', self.fc)
            self.fc_dropout = tf.nn.dropout(self.fc, 1 - self.output_dropout)
            tf.summary.histogram('fully_connected_with_dropout', self.fc_dropout)
            self.fc_bn = tf.contrib.layers.batch_norm(self.fc_dropout)
            tf.summary.histogram('fully_connected_with_batch_norm', self.fc_bn)
            #fc_bn = tf.nn.batch_normalization(fc_dropout, [0.], [1.])
            self.fc_relu = self.fc_bn # tf.nn.relu(self.fc_bn)
            tf.summary.histogram('fully_connected_with_activation', self.fc_relu)

        with tf.name_scope('similarity_measures'):
            self.mat_prod = tf.matmul(self.fc_relu, tf.transpose(self.weights['entity_embeddings']))
            self.mat_prod = self.mat_prod + tf.expand_dims(self.weights['output_bias'], 0)
            tf.summary.histogram('semantic_similarities', self.mat_prod)
        return self.mat_prod
        # return tf.sigmoid(self.mat_prod)

    def structure_loss(self, entities, similarity_mat):
        with tf.name_scope('structure_structure'):
            with tf.name_scope('structure_weights'):
                variable_summaries(self.weights['structure_weights'])
                variable_summaries(self.weights['structure_bias'])

            self.e_emb_struc = tf.reshape(tf.nn.embedding_lookup(self.weights['entity_embeddings'],
                                                      entities,
                                                      name="e1_struc_embedding"), (-1, self.emb_size))

            self.e_similarity = tf.matmul(self.e_emb_struc, self.weights['structure_weights'], transpose_b=True)
            self.e_similarity_with_bias = self.e_similarity + tf.expand_dims(self.weights['structure_bias'], 0)
            tf.summary.histogram('structure_similarities', self.e_similarity_with_bias)

            with tf.name_scope('structure_loss'):
                structure_element_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=similarity_mat,
                                                                         logits=self.e_similarity_with_bias)
                struct_loss = tf.reduce_sum(structure_element_loss)
                tf.summary.histogram('loss', struct_loss)

        return struct_loss



    def optimize(self, loss):
        # elem_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.targets, logits = self.prediction)
        # loss = tf.reduce_mean(elem_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(loss)

    def loss(self, predictions, targets):
        with tf.name_scope('semantics_loss'):
            elem_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=targets, logits=predictions)
            loss = tf.reduce_sum(elem_loss)
            tf.summary.histogram('loss', loss)
        return loss
    
    def log_parameters_info(self):
        pass
#    def get_weights(self):
#        return self.weights

#    def set_weights(self, weights):
#        for weight in weights:
#            self.weights[weight] = weights[weight]





