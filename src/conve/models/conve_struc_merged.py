from __future__ import absolute_import, division, print_function

import logging
import math
import tensorflow as tf

from ..utilities.amsgrad import AMSGradOptimizer

__all__ = ['ConvE']

LOGGER = logging.getLogger(__name__)


def _create_summaries(name, tensor):
    """Creates various summaries for the provided tensor, 
    which are useful for TensorBoard visualizations).
    """
    with tf.name_scope(name + '/summaries'):
        mean = tf.reduce_mean(tensor)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(tensor))
        tf.summary.scalar('min', tf.reduce_min(tensor))
        tf.summary.histogram('histogram', tensor)


class ConvE(object):
    def __init__(self, model_descriptors):
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
        optimizer = AMSGradOptimizer(learning_rate)

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
                'e1': tf.int64,
                'e2': tf.int64},
            output_shapes={
                'e1': [None],
                'e2': [None]})

        self.next_input_sample = self.input_iterator.get_next()
        self.next_struc_sample = self.struc_iterator.get_next()

        self.is_train = tf.placeholder_with_default(False, shape=[])
        self.e1 = self.next_input_sample['e1']
        self.rel = self.next_input_sample['rel']
        self.e2_multi = self.next_input_sample['e2_multi1']
        self.e1_struc = self.next_struc_sample['e1']
        self.e2_struc = self.next_struc_sample['e2']

        self.semant_loss_weight = tf.placeholder(tf.float32)
        self.struct_loss_weight = tf.placeholder(tf.float32)

        self.variables = self._create_variables()

        ent_emb = self.variables['ent_emb']
        rel_emb = self.variables['rel_emb']
        e1_emb = tf.nn.embedding_lookup(ent_emb, self.e1, name='e1_emb')
        e1_struc_emb = tf.nn.embedding_lookup(ent_emb, self.e1_struc, name='e1_struc_emb')
        e2_struc_emb = tf.nn.embedding_lookup(ent_emb, self.e2_struc, name='e2_struc_emb')
        rel_emb = tf.nn.embedding_lookup(rel_emb, self.rel, name='rel_emb')
        self.predictions = self._create_predictions(e1_emb, rel_emb)
        semant_loss = self._create_semant_loss(self.predictions, self.e2_multi)
        struct_loss = self._create_struct_loss(e1_struc_emb, e2_struc_emb)

        # Combine the two losses according to the provided weights.
        self.loss = (
            (semant_loss * self.semant_loss_weight) + 
            (struct_loss * self.struct_loss_weight))

        # The following control dependency is needed in order for batch
        # normalization to work correctly.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        self.summaries = tf.summary.merge_all()

    def _create_variables(self):
        """Creates the network variables and returns them in a dictionary."""
        ent_emb = tf.get_variable(
            name='ent_emb', dtype=tf.float32,
            shape=[self.num_ent, self.emb_size],
            initializer=tf.contrib.layers.xavier_initializer())
        rel_emb = tf.get_variable(
            name='rel_emb', dtype=tf.float32,
            shape=[self.num_rel, self.emb_size],
            initializer=tf.contrib.layers.xavier_initializer())

        conv1_weights = tf.get_variable(
            name='conv1_weights', dtype=tf.float32,
            shape=[3, 3, 1, 32],
            initializer=tf.contrib.layers.xavier_initializer())
        conv1_bias = tf.get_variable(
            name='conv1_bias', dtype=tf.float32,
            shape=[32], initializer=tf.zeros_initializer())
        fc_weights = tf.get_variable(
            name='fc_weights', dtype=tf.float32,
            shape=[10368, self.emb_size],
            initializer=tf.contrib.layers.xavier_initializer())
        fc_bias = tf.get_variable(
            name='fc_bias', dtype=tf.float32,
            shape=[self.emb_size], initializer=tf.zeros_initializer())

        output_bias = tf.get_variable(
            name='output_bias', dtype=tf.float32,
            shape=[1, self.num_ent], initializer=tf.zeros_initializer())

        structure_weights = tf.get_variable(
            name='structure_weights', dtype=tf.float32,
            shape=[self.num_ent, self.emb_size],
            initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(self.emb_size)))
        structure_bias = tf.get_variable(
            name='structure_bias', dtype=tf.float32,
            shape=[self.num_ent], initializer=tf.zeros_initializer())
        
        variables = {
            'ent_emb': ent_emb,
            'rel_emb': rel_emb,
            'conv1_weights': conv1_weights,
            'conv1_bias': conv1_bias,
            'fc_weights': fc_weights,
            'fc_bias': fc_bias,
            'output_bias': output_bias,
            'structure_weights': structure_weights,
            'structure_bias': structure_bias}

        if self._variable_summaries:
            _create_summaries('emb/ent', ent_emb)
            _create_summaries('emb/rel', rel_emb)
            _create_summaries('predictions/conv1_weights', conv1_weights)
            _create_summaries('predictions/conv1_bias', conv1_bias)
            _create_summaries('predictions/fc_weights', fc_weights)
            _create_summaries('predictions/fc_bias', fc_bias)
            _create_summaries('predictions/output_bias', output_bias)
            _create_summaries('structure/weights', structure_weights)
            _create_summaries('structure/bias', structure_bias)
        
        return variables

    def _create_predictions(self, e1_emb, rel_emb):
        e1_emb = tf.reshape(e1_emb, [-1, 10, 20, 1])
        rel_emb = tf.reshape(rel_emb, [-1, 10, 20, 1])

        is_train_float = tf.cast(self.is_train, tf.float32)

        stacked_emb = tf.concat([e1_emb, rel_emb], 1)
        stacked_emb = tf.layers.batch_normalization(
            stacked_emb, momentum=0.1, reuse=tf.AUTO_REUSE,
            training=self.is_train, fused=True, name='StackedEmbBN')
        stacked_emb = tf.nn.dropout(
            stacked_emb, 1 - (self.input_dropout * is_train_float))

        with tf.name_scope('conv1'):
            weights = self.variables['conv1_weights']
            bias = self.variables['conv1_bias']
            conv1 = tf.nn.conv2d(
                input=stacked_emb, filter=weights,
                strides=[1, 1, 1, 1], padding='VALID')
            conv1_plus_bias = conv1 + bias
            conv1_bn = tf.layers.batch_normalization(
                conv1_plus_bias, momentum=0.1, scale=False, reuse=tf.AUTO_REUSE,
                training=self.is_train, fused=True, name='Conv1BN')
            conv1_relu = tf.nn.relu(conv1_bn)
            conv1_dropout = tf.nn.dropout(
                conv1_relu, 1 - (self.hidden_dropout * is_train_float))

            if self._tensor_summaries:
                _create_summaries('conv1', conv1)
                _create_summaries('conv1_plus_bias', conv1_plus_bias)
                _create_summaries('conv1_bn', conv1_bn)
                _create_summaries('conv1_activation', conv1_relu)
                _create_summaries('conv1_with_dropout', conv1_dropout)

        with tf.name_scope('fc_layer'):
            weights = self.variables['fc_weights']
            bias = self.variables['fc_bias']
            batch_size = tf.shape(conv1_dropout)[0]
            fc_input = tf.reshape(conv1_dropout, [batch_size, -1])
            fc = tf.matmul(fc_input, weights) + bias
            fc_dropout = tf.nn.dropout(
                fc, 1 - (self.output_dropout * is_train_float))
            fc_bn = tf.layers.batch_normalization(
                fc_dropout, momentum=0.1, scale=False, reuse=tf.AUTO_REUSE,
                training=self.is_train, fused=True, name='FCBN')
            fc_relu = tf.nn.relu(fc_bn)

            if self._tensor_summaries:
                _create_summaries('fc_result', fc)
                _create_summaries('fc_with_dropout', fc_dropout)
                _create_summaries('fc_with_batch_norm', fc_bn)
                _create_summaries('fc_with_activation', fc_relu)

        with tf.name_scope('output_layer'):
            ent_emb = self.variables['ent_emb']
            bias = self.variables['output_bias']
            ent_emb_t = tf.transpose(ent_emb)
            predictions = tf.matmul(fc_relu, ent_emb_t)
            predictions = predictions + bias

            if self._tensor_summaries:
                _create_summaries('predictions', predictions)

        return predictions

    def _create_semant_loss(self, predictions, targets):
        with tf.name_scope('semant_loss'):
            semant_loss = tf.reduce_sum(
                tf.losses.sigmoid_cross_entropy(
                    targets, predictions,
                    label_smoothing=self.label_smoothing_epsilon))

            if self._loss_summaries:
                tf.summary.scalar('loss', semant_loss)
        return semant_loss

    def _create_struct_loss(self, e1_emb, e2_emb):
        with tf.name_scope('struct_loss'):
            e1_emb = tf.nn.l2_normalize(e1_emb, axis=1)
            e2_emb = tf.nn.l2_normalize(e2_emb, axis=1)
            struct_loss = tf.losses.cosine_distance(e1_emb, e2_emb, axis=1)

            if self._loss_summaries:
                tf.summary.scalar('loss', struct_loss)
        return struct_loss

    def log_parameters_info(self):
        """Logs the trainable parameters of this model,
        along with their shapes.
        """
        LOGGER.info('Trainable parameters:')
        num_parameters = 0
        # TODO: This is not entirely correct. You have to be careful about which graph you compute this for.
        for variable in tf.trainable_variables():
            LOGGER.info('\t%s %s' % (variable.name, variable.shape))
            num_parameters += variable.shape.num_elements()
        LOGGER.info('Number of trainable parameters: %d' % num_parameters)
