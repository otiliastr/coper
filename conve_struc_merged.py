import math
import tensorflow as tf

from spodernet.utils.logger import Logger

from amsgrad import AMSGradOptimizer

__all__ = ['ConvE']


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
        self.num_ent = model_descriptors['num_ent']
        self.num_rel = model_descriptors['num_rel']
        self.emb_size = model_descriptors['emb_size']

        self._variable_summaries = model_descriptors['add_variable_summaries']
        self._tensor_summaries = model_descriptors['add_tensor_summaries']

        learning_rate = model_descriptors['learning_rate']
        optimizer = AMSGradOptimizer(learning_rate)
        
        # Build the graph.

        self.e1 = tf.placeholder(tf.int32, [None, 1])
        self.rel = tf.placeholder(tf.int32, [None, 1])
        self.e2_multi = tf.placeholder(tf.float32, [None, self.num_ent])
        self.e2_struct = tf.placeholder(tf.float32, [None, self.num_ent])
        self.input_dropout = tf.placeholder(tf.float32)
        self.hidden_dropout = tf.placeholder(tf.float32)
        self.output_dropout = tf.placeholder(tf.float32)
        self.semant_loss_weight = tf.placeholder(tf.float32)
        self.struct_loss_weight = tf.placeholder(tf.float32)

        # We are using resource variables because due to
        # some implementation details, this allows us to
        # better utilize GPUs while training.
        with tf.variable_scope('variables', use_resource=True):
            self.variables = self._create_variables()

        ent_emb = self.variables['ent_emb']
        rel_emb = self.variables['rel_emb']
        e1_emb = tf.nn.embedding_lookup(ent_emb, self.e1, name='e1_emb')
        rel_emb = tf.nn.embedding_lookup(rel_emb, self.rel, name='rel_emb')
        predictions = self._create_predictions(e1_emb, rel_emb)
        semant_loss = self._create_semant_loss(predictions, self.e2_multi)
        struct_loss = self._create_struct_loss(e1_emb, self.e2_struct)

        # Combine the two losses according to the provided weights.
        self.loss = (
            (semant_loss * self.semant_loss_weight) +
            (struct_loss * self.struct_loss_weight))

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
            'structure_bias': structure_bias
        }

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

        stacked_emb = tf.concat([e1_emb, rel_emb], 1)
        stacked_emb = tf.contrib.layers.batch_norm(stacked_emb)
        stacked_emb = tf.nn.dropout(stacked_emb, 1-self.input_dropout)

        with tf.name_scope('conv1'):
            weights = self.variables['conv1_weights']
            bias = self.variables['conv1_bias']
            conv1 = tf.nn.conv2d(
                input=stacked_emb, filter=weights,
                strides=[1, 1, 1, 1], padding='VALID')
            conv1_plus_bias = conv1 + bias
            conv1_bn = tf.contrib.layers.batch_norm(conv1_plus_bias)
            conv1_relu = tf.nn.relu(conv1_bn)
            conv1_dropout = tf.nn.dropout(conv1_relu, 1-self.hidden_dropout)

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
            fc_dropout = tf.nn.dropout(fc, 1 - self.output_dropout)
            fc_bn = tf.contrib.layers.batch_norm(fc_dropout)
            fc_relu = fc_bn # tf.nn.relu(fc_bn)

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
                tf.losses.sigmoid_cross_entropy(targets, predictions))
            
            if self._tensor_summaries:
                _create_summaries('loss', semant_loss)
        return semant_loss

    def _create_struct_loss(self, e1_emb, ent_similarities):
        with tf.name_scope('struct_loss'):
            weights = self.variables['structure_weights']
            bias = self.variables['structure_bias']
            e1_emb = tf.reshape(e1_emb, (-1, self.emb_size))
            e1_sim = tf.matmul(e1_emb, weights, transpose_b=True)
            e1_sim = e1_sim + tf.expand_dims(bias, 0)

            if self._tensor_summaries:
                _create_summaries('e1_struct_sim', e1_sim)

            struct_loss = tf.reduce_sum(
                tf.losses.sigmoid_cross_entropy(ent_similarities, e1_sim))

            if self._tensor_summaries:
                _create_summaries('loss', struct_loss)
        return struct_loss

    def log_parameters(self):
        """Logs the trainable parameters of this model,
        along with their shapes.
        """
        Logger.info('Trainable parameters:')
        num_parameters = 0
        # TODO: This is not entirely correct. You have to be careful about which graph you compute this for.
        for variable in tf.trainable_variables():
            Logger.info('\t%s %s' % (variable.name, variable.shape))
            num_parameters += variable.shape.num_elements()
        Logger.info('Number of trainable parameters: %d' % num_parameters)
