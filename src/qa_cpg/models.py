from __future__ import absolute_import, division, print_function

import logging
import math
import tensorflow as tf

from functools import reduce
from operator import mul

from .utils.amsgrad import AMSGradOptimizer

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


def batch_gather(params, indices, name=None):
    """Gather slices from `params` according to `indices` with leading batch dims.
    This operation assumes that the leading dimensions of `indices` are dense,
    and the gathers on the axis corresponding to the last dimension of `indices`.
    More concretely it computes:
    result[i1, ..., in] = params[i1, ..., in-1, indices[i1, ..., in]]
    Therefore `params` should be a Tensor of shape [A1, ..., AN, B1, ..., BM],
    `indices` should be a Tensor of shape [A1, ..., AN-1, C] and `result` will be
    a Tensor of size `[A1, ..., AN-1, C, B1, ..., BM]`.
    In the case in which indices is a 1D tensor, this operation is equivalent to
    `tf.gather`.
    See also `tf.gather` and `tf.gather_nd`.
    Args:
      params: A Tensor. The tensor from which to gather values.
      indices: A Tensor. Must be one of the following types: int32, int64. Index
          tensor. Must be in range `[0, params.shape[axis]`, where `axis` is the
          last dimension of `indices` itself.
      name: A name for the operation (optional).
    Returns:
      A Tensor. Has the same type as `params`.
    Raises:
      ValueError: if `indices` has an unknown shape.
    """

    with tf.name_scope(name):
        indices = tf.convert_to_tensor(indices, name="indices")
        params = tf.convert_to_tensor(params, name="params")
        indices_shape = tf.shape(indices)
        params_shape = tf.shape(params)
        ndims = indices.shape.ndims
        if ndims is None:
            raise ValueError("batch_gather does not allow indices with unknown "
                             "shape.")
        batch_indices = indices
        accum_dim_value = 1
        for dim in range(ndims-1, 0, -1):
            dim_value = params_shape[dim-1]
            accum_dim_value *= params_shape[dim]
            dim_indices = tf.range(0, dim_value, 1)
            dim_indices *= accum_dim_value
            dim_shape = tf.stack([1] * (dim - 1) + [dim_value] + [1] * (ndims - dim),
                              axis=0)
            batch_indices += tf.reshape(dim_indices, dim_shape)

        flat_indices = tf.reshape(batch_indices, [-1])
        outer_shape = params_shape[ndims:]
        flat_inner_shape = tf.reduce_prod(params_shape[:ndims], [0], False)

        flat_params = tf.reshape(
            params, tf.concat([[flat_inner_shape], outer_shape], axis=0))
        flat_result = tf.gather(flat_params, flat_indices)
        result = tf.reshape(flat_result, tf.concat([indices_shape, outer_shape], axis=0))
        final_shape = indices.get_shape()[:ndims-1].merge_with(
            params.get_shape()[:ndims -1])
        final_shape = final_shape.concatenate(indices.get_shape()[ndims-1])
        final_shape = final_shape.concatenate(params.get_shape()[ndims:])
        result.set_shape(final_shape)
        return result


class ContextualParameterGenerator(object):
    def __init__(self, context_size, name, dtype, shape, initializer, dropout=0.5, use_batch_norm=False):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.num_elements = reduce(mul, self.shape, 1)

        # Create the projection matrices.
        self.projections = []
        in_size = context_size[0]
        for i, n in enumerate(context_size[1:] + [self.num_elements]):
            self.projections.append(
                tf.get_variable(
                    name='%s/CPG/Projection%d' % (name, i),
                    dtype=tf.float32,
                    shape=[in_size, n],
                    initializer=initializer))
            in_size = n

    def generate(self, context, is_train):
        # Generate the parameter values.
        generated_value = context
        for i, projection in enumerate(self.projections[:-1]):
            generated_value = tf.matmul(generated_value, projection)
            if self.use_batch_norm:
                generated_value = tf.layers.batch_normalization(
                    generated_value, momentum=0.1, scale=False, reuse=tf.AUTO_REUSE,
                    training=False, fused=True, name='%s/CPG/Projection%d/BatchNorm' % (self.name, i))
            generated_value = tf.nn.relu(generated_value)
            generated_value = tf.nn.dropout(
                generated_value, 1 - (self.dropout * tf.cast(is_train, tf.float32)))

        generated_value = tf.matmul(generated_value, self.projections[-1])

        # Reshape and cast to the requested type.
        generated_value = tf.reshape(generated_value, [-1] + self.shape)
        generated_value = tf.cast(generated_value, self.dtype)

        return generated_value


class ConvE(object):
    def __init__(self, model_descriptors):
        self.label_smoothing_epsilon = model_descriptors['label_smoothing_epsilon']

        self.num_ent = model_descriptors['num_ent']
        self.num_rel = model_descriptors['num_rel']
        self.ent_emb_size = model_descriptors['ent_emb_size']
        self.rel_emb_size = model_descriptors['rel_emb_size']

        self.conv_filter_height = model_descriptors.get('conv_filter_height', 3)
        self.conv_filter_width = model_descriptors.get('conv_filter_width', 3)
        self.conv_num_channels = model_descriptors.get('conv_num_channels', 32)

        self.concat_rel = model_descriptors.get('concat_rel', False)
        self.context_rel_conv = model_descriptors.get('context_rel_conv', None)
        self.context_rel_out = model_descriptors.get('context_rel_out', None)
        self.context_rel_dropout = model_descriptors.get('context_rel_dropout', 0.0)
        self.context_rel_use_batch_norm = model_descriptors.get('context_rel_use_batch_norm', False)

        self.input_dropout = model_descriptors['input_dropout']
        self.hidden_dropout = model_descriptors['hidden_dropout']
        self.output_dropout = model_descriptors['output_dropout']

        self._loss_summaries = model_descriptors['add_loss_summaries']
        self._variable_summaries = model_descriptors['add_variable_summaries']
        self._tensor_summaries = model_descriptors['add_tensor_summaries']

        learning_rate = model_descriptors['learning_rate']
        optimizer = AMSGradOptimizer(learning_rate)

        # Build the graph.
        with tf.device('/CPU:0'):
            self.input_iterator_handle = tf.placeholder(tf.string, shape=[])
            self.input_iterator = tf.data.Iterator.from_string_handle(
                self.input_iterator_handle,
                output_types={
                    'e1': tf.int64,
                    'e2': tf.int64,
                    'rel': tf.int64,
                    'e2_multi': tf.float32,
                    'lookup_values': tf.int32
                },
                output_shapes={
                    'e1': [None],
                    'e2': [None],
                    'rel': [None],
                    'e2_multi': [None, self.num_ent],
                    'lookup_values': [None, None]
                })

            self.eval_iterator_handle = tf.placeholder(tf.string, shape=[])
            self.eval_iterator = tf.data.Iterator.from_string_handle(
                self.eval_iterator_handle,
                output_types={
                    'e1': tf.int64,
                    'rel': tf.int64,
                    'e2': tf.int64,
                    'e2_multi1': tf.float32},
                output_shapes={
                    'e1': [None],
                    'rel': [None],
                    'e2': [None],
                    'e2_multi1': [None, self.num_ent]
                })
        
        # Get the next samples from the training and the evaluation iterators.
        self.next_input_sample = self.input_iterator.get_next()
        self.next_eval_sample = self.eval_iterator.get_next()

        # Training Data
        self.is_train = tf.placeholder_with_default(False, shape=[])
        self.e1 = self.next_input_sample['e1']
        self.rel = self.next_input_sample['rel']
        self.e2 = self.next_input_sample['e2']
        self.e2_multi = self.next_input_sample['e2_multi']
        self.obj_lookup_values = self.next_input_sample['lookup_values']

        # Evaluation Data
        self.eval_e1 = self.next_eval_sample['e1']
        self.eval_rel = self.next_eval_sample['rel']
        self.eval_e2 = self.next_eval_sample['e2']
        self.eval_e2_multi = self.next_eval_sample['e2_multi1']

        self.variables = self._create_variables()

        ent_emb = self.variables['ent_emb']
        rel_emb = self.variables['rel_emb']
        # objective embeddings
        conve_e1_emb = tf.nn.embedding_lookup(ent_emb, self.e1, name='e1_emb')
        conve_rel_emb = tf.nn.embedding_lookup(rel_emb, self.rel, name='rel_emb')
        # eval embeddings
        eval_e1_emb = tf.nn.embedding_lookup(ent_emb, self.eval_e1, name='eval_e1_emb')
        eval_rel_emb = tf.nn.embedding_lookup(rel_emb, self.eval_rel, name='eval_rel_emb')

        self.prediction_vector = self._create_predictions(conve_e1_emb, conve_rel_emb)
        self.predictions = self._compute_likelihoods(self.prediction_vector, self.obj_lookup_values)

        self.eval_prediction_vector = self._create_predictions(eval_e1_emb, eval_rel_emb)
        self.eval_predictions = self._compute_likelihoods(self.eval_prediction_vector)

        self.loss = self._create_loss(self.predictions, self.e2_multi)

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
            shape=[self.num_ent, self.ent_emb_size],
            initializer=tf.contrib.layers.xavier_initializer())

        rel_emb = tf.get_variable(
            name='rel_emb', dtype=tf.float32,
            shape=[self.num_rel, self.rel_emb_size],
            initializer=tf.contrib.layers.xavier_initializer())

        if self.context_rel_conv is not None:
            conv1_weights = ContextualParameterGenerator(
                context_size=[self.rel_emb_size] + self.context_rel_conv, 
                name='conv1_weights',
                dtype=tf.float32,
                shape=[self.conv_filter_height, self.conv_filter_width, 1, self.conv_num_channels],
                dropout=self.context_rel_dropout,
                use_batch_norm=self.context_rel_use_batch_norm,
                initializer=tf.contrib.layers.xavier_initializer())
            conv1_bias = ContextualParameterGenerator(
                context_size=[self.rel_emb_size] + self.context_rel_conv,
                name='conv1_bias',
                dtype=tf.float32,
                shape=[self.conv_num_channels],
                dropout=self.context_rel_dropout,
                use_batch_norm=self.context_rel_use_batch_norm,
                initializer=tf.zeros_initializer())
        else:
            conv1_weights = tf.get_variable(
                name='conv1_weights', dtype=tf.float32,
                shape=[self.conv_filter_height, self.conv_filter_width, 1, self.conv_num_channels],
                initializer=tf.contrib.layers.xavier_initializer())
            conv1_bias = tf.get_variable(
                name='conv1_bias', dtype=tf.float32,
                shape=[self.conv_num_channels], initializer=tf.zeros_initializer())
        
        # Calculating the size of the convolution layer output.
        conv_in_height = 10
        conv_in_width = self.ent_emb_size // 10
        if self.context_rel_conv is None and self.context_rel_out is None:
            conv_in_height += 10
        conv_out_height = math.ceil(float(conv_in_height - self.conv_filter_height + 1))
        conv_out_width = math.ceil(float(conv_in_width - self.conv_filter_width + 1))

        fc_input_size = conv_out_height * conv_out_width * self.conv_num_channels
        if self.concat_rel:
            fc_input_size += self.rel_emb_size

        if self.context_rel_out is not None:
            fc_weights = ContextualParameterGenerator(
                context_size=[self.rel_emb_size] + self.context_rel_out,
                name='fc_weights',
                dtype=tf.float32,
                shape=[fc_input_size, self.ent_emb_size],
                dropout=self.context_rel_dropout,
                use_batch_norm=self.context_rel_use_batch_norm,
                initializer=tf.contrib.layers.xavier_initializer())
            fc_bias = ContextualParameterGenerator(
                context_size=[self.rel_emb_size] + self.context_rel_out, 
                name='fc_bias', 
                dtype=tf.float32,
                shape=[self.ent_emb_size],
                dropout=self.context_rel_dropout,
                use_batch_norm=self.context_rel_use_batch_norm,
                initializer=tf.zeros_initializer())
        else:
            fc_weights = tf.get_variable(
                name='fc_weights', dtype=tf.float32,
                shape=[fc_input_size, self.ent_emb_size],
                initializer=tf.contrib.layers.xavier_initializer())
            fc_bias = tf.get_variable(
                name='fc_bias', dtype=tf.float32,
                shape=[self.ent_emb_size], initializer=tf.zeros_initializer())

        variables = {
            'ent_emb': ent_emb,
            'rel_emb': rel_emb,
            'conv1_weights': conv1_weights,
            'conv1_bias': conv1_bias,
            'fc_weights': fc_weights,
            'fc_bias': fc_bias}

        if self._variable_summaries:
            _create_summaries('emb/ent', ent_emb)
            _create_summaries('emb/rel', rel_emb)
            _create_summaries('predictions/conv1_weights', conv1_weights)
            _create_summaries('predictions/conv1_bias', conv1_bias)
            _create_summaries('predictions/fc_weights', fc_weights)
            _create_summaries('predictions/fc_bias', fc_bias)
        
        return variables

    def _get_conv_params(self, rel_emb, is_train):
        weights = self.variables['conv1_weights']
        bias = self.variables['conv1_bias']
        if self.context_rel_conv is not None:
            weights = weights.generate(rel_emb, is_train)
            bias = bias.generate(rel_emb, is_train)
        return (weights, bias)

    def _get_fc_params(self, rel_emb, is_train):
        weights = self.variables['fc_weights']
        bias = self.variables['fc_bias']
        if self.context_rel_out is not None:
            weights = weights.generate(rel_emb, is_train)
            bias = bias.generate(rel_emb, is_train)
        return (weights, bias)

    def _create_predictions(self, e1_emb, rel_emb):
        e1_emb = tf.reshape(e1_emb, [-1, 10, self.ent_emb_size // 10, 1])

        is_train_float = tf.cast(self.is_train, tf.float32)

        if self.context_rel_conv is None and self.context_rel_out is None:
            reshaped_rel_emb = tf.reshape(rel_emb, [-1, 10, self.rel_emb_size // 10, 1])
            stacked_emb = tf.concat([e1_emb, reshaped_rel_emb], 1)
        else:
            stacked_emb = e1_emb
        
        stacked_emb = tf.layers.batch_normalization(
            stacked_emb, momentum=0.1, reuse=tf.AUTO_REUSE,
            training=False, fused=True, name='StackedEmbBN')
        stacked_emb = tf.nn.dropout(
            stacked_emb, 1 - (self.input_dropout * is_train_float))

        with tf.name_scope('conv1'):
            weights, bias = self._get_conv_params(rel_emb, self.is_train)
            if self.context_rel_conv is not None:
                def conv(pair):
                    return (tf.nn.conv2d(
                        input=pair[0][None], filter=pair[1],
                        strides=[1, 1, 1, 1], padding='VALID')[0], tf.zeros([]))
                conv1 = tf.map_fn(fn=conv, elems=(stacked_emb, weights))[0]
                conv1_plus_bias = conv1 + bias[:, None, None, :]
            else:
                conv1 = tf.nn.conv2d(
                    input=stacked_emb, filter=weights,
                    strides=[1, 1, 1, 1], padding='VALID')
                conv1_plus_bias = conv1 + bias
            conv1_bn = tf.layers.batch_normalization(
                conv1_plus_bias, momentum=0.1, scale=False, reuse=tf.AUTO_REUSE,
                training=False, fused=True, name='Conv1BN')
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
            weights, bias = self._get_fc_params(rel_emb, self.is_train)
            batch_size = tf.shape(conv1_dropout)[0]

            fc_input = tf.reshape(conv1_dropout, [batch_size, -1])

            if self.concat_rel:
                fc_input = tf.concat([fc_input, rel_emb], axis=1)

            if self.context_rel_out is None:
                fc = tf.matmul(fc_input, weights) + bias
            else:
                fc = tf.matmul(fc_input[:, None, :], weights)[:, 0, :] + bias

            fc_dropout = tf.nn.dropout(
                fc, 1 - (self.output_dropout * is_train_float))
            fc_bn = tf.layers.batch_normalization(
                fc_dropout, momentum=0.1, scale=False, reuse=tf.AUTO_REUSE,
                training=False, fused=True, name='FCBN')

            if self._tensor_summaries:
                _create_summaries('fc_result', fc)
                _create_summaries('fc_with_dropout', fc_dropout)
                _create_summaries('fc_with_batch_norm', fc_bn)
        
        return fc_bn

    def _compute_likelihoods(self, prediction_relu, ent_indices=None):
        if self._tensor_summaries:
            _create_summaries('fc_with_activation', prediction_relu)

        with tf.name_scope('output_layer'):
            if ent_indices is None:
                ent_emb = self.variables['ent_emb']
                ent_emb_t = tf.transpose(ent_emb)
                predictions = tf.matmul(prediction_relu, ent_emb_t)
            else:
                ent_emb = tf.gather(self.variables['ent_emb'], ent_indices) # Returns shape [BatchSize, NumSamples, EmbSize]
                ent_emb_t = tf.transpose(ent_emb, [0, 2, 1])
                predictions = tf.matmul(prediction_relu[:, None, :], ent_emb_t)[:, 0, :]
            if self._tensor_summaries:
                _create_summaries('predictions', predictions)
        return predictions

    def _create_loss(self, predictions, targets):
        with tf.name_scope('loss'):
            loss = tf.reduce_sum(
                tf.losses.sigmoid_cross_entropy(
                    targets, predictions,
                    label_smoothing=self.label_smoothing_epsilon))

            if self._loss_summaries:
                tf.summary.scalar('loss', loss)
        return loss
