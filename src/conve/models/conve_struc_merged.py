from __future__ import absolute_import, division, print_function

import logging
import math
import tensorflow as tf

from functools import reduce
from operator import mul

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
    def __init__(self, context_size, name, dtype, shape, inner_dims=None):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.num_elements = reduce(mul, self.shape, 1)

        # Create the projection matrices.
        if inner_dims is None:
            projection = tf.get_variable(
                name='%s/CPG/Projection' % name, 
                dtype=tf.float32,
                shape=[context_size, self.num_elements],
                initializer=tf.contrib.layers.xavier_initializer())
            self.projections = [projection]
        else:
            self.projections = []
            in_size = context_size
            for i, n in enumerate(inner_dims + [self.num_elements]):
                self.projections.append(
                    tf.get_variable(
                        name='%s/CPG/Projection%d' % (name, i), 
                        dtype=tf.float32,
                        shape=[in_size, n],
                        initializer=tf.contrib.layers.xavier_initializer()))
                in_size = n

    def generate(self, context):
        # Generate the parameter values.
        generated_value = context
        for projection in self.projections:
            generated_value = tf.matmul(generated_value, projection)

        # Reshape and cast to the requested type.
        generated_value = tf.reshape(generated_value, [-1] + self.shape)
        generated_value = tf.cast(generated_value, self.dtype)

        return generated_value


class ConvE(object):
    def __init__(self, model_descriptors):
        self.concat_rel = model_descriptors.get('concat_rel', False)
        self.context_rel = model_descriptors.get('context_rel', 4)

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
       
        self.obj_weight = tf.placeholder(tf.float32)
        self.reg_weight = tf.placeholder(tf.float32)
        self.seq_weight = tf.placeholder(tf.float32)
        self.dist_weight = tf.placeholder(tf.float32)
        self.epsilon = tf.placeholder(tf.float32)
        self.use_ball = tf.placeholder(tf.bool)
        # Build the graph.
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

        self.relreg_iterator_handle = tf.placeholder_with_default("", shape=[])
        self.relreg_iterator = tf.data.Iterator.from_string_handle(
            self.relreg_iterator_handle,
            output_types={
                'e1': tf.int64,
                'rel': tf.int64,
                'seq_0': tf.int64,
                'seq_1': tf.int64,
                'seq_2': tf.int64,
                'sim': tf.float32,
                'seq_mask': tf.int64,
                'rel_e2_multi': tf.float32,
                'seq_e2_multi': tf.float32,
                'lookup_values': tf.int32
            },
            output_shapes={
                'e1': [None],
                'rel': [None],
                'seq_0': [None],
                'seq_1': [None],
                'seq_2': [None],
                'sim': [None],
                'seq_mask': [None, 3],
                'rel_e2_multi': [None, self.num_ent],
                'seq_e2_multi': [None, self.num_ent],
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
                #'truth_scores': tf.float32},
            output_shapes={
                'e1': [None],
                'rel': [None],
                'e2': [None],
                'e2_multi1': [None, self.num_ent]
            })
        
        # get next samples from iterators
        self.next_input_sample = self.input_iterator.get_next()
        self.next_relreg_sample = self.relreg_iterator.get_next()
        self.next_eval_sample = self.eval_iterator.get_next()
        # get obj samples
        self.is_train = tf.placeholder_with_default(False, shape=[])
        self.e1 = self.next_input_sample['e1']
        self.rel = self.next_input_sample['rel']
        self.e2 = self.next_input_sample['e2']
        #self.truth_scores = self.next_input_sample['truth_scores']
        self.e2_multi = self.next_input_sample['e2_multi']
        self.obj_lookup_values = self.next_input_sample['lookup_values']
        # get reg samples
        self.reg_e1 = self.next_relreg_sample['e1']
        self.reg_rel = self.next_relreg_sample['rel']
        self.reg_seq_0 = self.next_relreg_sample['seq_0']
        self.reg_seq_1 = self.next_relreg_sample['seq_1']
        self.reg_seq_2 = self.next_relreg_sample['seq_2']
        self.reg_sim = self.next_relreg_sample['sim']
        #self.reg_agg_sim = self.next_relreg_sample['agg_sim']
        self.reg_seq_mask = self.next_relreg_sample['seq_mask']
        self.reg_seq_e2_multi = self.next_relreg_sample['seq_e2_multi']
        self.reg_rel_e2_multi = self.next_relreg_sample['rel_e2_multi']
        self.reg_lookup_values = self.next_relreg_sample['lookup_values']
        # get eval samples
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

        filtered_targets = batch_gather(self.e2_multi, self.obj_lookup_values)
        self.collective_loss = self._create_loss(self.predictions, filtered_targets)

        # The following control dependency is needed in order for batch
        # normalization to work correctly.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.collective_loss)
        self.summaries = tf.summary.merge_all()

    def _create_variables(self):
        """Creates the network variables and returns them in a dictionary."""
        ent_emb = tf.get_variable(
            name='ent_emb', dtype=tf.float32,
            shape=[self.num_ent, self.emb_size],
            initializer=tf.contrib.layers.xavier_initializer())
        
        rel_emb_size = self.context_rel if self.context_rel is not None else self.emb_size

        rel_emb = tf.get_variable(
            name='rel_emb', dtype=tf.float32,
            shape=[self.num_rel, rel_emb_size],
            initializer=tf.contrib.layers.xavier_initializer())

        conv1_weights = tf.get_variable(
            name='conv1_weights', dtype=tf.float32,
            shape=[3, 3, 1, 32],
            initializer=tf.contrib.layers.xavier_initializer())
        conv1_bias = tf.get_variable(
            name='conv1_bias', dtype=tf.float32,
            shape=[32], initializer=tf.zeros_initializer())

        fc_input_size = 4608 # 10368
        if self.concat_rel:
            fc_input_size += self.emb_size

        if self.context_rel is not None:
            fc_weights = ContextualParameterGenerator(
                context_size=rel_emb_size, 
                name='fc_weights', 
                dtype=tf.float32, 
                shape=[fc_input_size, self.emb_size])
            fc_bias = ContextualParameterGenerator(
                context_size=rel_emb_size, 
                name='fc_bias', 
                dtype=tf.float32, 
                shape=[self.emb_size])
        else:
            fc_weights = tf.get_variable(
                name='fc_weights', dtype=tf.float32,
                shape=[fc_input_size, self.emb_size],
                initializer=tf.contrib.layers.xavier_initializer())
            fc_bias = tf.get_variable(
                name='fc_bias', dtype=tf.float32,
                shape=[self.emb_size], initializer=tf.zeros_initializer())

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

    def _get_fc_params(self, rel_emb):
        weights = self.variables['fc_weights']
        bias = self.variables['fc_bias']
        if self.context_rel is not None:
            weights = weights.generate(rel_emb)
            bias = bias.generate(rel_emb)
        return (weights, bias)

    def _create_predictions(self, e1_emb, rel_emb):
        e1_emb = tf.reshape(e1_emb, [-1, 10, 20, 1])

        if self.context_rel is None:
            rel_emb = tf.reshape(rel_emb, [-1, 10, 20, 1])

        is_train_float = tf.cast(self.is_train, tf.float32)

        if self.context_rel is None:
            stacked_emb = tf.concat([e1_emb, rel_emb], 1)
        else:
            stacked_emb = e1_emb
        
        stacked_emb = tf.layers.batch_normalization(
            stacked_emb, momentum=0.1, reuse=tf.AUTO_REUSE,
            training=False, fused=True, name='StackedEmbBN')
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
            weights, bias = self._get_fc_params(rel_emb)
            batch_size = tf.shape(conv1_dropout)[0]

            fc_input = tf.reshape(conv1_dropout, [batch_size, -1])

            if self.concat_rel:
                fc_input = tf.concat([fc_input, tf.reshape(rel_emb, [-1, 200])], axis=1)

            if self.context_rel is None:
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
                #_create_summaries('fc_with_activation', fc_relu)
        
        return fc_bn

    def _create_sequence_predictions(self, e1, rel_1, rel_2, rel_3):
        batch_size = tf.shape(rel_1)[0]
        seq1_predictions = self._create_predictions(e1, rel_1)
        seq2_predictions = self._create_predictions(seq1_predictions, rel_2)
        seq3_predictions = self._create_predictions(seq1_predictions, rel_3)
        
        seq1_predictions = tf.reshape(seq1_predictions, [batch_size, 1, self.emb_size])
        seq2_predictions = tf.reshape(seq2_predictions, [batch_size, 1, self.emb_size])
        seq3_predictions = tf.reshape(seq3_predictions, [batch_size, 1, self.emb_size])

        aggregate_predictions = tf.concat([seq1_predictions, seq2_predictions, seq3_predictions], 1)
        selected_predictions = tf.boolean_mask(aggregate_predictions, self.reg_seq_mask)
        
        return selected_predictions

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
            semant_loss = tf.reduce_sum(
                tf.losses.sigmoid_cross_entropy(
                    targets, predictions,
                    label_smoothing=self.label_smoothing_epsilon))

            if self._loss_summaries:
                tf.summary.scalar('loss', semant_loss)
        return semant_loss

    def _find_candidates(self, epsilon):
        batch_size = tf.shape(self.predictions)[0]
        min_correct_weight = tf.reduce_min(self.predictions * self.e2_multi, 1)
        min_correct_weight = tf.reshape(min_correct_weight, [batch_size, 1])
        lower_bound = self.predictions + epsilon
        candidates = tf.greater_equal(lower_bound, min_correct_weight)
        return tf.cast(candidates, tf.float32)

    def _comp_rest_with_seq(self, relation_probs, sequence_probs):
        batch_size = tf.shape(relation_probs)[0]
        ones = tf.ones([batch_size, self.num_ent], tf.float32)
        loss_weights = ones - self.reg_rel_e2_multi
        elem_loss = tf.square(relation_probs - sequence_probs) * loss_weights
        sample_loss = tf.reduce_sum(elem_loss, 1)
        sample_loss = tf.reshape(sample_loss, [batch_size])
        reg_similarity = tf.reshape(self.reg_sim, [batch_size])
        total_loss = tf.reduce_mean(sample_loss * reg_similarity)
        return total_loss

    def _match_rel_to_seq(self, relation_probs, sequence_probs):
        batch_size = tf.shape(relation_probs)[0]
        ones = tf.ones([batch_size, self.num_ent], tf.float32)
        loss_weights = ones - self.reg_rel_e2_multi
        #sequence_probs = tf.Print(sequence_probs, [sequence_probs], 'sequence_probs')
        #sequence_probs = tf.nn.sigmoid(sequence_probs)
        #sequence_probs = tf.Print(sequence_probs, [sequence_probs], 'sequence_probs')
        
        reg_elem_loss = tf.losses.sigmoid_cross_entropy(
                    sequence_probs, relation_probs,
                    weights = tf.maximum(0., self.reg_seq_e2_multi - self.reg_rel_e2_multi),
                    reduction = tf.losses.Reduction.NONE)
        reg_batch_loss = tf.reduce_sum(reg_elem_loss, axis = 1)
        #reg_batch_loss = tf.Print(reg_batch_loss, [reg_batch_loss])
        reg_batch_loss = tf.reshape(reg_batch_loss, [batch_size])
        reg_sim = tf.reshape(self.reg_sim, [batch_size])
        reg_batch_weighted_loss = reg_batch_loss * reg_sim
        #reg_batch_weighted_loss = tf.Print(reg_batch_weighted_loss, [reg_batch_weighted_loss])
        reg_batch_weighted_loss = tf.nn.relu(reg_batch_weighted_loss)
        reg_loss = tf.reduce_sum(reg_batch_weighted_loss)
        return reg_loss

    def compute_distribution_loss(self, prediction_vectors):
        entity_embs = self.variables['ent_emb']
        mean, variance = tf.nn.moments(prediction_vectors, 0)
        ent_maxes = tf.reduce_max(entity_embs, 0)
        ent_mins = tf.reduce_min(entity_embs, 0)
        ent_mean = (ent_maxes + ent_mins) / 2.
        ent_variance = tf.square(ent_maxes - ent_mins) / 12.
        mean_loss = tf.reduce_sum(tf.square(mean - ent_mean))
        variance_loss = tf.reduce_sum(tf.square(variance - ent_variance))
        distribution_loss = mean_loss + variance_loss
        return distribution_loss 

    def cosine_match(self, rel_pred, seq_pred):
        pass

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
