from __future__ import absolute_import, division, print_function

import logging
import os

import numpy as np
import tensorflow as tf

from ..data.loaders import KinshipLoader
from ..evaluation.metrics import ranking_and_hits
from ..models.conve_struc_merged import ConvE
from ..utilities.structure import load_adjacency_matrix, prune_adjacency_matrix

LOGGER = logging.getLogger(__name__)

DEVICE = '/CPU:0'
MODEL_NAME = 'conve_equal_merge_opt_bl_params_test_2'
MAX_STEPS = 10000
LOG_STEPS = 100
SUMMARY_STEPS = 100
CKPT_STEPS = 100
EVAL_STEPS = 100

EMB_SIZE = 200
INPUT_DROPOUT = 0.0
FEATURE_MAP_DROPOUT = 0.0
OUTPUT_DROPOUT = 0.0
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
LABEL_SMOOTHING_EPSILON = 0.1

WORKING_DIR = os.path.join(os.getcwd(), 'temp')
DATA_DIR = os.path.join(WORKING_DIR, 'data')
LOG_DIR = os.path.join(WORKING_DIR, 'models', MODEL_NAME, 'logs')
CKPT_PATH = os.path.join(WORKING_DIR, 'models', MODEL_NAME, 'model_weights.ckpt')

# TODO: Standardize this.
STRUCTURE_WALKS_PATH = '/Users/anthony/Development/GitHub/qa_types/src/temp/data/kinship/random_walks.txt'

ADD_LOSS_SUMMARIES = True
ADD_VARIABLE_SUMMARIES = False
ADD_TENSOR_SUMMARIES = False


def main():
    loader = KinshipLoader()
    loader.create_tf_record_files(DATA_DIR)

    # Load the adjacency matrix of nodes with similar structure.
    adj_matrix = load_adjacency_matrix(STRUCTURE_WALKS_PATH, loader.num_ent)
    adj_matrix = prune_adjacency_matrix(adj_matrix)

    # Create the model.
    with tf.device(DEVICE):
        # We are using resource variables because due to
        # some implementation details, this allows us to
        # better utilize GPUs while training.
        with tf.variable_scope('variables', use_resource=True):
            model = ConvE(model_descriptors={
                'num_ent': loader.num_ent,
                'num_rel': loader.num_rel,
                'emb_size': EMB_SIZE,
                'input_dropout': INPUT_DROPOUT,
                'hidden_dropout': FEATURE_MAP_DROPOUT,
                'output_dropout': OUTPUT_DROPOUT,
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'add_loss_summaries': ADD_LOSS_SUMMARIES,
                'add_variable_summaries': ADD_VARIABLE_SUMMARIES,
                'add_tensor_summaries': ADD_TENSOR_SUMMARIES})

            # Create dataset iterator initializers.
            train_dataset = loader.train_dataset(
                DATA_DIR, BATCH_SIZE, adj_matrix, LABEL_SMOOTHING_EPSILON)
            train_init_op = model.input_iterator.make_initializer(train_dataset)

    # Log some information.
    LOGGER.info('Number of entities: %d', loader.num_ent)
    LOGGER.info('Number of relations: %d', loader.num_rel)
    model.log_parameters_info()

    # Create a TensorFlow session and start training.
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOG_DIR, session.graph)

    # Initialize the values of all variables and the train dataset iterator.
    session.run(tf.global_variables_initializer())
    session.run(train_init_op)

    # Initalize the loss term weights.
    semant_loss_weight = 1.0
    struct_loss_weight = 1.0

    for step in range(MAX_STEPS):
        feed_dict = {
            model.input_dropout: INPUT_DROPOUT,
            model.hidden_dropout: FEATURE_MAP_DROPOUT,
            model.output_dropout: OUTPUT_DROPOUT,
            model.semant_loss_weight: semant_loss_weight,
            model.struct_loss_weight: struct_loss_weight}

        if model.summaries is not None and \
            SUMMARY_STEPS is not None and \
            step % SUMMARY_STEPS == 0:
            summaries, loss, _ = session.run(
                (model.summaries, model.loss, model.train_op), feed_dict)
        else:
            summaries = None
            loss, _ = session.run((model.loss, model.train_op), feed_dict)

        # Log the loss, if necessary.
        if step % LOG_STEPS == 0:
            LOGGER.info('Step %6d | Loss: %10.4f', step, loss)

        # Write summaries, if necessary.
        if summaries is not None:
            summary_writer.add_summary(summaries, step)

        # # Evaluate, if necessary.
        # if step % EVAL_STEPS == 0:
        #     LOGGER.info('Running dev evaluation.')
        #     ranking_and_hits(
        #         model, MODEL_NAME, dev_batcher,
        #         vocab, 'dev_evaluation', session)
        #     LOGGER.info('Running test evaluation.')
        #     ranking_and_hits(
        #         model, MODEL_NAME, test_batcher,
        #         vocab, 'test_evaluation', session)

        if step % CKPT_STEPS == 0:
            LOGGER.info('Saving checkpoint at %s.' % CKPT_PATH)
            saver.save(session, CKPT_PATH)


if __name__ == '__main__':
    main()
