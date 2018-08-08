from __future__ import absolute_import, division, print_function

import logging
import os

import tensorflow as tf

from ..data.loaders import KinshipLoader
from ..evaluation.metrics import ranking_and_hits
from ..models.conve_struc_merged import ConvE

LOGGER = logging.getLogger(__name__)

DEVICE = '/GPU:0'
MODEL_NAME = 'conve_baseline_opt_params'
MAX_STEPS = 10000000
LOG_STEPS = 100
SUMMARY_STEPS = 1000
CKPT_STEPS = 5000
EVAL_STEPS = 5000

EMB_SIZE = 200
INPUT_DROPOUT = 0.2
FEATURE_MAP_DROPOUT = 0.3
OUTPUT_DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
LABEL_SMOOTHING_EPSILON = 0.1

WORKING_DIR = os.path.join(os.getcwd(), 'temp')
DATA_DIR = os.path.join(WORKING_DIR, 'data')
LOG_DIR = os.path.join(WORKING_DIR, 'models', MODEL_NAME, 'logs')
CKPT_PATH = os.path.join(WORKING_DIR, 'models', MODEL_NAME, 'model_weights.ckpt')
EVAL_PATH = os.path.join(WORKING_DIR, 'evaluation', MODEL_NAME)

ADD_LOSS_SUMMARIES = True
ADD_VARIABLE_SUMMARIES = False
ADD_TENSOR_SUMMARIES = False
STRUC2VEC_ARGS = {'walk_length': 80,
                  'num_walks': 10,
                  'until-layer': None,
                  'iter': 5,
                  'workers': 6,
                  'weighted': False,
                  'directed': True,
                  'undirected': False,
                  'OPT1': False,
                  'OPT2': False,
                  'OPT3': False}


def main():
    loader = KinshipLoader()
    loader.create_tf_record_files(DATA_DIR, STRUC2VEC_ARGS)

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
    train_dataset, struc_dataset = loader.train_dataset(
        DATA_DIR, BATCH_SIZE, STRUC2VEC_ARGS, LABEL_SMOOTHING_EPSILON)

    dev_datasets = loader.dev_datasets(DATA_DIR, BATCH_SIZE)
    test_datasets = loader.test_datasets(DATA_DIR, BATCH_SIZE)

    train_iterator = train_dataset.make_one_shot_iterator()
    struc_iterator = struc_dataset.make_one_shot_iterator()
    dev_iterators = [d.make_initializable_iterator() for d in dev_datasets]
    test_iterators = [d.make_initializable_iterator() for d in test_datasets]

    dev_iterators_init = tf.group([d.initializer for d in dev_iterators])
    test_iterators_init = tf.group([d.initializer for d in test_iterators])

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

    # Obtain the dataset iterator handles.
    train_iterator_handle = session.run(train_iterator.string_handle())
    struc_iterator_handle = session.run(struc_iterator.string_handle())
    dev_iterator_handles = session.run([d.string_handle() for d in dev_iterators])
    test_iterator_handles = session.run([d.string_handle() for d in test_iterators])

    # Initalize the loss term weights.
    semant_loss_weight = 1.0
    struct_loss_weight = 0.0

    for step in range(MAX_STEPS):
        feed_dict = {
            model.input_iterator_handle: train_iterator_handle,
            model.struc_iterator_handle: struc_iterator_handle,
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

        # Evaluate, if necessary.
        if step % EVAL_STEPS == 0 and step > 0:
            LOGGER.info('Running dev evaluation.')
            session.run(dev_iterators_init)
            ranking_and_hits(
                model, EVAL_PATH, dev_iterator_handles,
                'dev_evaluation', session)
            LOGGER.info('Running test evaluation.')
            session.run(test_iterators_init)
            ranking_and_hits(
                model, EVAL_PATH, test_iterator_handles,
                'test_evaluation', session)

        if step % CKPT_STEPS == 0 and step > 0:
            LOGGER.info('Saving checkpoint at %s.', CKPT_PATH)
            saver.save(session, CKPT_PATH)


if __name__ == '__main__':
    main()
