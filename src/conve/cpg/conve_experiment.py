from __future__ import absolute_import, division, print_function

import logging
import os

import tensorflow as tf

from .conve import ConvE
from .data import *
from ..evaluation.metrics import ranking_and_hits

LOGGER = logging.getLogger(__name__)

DATA_LOADER = NELL995Loader()

ENT_EMB_SIZE = 200
REL_EMB_SIZE = 50
CONCAT_REL = False      # Set to `False` for plain ConvE
CONTEXT_REL_CONV = None # Set to `None` for plain ConvE
CONTEXT_REL_OUT = []    # Set to `None` for plain ConvE

CONTEXT_REL_DROPOUT = 0.2
CONTEXT_REL_USE_BATCH_NORM = True

INPUT_DROPOUT = 0.2
FEATURE_MAP_DROPOUT = 0.3
OUTPUT_DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
LABEL_SMOOTHING_EPSILON = 0.1

DEVICE = '/GPU:0'
MAX_STEPS = 10000000
LOG_STEPS = 100
SUMMARY_STEPS = None
CKPT_STEPS = 1000
EVAL_STEPS = 100
LOG_LOSS = 100

EVAL_ON_TRAIN = False
EVAL_ON_DEV = False
EVAL_ON_TEST = True

NUM_PRETRAIN_STEPS = 0

ADD_LOSS_SUMMARIES = True
ADD_VARIABLE_SUMMARIES = False
ADD_TENSOR_SUMMARIES = False

MODEL_NAME = 'ConvE_negative_sampling_{}_ent_emb_{}_rel_emb_{}_batch_size_{}'.format(DATA_LOADER.dataset_name, ENT_EMB_SIZE, REL_EMB_SIZE, BATCH_SIZE)

WORKING_DIR = os.path.join(os.getcwd(), 'temp')
DATA_DIR = os.path.join(WORKING_DIR, 'data')
LOG_DIR = os.path.join(WORKING_DIR, 'models', MODEL_NAME, 'logs')
CKPT_PATH = os.path.join(WORKING_DIR, 'models', MODEL_NAME, 'model_weights.ckpt')
EVAL_PATH = os.path.join(WORKING_DIR, 'evaluation', MODEL_NAME)

os.makedirs(EVAL_PATH, exist_ok=True)

if __name__ == '__main__':
    DATA_LOADER.create_tf_record_files(DATA_DIR)

    # Create the model.
    with tf.device(DEVICE):
        # We are using resource variables because due to
        # some implementation details, this allows us to
        # better utilize GPUs while training.
        with tf.variable_scope('variables', use_resource=True):
            model = ConvE(model_descriptors={
                'label_smoothing_epsilon': LABEL_SMOOTHING_EPSILON,
                'num_ent': DATA_LOADER.num_ent,
                'num_rel': DATA_LOADER.num_rel,
                'ent_emb_size': ENT_EMB_SIZE,
                'rel_emb_size': REL_EMB_SIZE,
                'concat_rel': CONCAT_REL,
                'context_rel_conv': CONTEXT_REL_CONV,
                'context_rel_out': CONTEXT_REL_OUT,
                'context_rel_dropout': CONTEXT_REL_DROPOUT,
                'context_rel_use_batch_norm': CONTEXT_REL_USE_BATCH_NORM,
                'input_dropout': INPUT_DROPOUT,
                'hidden_dropout': FEATURE_MAP_DROPOUT,
                'output_dropout': OUTPUT_DROPOUT,
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'add_loss_summaries': ADD_LOSS_SUMMARIES,
                'add_variable_summaries': ADD_VARIABLE_SUMMARIES,
                'add_tensor_summaries': ADD_TENSOR_SUMMARIES})

    # Create dataset iterator initializers.
    train_dataset = DATA_LOADER.train_dataset(
        directory=DATA_DIR,
        batch_size=BATCH_SIZE,
        include_inv_relations=True,
        buffer_size=1024,
        prefetch_buffer_size=16)
    train_eval_dataset = DATA_LOADER.eval_dataset(
        directory=DATA_DIR,
        dataset_type='train',
        batch_size=BATCH_SIZE,
        include_inv_relations=False,
        buffer_size=1024,
        prefetch_buffer_size=16)
    dev_eval_dataset = DATA_LOADER.eval_dataset(
        directory=DATA_DIR,
        dataset_type='dev',
        batch_size=BATCH_SIZE,
        include_inv_relations=False,
        buffer_size=1024,
        prefetch_buffer_size=16)
    test_eval_dataset = DATA_LOADER.eval_dataset(
        directory=DATA_DIR,
        dataset_type='test',
        batch_size=BATCH_SIZE,
        include_inv_relations=False,
        buffer_size=1024,
        prefetch_buffer_size=16)

    train_iterator = train_dataset.make_one_shot_iterator()
    train_eval_iterator = train_eval_dataset.make_initializable_iterator()
    dev_eval_iterator = dev_eval_dataset.make_initializable_iterator()
    test_eval_iterator = test_eval_dataset.make_initializable_iterator()

    # Log some information.
    LOGGER.info('Number of entities: %d', DATA_LOADER.num_ent)
    LOGGER.info('Number of relations: %d', DATA_LOADER.num_rel)
    # model.log_parameters_info()

    # Create a TensorFlow session and start training.
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOG_DIR, session.graph)

    # Initialize the values of all variables and the train dataset iterator.
    session.run(tf.global_variables_initializer())

    # Obtain the dataset iterator handles.
    train_iterator_handle = session.run(train_iterator.string_handle())
    train_eval_iterator_handle = session.run(train_eval_iterator.string_handle())
    dev_eval_iterator_handle = session.run(dev_eval_iterator.string_handle())
    test_eval_iterator_handle = session.run(test_eval_iterator.string_handle())

    for step in range(MAX_STEPS):
        feed_dict = {
            model.is_train: True,
            model.input_iterator_handle: train_iterator_handle}

        if model.summaries is not None and \
                SUMMARY_STEPS is not None and \
                step % SUMMARY_STEPS == 0:
            summaries, loss, obj_loss, reg_loss, _ = session.run(
                (model.summaries, model.loss, model.obj_weighted_loss,
                 model.reg_weighted_loss, model.train_op), feed_dict)
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
        if step % EVAL_STEPS == 0 and step > NUM_PRETRAIN_STEPS:
            # Perform evaluation.
            if EVAL_ON_TRAIN:
                LOGGER.info('Running train evaluation.')
                session.run(train_eval_iterator.initializer)
                ranking_and_hits(
                    model, EVAL_PATH, train_eval_iterator_handle,
                    'train_evaluation', session)
            if EVAL_ON_DEV:
                LOGGER.info('Running dev evaluation.')
                session.run(dev_eval_iterator.initializer)
                ranking_and_hits(
                    model, EVAL_PATH, dev_eval_iterator_handle,
                    'dev_evaluation', session)
            if EVAL_ON_TEST:
                LOGGER.info('Running test evaluation.')
                session.run(test_eval_iterator.initializer)
                ranking_and_hits(
                    model, EVAL_PATH, test_eval_iterator_handle,
                    'test_evaluation', session)

        if step % CKPT_STEPS == 0 and step > NUM_PRETRAIN_STEPS:
            LOGGER.info('Saving checkpoint at %s.', CKPT_PATH)
            saver.save(session, CKPT_PATH)
