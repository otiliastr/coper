from __future__ import absolute_import, division, print_function

import logging
import os

import tensorflow as tf
import numpy as np

from ..data.dataloaders import *
from ..evaluation.metrics import ranking_and_hits
#from ..models.conve_struc_merged import ConvE
from ..models.conve_initial_baseline import ConvE
#from ..models.conve_autoencoder import ConvE
from ..models.DistMultRelReg import DistMultReg

LOGGER = logging.getLogger(__name__)

DATA_LOADER = KinshipLoader()

DEVICE = '/GPU:0'
MODEL_NAME = 'distmult_Kinship'
MAX_STEPS = 10000000
LOG_STEPS = 100
SUMMARY_STEPS = None
CKPT_STEPS = 1000
EVAL_STEPS = 1000

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
REL_EMBEDDING_PATH = os.path.join(EVAL_PATH, 'rel_emb.txt')

ADD_LOSS_SUMMARIES = True
ADD_VARIABLE_SUMMARIES = False
ADD_TENSOR_SUMMARIES = False
RELREG_ARGS = {'seq_threshold': 0.0,
               'seq_lengths': [1, 2, 3]}


def main():
    DATA_LOADER.create_tf_record_files(DATA_DIR, RELREG_ARGS)

    # Create the model.
    with tf.device(DEVICE):
        # We are using resource variables because due to
        # some implementation details, this allows us to
        # better utilize GPUs while training.
        with tf.variable_scope('variables', use_resource=True):
            # model = ConvE(model_descriptors={
            #     'label_smoothing_epsilon': LABEL_SMOOTHING_EPSILON,
            #     'num_ent': DATA_LOADER.num_ent,
            #     'num_rel': DATA_LOADER.num_rel,
            #     'emb_size': EMB_SIZE,
            #     'input_dropout': INPUT_DROPOUT,
            #     'hidden_dropout': FEATURE_MAP_DROPOUT,
            #     'output_dropout': OUTPUT_DROPOUT,
            #     'learning_rate': LEARNING_RATE,
            #     'batch_size': BATCH_SIZE,
            #     'add_loss_summaries': ADD_LOSS_SUMMARIES,
            #     'add_variable_summaries': ADD_VARIABLE_SUMMARIES,
            #     'add_tensor_summaries': ADD_TENSOR_SUMMARIES,
            #     'embedding_dim': 200})

            model = DistMultReg(Config = {
                'num_ent': DATA_LOADER.num_ent,
                'num_rel': DATA_LOADER.num_rel,
                'emb_dim': EMB_SIZE,
                'batch_size': BATCH_SIZE,
                'max_seq_len': 3,
                'lr': 0.001
            })

    # Create dataset iterator initializers.
    train_dataset, relreg_dataset = DATA_LOADER.train_dataset(
        DATA_DIR, BATCH_SIZE, RELREG_ARGS, include_inv_relations=True)

    dev_dataset = DATA_LOADER.dev_dataset(
        DATA_DIR, BATCH_SIZE, include_inv_relations=False)
    test_dataset = DATA_LOADER.test_dataset(
        DATA_DIR, BATCH_SIZE, include_inv_relations=False)

    train_iterator = train_dataset.make_one_shot_iterator()
    relreg_iterator = relreg_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    # Log some information.
    LOGGER.info('Number of entities: %d', DATA_LOADER.num_ent)
    LOGGER.info('Number of relations: %d', DATA_LOADER.num_rel)
    # TODO: Uncomment line when runnig struc_conve!!
    model.log_parameters_info()

    # Create a TensorFlow session and start training.
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOG_DIR, session.graph)

    # Initialize the values of all variables and the train dataset iterator.
    session.run(tf.global_variables_initializer())

    # Obtain the dataset iterator handles.
    train_iterator_handle = session.run(train_iterator.string_handle())
    relreg_iterator_handle = session.run(relreg_iterator.string_handle())
    dev_iterator_handle = session.run(dev_iterator.string_handle())
    test_iterator_handle = session.run(test_iterator.string_handle())

    # Initalize the loss term weights.
    baseline_weight = 1.0
    reg_weight = 0.0
    
    for step in range(MAX_STEPS):
        feed_dict = {
            model.is_train: True,
            model.distmult_iterator_handle: train_iterator_handle,
            model.relreg_iterator_handle: relreg_iterator_handle,
            model.baseline_weight: baseline_weight,
            model.reg_weight: reg_weight}


        if model.summaries is not None and \
            SUMMARY_STEPS is not None and \
            step % SUMMARY_STEPS == 0:
            summaries, loss, _ = session.run(
                (model.summaries, model.collective_loss, model.train_op), feed_dict)
        else:
            summaries = None
            loss, _ = session.run((model.collective_loss, model.train_op), feed_dict)
        #if step > 0 and step % 1000== 0:
         #   rel_emb = session.run(model.variables['rel_emb'])
            #print("THE SHAPE OF REL_EMB IS {}".format(rel_emb.shape))
            #BUG
            #with open(REL_EMBEDDING_PATH, 'w+') as handle:
             #   for line in rel_emb:
              #     write_line = " ".join(line)
               #    handle.write(write_line + "\n")

          #  np.savetxt(REL_EMBEDDING_PATH, rel_emb, delimiter = " ")
            
        # Log the loss, if necessary.
        if step % LOG_STEPS == 0:
            LOGGER.info('Step %6d | Loss: %10.4f', step, loss)

        # Write summaries, if necessary.
        if summaries is not None:
            summary_writer.add_summary(summaries, step)

        # Evaluate, if necessary.
        if step % EVAL_STEPS == 0 and step > 0:
            LOGGER.info('Running dev evaluation.')
            session.run(dev_iterator.initializer)
            ranking_and_hits(
                model, EVAL_PATH, dev_iterator_handle,
                'dev_evaluation', session)
            LOGGER.info('Running test evaluation.')
            session.run(test_iterator.initializer)
            ranking_and_hits(
                model, EVAL_PATH, test_iterator_handle,
                'test_evaluation', session)

        if step % CKPT_STEPS == 0 and step > 0:
            LOGGER.info('Saving checkpoint at %s.', CKPT_PATH)
            saver.save(session, CKPT_PATH)


if __name__ == '__main__':
    main()
