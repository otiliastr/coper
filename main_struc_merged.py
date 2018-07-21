import copy
import os
import sys

import numpy as np
import tensorflow as tf

from spodernet.hooks import LossHook, ETAHook
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.logger import Logger, LogLevel

from conve_struc_merged import ConvE
from evaluation_tf import ranking_and_hits
from utilities import (
    preprocess_dataset, load_adjacency_matrix, prune_adjacency_matrix)


DEVICE = '/GPU:0'
MODEL_NAME = 'conve_equal_merge_opt_bl_params_test_2'
MAX_EPOCHS = 1000
SUMMARY_EPOCHS = 50
CHECKPOINT_EPOCHS = 50
EVAL_EPOCHS = 50

ADD_LOSS_SUMMARIES = True
ADD_VARIABLE_SUMMARIES = False
ADD_TENSOR_SUMMARIES = False

# /zfsauton/home/gis/research/qa/models/prelim_tests/ConvE/tmp/merged_v0/
WORKING_DIR = '/usr0/home/ostretcu/code/george/models/prelim_tests/ConvE/tmp/merged_v0'
LOG_DIR = os.path.join(WORKING_DIR, 'logs')
MODEL_PATH = os.path.join(WORKING_DIR, 'model_weights.ckpt')

INPUT_KEYS = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']


def main():
    # Set some general configuration settings.
    np.set_printoptions(precision=3)
    Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG

    # Obtain the structure walks path from the list of arguments.
    structure_walks_path_idx = sys.argv.index('structure_walks') + 1
    structure_walks_path = sys.argv[structure_walks_path_idx]

    # Remove command-line arguments that are not in the Config list of allowed
    # arguments so that it does not crash. This includes the tag and the value
    # arguments.
    args = copy.deepcopy(sys.argv)
    args.remove(sys.argv[structure_walks_path_idx-1])
    args.remove(sys.argv[structure_walks_path_idx])

    # Parse console parameters and set global variables
    Config.backend = Backends.CNTK
    Config.parse_argv(args)
    # Config.label_smoothing_epsilon = 0.05
    # Config.learning_rate = 0.003
    # Config.L2 = 0.995
    Config.cuda = False
    Config.emb_size = 200

    # Preprocess the dataset, if needed, and then load it.
    if Config.process:
        preprocess_dataset(Config.dataset, INPUT_KEYS, delete_data=True)
    pipeline = Pipeline(Config.dataset, keys=INPUT_KEYS)
    pipeline.load_vocabs()
    vocab = pipeline.state['vocab']

    # Create the train, dev, and test batch iterators.
    train_batcher = StreamBatcher(
        Config.dataset, 'train', Config.batch_size,
        randomize=True, loader_threads=32, keys=INPUT_KEYS,
        cache_size_GB=64)
    dev_batcher = StreamBatcher(
        Config.dataset, 'dev', Config.batch_size,
        randomize=False, loader_threads=4, keys=INPUT_KEYS)
    test_batcher = StreamBatcher(
        Config.dataset, 'test', Config.batch_size,
        randomize=False, loader_threads=4, keys=INPUT_KEYS)

    train_batcher.at_batch_prepared_observers.insert(1, TargetIdx2MultiTarget(
        vocab['e1'].num_token, 'e2_multi1', 'e2_multi1_binary'))
    eta = ETAHook('train', print_every_x_batches=100)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=100))

    # Create the model.
    with tf.device(DEVICE):
        # We are using resource variables because due to
        # some implementation details, this allows us to
        # better utilize GPUs while training.
        with tf.variable_scope('variables', use_resource=True):
            model = ConvE(model_descriptors={
                'num_ent': vocab['e1'].num_token,
                'num_rel': vocab['rel'].num_token,
                'emb_size': Config.emb_size,
                'batch_size': Config.batch_size,
                'input_dropout': Config.input_dropout,
                'hidden_dropout': Config.feature_map_dropout,
                'output_dropout': Config.dropout,
                'learning_rate': Config.learning_rate,
                'add_loss_summaries': ADD_LOSS_SUMMARIES,
                'add_variable_summaries': ADD_VARIABLE_SUMMARIES,
                'add_tensor_summaries': ADD_TENSOR_SUMMARIES})

    # Load the adjacency matrix of nodes with similar structure.
    adj_matrix = load_adjacency_matrix(
        structure_walks_path, num_ent=vocab['e1'].num_token)
    adj_matrix = prune_adjacency_matrix(adj_matrix)

    # Log some information.
    print('Number of entities: %d' % vocab['e1'].num_token)
    print('Number of relations: %d' % vocab['rel'].num_token)
    model.print_parameters()

    # Create a TensorFlow session and start training.
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOG_DIR, session.graph)

    # Initialize the values of all variables.
    session.run(tf.global_variables_initializer())

    # Initalize the loss term weights.
    semant_loss_weight = 1.0
    struct_loss_weight = 1.0

    iteration = 0
    for epoch in range(MAX_EPOCHS):
        # struct_loss_weight = max(0.5, struct_loss_weight - .01)

        # Normalize for pmf
        # total_weight = semant_loss_weight + struct_loss_weight
        # semant_loss_weight /= total_weight
        # struct_loss_weight /= total_weight
        
        new_epoch = True

        for str2var in train_batcher:
            # Apply label smoothing.
            e2_multi_val = str2var['e2_multi1_binary']
            e2_multi_val = (
                ((1.0 - Config.label_smoothing_epsilon) * e2_multi_val) +
                (1.0 / e2_multi_val.size(1)))

            batch_e1 = np.reshape(str2var['e1'], (str2var['e1'].shape[0]))

            feed_dict = {
                model.e1: str2var['e1'],
                model.rel: str2var['rel'],
                model.e2_multi: e2_multi_val,
                model.e2_struct: adj_matrix[batch_e1],
                model.input_dropout: Config.input_dropout,
                model.hidden_dropout: Config.feature_map_dropout,
                model.output_dropout: Config.dropout,
                model.semant_loss_weight: semant_loss_weight,
                model.struct_loss_weight: struct_loss_weight}

            if model.summaries is not None and \
               new_epoch and \
               SUMMARY_EPOCHS is not None and \
               epoch % SUMMARY_EPOCHS == 0:
                summaries, model_loss, _ = session.run(
                    (model.summaries, model.loss, model.train_op), feed_dict)
            else:
                summaries = None
                model_loss, _ = session.run(
                    (model.loss, model.train_op), feed_dict)

            train_batcher.state.loss = model_loss
            if summaries is not None:
                summary_writer.add_summary(summaries, iteration)

            iteration += 1

        new_epoch = False

        # Evaluate, if necessary.
        if epoch % EVAL_EPOCHS == 0:
            print('Running dev evaluation.')
            ranking_and_hits(
                model, MODEL_NAME, dev_batcher,
                vocab, 'dev_evaluation', session)
            if iteration != 0:
                print('Running test evaluation.')
                ranking_and_hits(
                    model, MODEL_NAME, test_batcher,
                    vocab, 'test_evaluation', session)

        if epoch % CHECKPOINT_EPOCHS == 0:
            print('Saving trained model at %s.' % MODEL_PATH)
            saver.save(session, MODEL_PATH)


if __name__ == '__main__':
    main()
