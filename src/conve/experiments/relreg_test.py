from __future__ import absolute_import, division, print_function

import logging
import os
import math
import pickle

import tensorflow as tf
import numpy as np

from ..data.weighted_guess_loaders import *
from ..evaluation.metrics import ranking_and_hits
from ..models.conve_struc_merged import ConvE
#from ..models.conve_initial_baseline import ConvE
#from ..models.conve_autoencoder import ConvE
#from ..models.DistMultRelReg import DistMultReg
from ..models.bilinear_reg import BiLinearReg
#from ..models.bilinear_perturbation import BiLinearPerturbation

LOGGER = logging.getLogger(__name__)

DATA_LOADER = FB15k237Loader()#FB15k237Loader()

beta = 1.001
DATASET = 'FB15k237'
EXP_TYPE = 'hybrid_1'
REG_TYPE = 'method_7'

SIM_THRESHOLD = 'None'
NUM_PRETRAIN_STEPS = 0
OBJ_WEIGHT = 1.0
SEQ_WEIGHT = 0.0
REG_WEIGHT = 0.0
EPSILON = 0.1
USE_BALL = False

DEVICE = '/CPU:0'
MAX_STEPS = 10000000
LOG_STEPS = 100
SUMMARY_STEPS = None
CKPT_STEPS = 1000
EVAL_STEPS = 1000
LOG_LOSS = 100

EMB_SIZE = 200
INPUT_DROPOUT = 0.2
FEATURE_MAP_DROPOUT = 0.3
OUTPUT_DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
LABEL_SMOOTHING_EPSILON = 0.1

ADD_LOSS_SUMMARIES = True
ADD_VARIABLE_SUMMARIES = False
ADD_TENSOR_SUMMARIES = False
RELREG_ARGS = {'seq_threshold': 0.0,
               'seq_lengths': [2],
               'sim_threshold': 0.0}


if EXP_TYPE != 'bl':
    print('HI')
    #MODEL_NAME = 'bi_non_linear_{}_{}_with_{}_and_batch_size_{}_emb_size_{}'.format(EXP_TYPE, REG_TYPE, DATASET, BATCH_SIZE, EMB_SIZE)
    #MODEL_NAME = 'bi_non_linear_{}_on_{}_with_{}_and_batch_size_{}_emb_size_{}'.format(EXP_TYPE, DATASET, REG_TYPE, BATCH_SIZE, EMB_SIZE)
    #MODEL_NAME = 'ConvE_{}_{}_{}_reg-weight_{}_seq-weight_{}_sim-threshold_{}_emb_{}_batch_size_{}'.format(DATASET, EXP_TYPE, REG_TYPE, str(REG_WEIGHT), str(SEQ_WEIGHT), RELREG_ARGS['sim_threshold'], EMB_SIZE, BATCH_SIZE)
    #MODEL_NAME = 'bilinear_{}_{}_{}_{}_None_emb_{}_batch_size_{}'.format(DATASET, EXP_TYPE, REG_WEIGHT, REG_TYPE, EMB_SIZE, BATCH_SIZE)
    MODEL_NAME = 'ConvE_{}_{}_sim_threshold_{}_emb_size_{}_batch_size_{}'.format(DATASET, EXP_TYPE, RELREG_ARGS['sim_threshold'], EMB_SIZE, BATCH_SIZE)
else:
    MODEL_NAME = 'ConvE_negative_sampling_{}_emb_{}_batch_size_{}'.format(DATASET, EMB_SIZE, BATCH_SIZE)

WORKING_DIR = os.path.join(os.getcwd(), 'temp')
DATA_DIR = os.path.join(WORKING_DIR, 'data')
LOG_DIR = os.path.join(WORKING_DIR, 'models', MODEL_NAME, 'logs')
CKPT_PATH = os.path.join(WORKING_DIR, 'models', MODEL_NAME, 'model_weights.ckpt')
EVAL_PATH = os.path.join(WORKING_DIR, 'evaluation', MODEL_NAME)
REL_EMBEDDING_PATH = os.path.join(EVAL_PATH, 'rel_emb.txt')
OBJ_LOSS_PATH = os.path.join(EVAL_PATH, 'obj_loss.txt')
REG_LOSS_PATH = os.path.join(EVAL_PATH, 'reg_loss.txt')
COLL_LOSS_PATH = os.path.join(EVAL_PATH, 'coll_loss.txt')
SEQ_LOSS_PATH = os.path.join(EVAL_PATH, 'seq_loss.txt')
ENTITY_EMB_PATH = os.path.join(WORKING_DIR, 'entity_embs.pkl')
PRED_WEIGHTS_PATH = os.path.join(WORKING_DIR, 'pred_weights.txt')
OUTPUT_BIAS_PATH = os.path.join(WORKING_DIR, 'output_bias.pkl')


os.makedirs(EVAL_PATH, exist_ok=True)

def _write_data_to_file(file_path, data):
    if os.path.exists(file_path):
        append_write = 'a'
    else:
        append_write = 'w+'
    with open(file_path, append_write) as handle:
        handle.write(str(data) + "\n")

def save_obj(obj, fpath):
    #directory = os.getcwd()
    #fpath = os.path.join(directory, 'obj', name)
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def main():
    DATA_LOADER.create_tf_record_files(DATA_DIR, RELREG_ARGS)

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
                 'emb_size': EMB_SIZE,
                 'input_dropout': INPUT_DROPOUT,
                 'hidden_dropout': FEATURE_MAP_DROPOUT,
                 'output_dropout': OUTPUT_DROPOUT,
                 'learning_rate': LEARNING_RATE,
                 'batch_size': BATCH_SIZE,
                 'add_loss_summaries': ADD_LOSS_SUMMARIES,
                 'add_variable_summaries': ADD_VARIABLE_SUMMARIES,
                 'add_tensor_summaries': ADD_TENSOR_SUMMARIES,
                 'embedding_dim': 200})

            #model = BiLinearReg(Config = {
             #   'num_ent': DATA_LOADER.num_ent,
              #  'num_rel': DATA_LOADER.num_rel,
               # 'emb_dim': EMB_SIZE,
                #'batch_size': BATCH_SIZE,
                #'max_seq_len': 3,
                #'lr': 0.001
            #})

    # Create dataset iterator initializers.
    train_dataset = DATA_LOADER.train_dataset(
        DATA_DIR, BATCH_SIZE, RELREG_ARGS, include_inv_relations=True, buffer_size = 1024, prefetch_buffer_size = 16)

    dev_dataset = DATA_LOADER.dev_dataset(
        DATA_DIR, BATCH_SIZE, include_inv_relations=False, buffer_size = 1024, prefetch_buffer_size = 16)
    test_dataset = DATA_LOADER.test_dataset(
        DATA_DIR, BATCH_SIZE, include_inv_relations=False, buffer_size = 1024, prefetch_buffer_size = 16)

    train_iterator = train_dataset.make_one_shot_iterator()
    #relreg_iterator = relreg_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    # Log some information.
    LOGGER.info('Number of entities: %d', DATA_LOADER.num_ent)
    LOGGER.info('Number of relations: %d', DATA_LOADER.num_rel)
    # TODO: Uncomment line when runnig struc_conve!!
    
    # model.log_parameters_info()

    # Create a TensorFlow session and start training.
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOG_DIR, session.graph)

    # Initialize the values of all variables and the train dataset iterator.
    session.run(tf.global_variables_initializer())

    # Obtain the dataset iterator handles.
    train_iterator_handle = session.run(train_iterator.string_handle())
    #relreg_iterator_handle = session.run(relreg_iterator.string_handle())
    dev_iterator_handle = session.run(dev_iterator.string_handle())
    test_iterator_handle = session.run(test_iterator.string_handle())

    # Initalize the loss term weights.
    reg_weight = REG_WEIGHT #if SIM_THRESHOLD != 'None' else 0.0#* .98 ** (math.floor(step / 1000))
    #sim_threshold = float(SIM_THRESHOLD) if SIM_THRESHOLD != 'None' else 0.0
    
    prev_valid_loss = np.inf
    session.run(dev_iterator.initializer)
    start_reg_step = None
    is_reg = False
    obj_weight = OBJ_WEIGHT
    seq_weight = SEQ_WEIGHT
    epsilon = EPSILON
    use_ball = USE_BALL
    for step in range(MAX_STEPS):
        feed_dict = {
            model.is_train: True,
            model.input_iterator_handle: train_iterator_handle,
         #   model.relreg_iterator_handle: relreg_iterator_handle,
            model.seq_weight: seq_weight,
            model.obj_weight: obj_weight,
            model.epsilon: epsilon,
            model.use_ball: use_ball,
            #model.dist_weight: 0.0
            model.reg_weight: reg_weight, #* (float(1.0001 ** (step - 1000)) if step >= 1000 else 0.0),
            #model.sim_threshold: sim_threshold
           }


        if model.summaries is not None and \
            SUMMARY_STEPS is not None and \
            step % SUMMARY_STEPS == 0:
            summaries, loss, obj_loss, reg_loss, _ = session.run(
                (model.summaries, model.collective_loss, model.obj_weighted_loss, 
                 model.reg_weighted_loss, model.train_op), feed_dict)
        else:
            summaries = None
            loss, output_bias, _ = session.run((model.collective_loss,
                                                                        model.variables['output_bias'],
                           #                                             model.reg_weighted_loss,
                            #                                            model.seq_weighted_loss, 
                                                                        model.train_op),
                                                                        feed_dict)
        
        #if step > 0 and step % 1000== 0:
         #   rel_emb = session.run(model.variables['rel_emb'])
            #print("THE SHAPE OF REL_EMB IS {}".format(rel_emb.shape))
            #BUG
            #with open(REL_EMBEDDING_PATH, 'w+') as handle:
             #   for line in rel_emb:
              #     write_line = " ".join(line)
               #    handle.write(write_line + "\n")

          #  np.savetxt(REL_EMBEDDING_PATH, rel_emb, delimiter = " ")
        
        if step % LOG_LOSS == 0 and step > 0:
            # log loss weights
            _write_data_to_file(COLL_LOSS_PATH, loss)
        
        # Log the loss, if necessary.
        if step % LOG_STEPS == 0:
            #LOGGER.info('Step %6d | Loss: %10.4f', step, loss)

            session.run(dev_iterator.initializer)
            #valid_loss = session.run((model.obj_weighted_loss), 
             #                                            feed_dict= {model.eval_iterator_handle: dev_iterator_handle,
              #                                                       model.obj_weight: obj_weight})
            #LOGGER.info('Step %6d | Train Loss: %10.4f | Validation Loss: %10.4f | Obj Weight: %10.4f | Reg Weight: %10.4f', step, loss, valid_loss, baseline_weight, reg_weight)
            train_loss = loss
            #valid_loss = valid_pos_loss + valid_neg_loss
            LOGGER.info('Step %6d | Agg Loss: %10.4f | Train Loss %10.4f ', step, loss, train_loss)
            #if (valid_loss - prev_valid_loss > .0001) and start_reg_step is None:
            """
            if step >= 10000 and start_reg_step is None:
                print('Starting Regularization')
                start_reg_step = step
                reg_weight = .01
                is_reg = True
            prev_valid_loss = valid_loss
            """
        # Write summaries, if necessary.
        if summaries is not None:
            summary_writer.add_summary(summaries, step)

        # Evaluate, if necessary.
        if step % EVAL_STEPS == 0 and step > NUM_PRETRAIN_STEPS:
         #   LOGGER.info('Running dev evaluation.')
          #  session.run(dev_iterator.initializer)
           # ranking_and_hits(
            #    model, EVAL_PATH, dev_iterator_handle,
             #   'dev_evaluation', session)
            LOGGER.info('Running test evaluation.')
            session.run(test_iterator.initializer)
            ranking_and_hits(
                model, EVAL_PATH, test_iterator_handle,
                'test_evaluation', session)
            #ranking_and_hits(
             #   model, EVAL_PATH, train_iterator_handle,
              #  'train_evaluation', session)
            entity_embeddings = session.run(model.variables['ent_emb'])
            save_obj(entity_embeddings, ENTITY_EMB_PATH)
            save_obj(output_bias, OUTPUT_BIAS_PATH)


        if step % CKPT_STEPS == 0 and step > NUM_PRETRAIN_STEPS:
            LOGGER.info('Saving checkpoint at %s.', CKPT_PATH)
            saver.save(session, CKPT_PATH)



if __name__ == '__main__':
    main()
