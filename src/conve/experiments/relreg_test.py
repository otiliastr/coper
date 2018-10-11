from __future__ import absolute_import, division, print_function

import logging
import os
import math

import tensorflow as tf
import numpy as np

from ..data.method_5_loaders import *
from ..evaluation.metrics import ranking_and_hits
#from ..models.conve_struc_merged import ConvE
from ..models.conve_initial_baseline import ConvE
#from ..models.conve_autoencoder import ConvE
#from ..models.DistMultRelReg import DistMultReg
from ..models.bilinear_reg import BiLinearReg
#from ..models.bilinear_perturbation import BiLinearPerturbation

LOGGER = logging.getLogger(__name__)

DATA_LOADER = FB15k237Loader()#FB15k237Loader()

beta = 1.001
DATASET = 'FB15k237'
EXP_TYPE = 'reg'
REG_TYPE = 'method_5+'

REG_WEIGHT = '.0'
SIM_THRESHOLD = 'None'
NUM_PRETRAIN_STEPS = 0
POS_WEIGHT = 1.0
NEG_WEIGHT = 1.0


DEVICE = '/GPU:0'
MAX_STEPS = 10000000
LOG_STEPS = 100
SUMMARY_STEPS = None
CKPT_STEPS = 1000
EVAL_STEPS = 1000
LOG_LOSS = 100

EMB_SIZE = 64
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
               'sim_threshold': 0.85}


if EXP_TYPE == 'reg':
    print('HI')
    #MODEL_NAME = 'bi_non_linear_{}_{}_with_{}_and_batch_size_{}_emb_size_{}'.format(EXP_TYPE, REG_TYPE, DATASET, BATCH_SIZE, EMB_SIZE)
    #MODEL_NAME = 'bi_non_linear_{}_on_{}_with_{}_and_batch_size_{}_emb_size_{}'.format(EXP_TYPE, DATASET, REG_TYPE, BATCH_SIZE, EMB_SIZE)
    MODEL_NAME = 'bi_non_linear_{}_{}_{}_{}_{}_pos_weight_{}_neg_weight_{}_emb_{}_batch_size_{}'.format(DATASET, EXP_TYPE, REG_WEIGHT, REG_TYPE, RELREG_ARGS['sim_threshold'], POS_WEIGHT, NEG_WEIGHT, EMB_SIZE, BATCH_SIZE)
    #MODEL_NAME = 'bilinear_{}_{}_{}_{}_None_emb_{}_batch_size_{}'.format(DATASET, EXP_TYPE, REG_WEIGHT, REG_TYPE, EMB_SIZE, BATCH_SIZE)
else:
    MODEL_NAME = 'bi_non_linear_{}_emb_{}_batch_size_{}'.format(DATASET, EMB_SIZE, BATCH_SIZE)

WORKING_DIR = os.path.join(os.getcwd(), 'temp')
DATA_DIR = os.path.join(WORKING_DIR, 'data')
LOG_DIR = os.path.join(WORKING_DIR, 'models', MODEL_NAME, 'logs')
CKPT_PATH = os.path.join(WORKING_DIR, 'models', MODEL_NAME, 'model_weights.ckpt')
EVAL_PATH = os.path.join(WORKING_DIR, 'evaluation', MODEL_NAME)
REL_EMBEDDING_PATH = os.path.join(EVAL_PATH, 'rel_emb.txt')
OBJ_POS_LOSS_PATH = os.path.join(EVAL_PATH, 'obj_pos_loss.txt')
OBJ_NEG_LOSS_PATH = os.path.join(EVAL_PATH, 'obj_neg_loss.txt')
REG_LOSS_PATH = os.path.join(EVAL_PATH, 'reg_loss.txt')
COLL_LOSS_PATH = os.path.join(EVAL_PATH, 'coll_loss.txt')
os.makedirs(EVAL_PATH, exist_ok=True)

def _write_data_to_file(file_path, data):
    if os.path.exists(file_path):
        append_write = 'a'
    else:
        append_write = 'w+'
    with open(file_path, append_write) as handle:
        handle.write(str(data) + "\n")


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

            model = BiLinearReg(Config = {
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
    
    # model.log_parameters_info()

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
    reg_weight = float(REG_WEIGHT) #if SIM_THRESHOLD != 'None' else 0.0#* .98 ** (math.floor(step / 1000))
    sim_threshold = float(SIM_THRESHOLD) if SIM_THRESHOLD != 'None' else 0.0
    
    prev_valid_loss = np.inf
    session.run(dev_iterator.initializer)
    start_reg_step = None
    is_reg = False
    obj_pos_weight = POS_WEIGHT
    obj_neg_weight = NEG_WEIGHT
    for step in range(MAX_STEPS):
        #if step > 50000:
         #    reg_weight = min(.5, .1 * math.ceil((step - 50000.)/6081.))
        #obj_neg_weight = 1.0 #* (.9999 ** (step))
        #sim_threshold = 1.0 + .005 * step
        """
        if is_reg:
            reg_weight = min(10.0, .01 * (beta ** (step - start_reg_step)))
            #print("reg_weight is {}".format(reg_weight))
        else:
            baseline_weight = 1.0 #min(1., .01 * (float(1.001 ** (step - NUM_PRETRAIN_STEPS))))
            reg_weight = 0.0 #min(1., .1 * (float(1.0001 ** (step - NUM_PRETRAIN_STEPS))))
        #reg_weight = float(REG_WEIGHT) #* .98 ** (math.floor(step / 100))
        """
        feed_dict = {
            model.is_train: True,
            model.input_iterator_handle: train_iterator_handle,
            model.relreg_iterator_handle: relreg_iterator_handle,
            model.obj_pos_weight: obj_pos_weight,
            model.obj_neg_weight: obj_neg_weight,
            model.reg_weight: reg_weight, #* (float(1.0001 ** (step - 1000)) if step >= 1000 else 0.0),
            model.sim_threshold: sim_threshold
           }


        if model.summaries is not None and \
            SUMMARY_STEPS is not None and \
            step % SUMMARY_STEPS == 0:
            summaries, loss, obj_loss, reg_loss, _ = session.run(
                (model.summaries, model.collective_loss, model.bilinear_weighted_loss, 
                 model.reg_weighted_loss, model.train_op), feed_dict)
        else:
            summaries = None
            loss, obj_pos_loss, obj_neg_loss, reg_loss, _ = session.run((model.collective_loss, 
                                                                        model.obj_pos_weighted_loss,
                                                                        model.obj_neg_weighted_loss, 
                                                                        model.reg_weighted_loss, 
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
            _write_data_to_file(OBJ_POS_LOSS_PATH, obj_pos_loss)
            _write_data_to_file(OBJ_NEG_LOSS_PATH, obj_neg_loss)
            _write_data_to_file(REG_LOSS_PATH, reg_loss)
            _write_data_to_file(COLL_LOSS_PATH, loss)
        
        # Log the loss, if necessary.
        if step % LOG_STEPS == 0:
            #LOGGER.info('Step %6d | Loss: %10.4f', step, loss)

            session.run(dev_iterator.initializer)
            valid_pos_loss, valid_neg_loss = session.run((model.obj_pos_weighted_loss, 
                                                         model.obj_neg_weighted_loss), 
                                                         feed_dict= {model.input_iterator_handle: dev_iterator_handle,
                                                                   #model.relreg_iterator_handle: relreg_iterator_handle,
                                                                     model.obj_pos_weight: obj_pos_weight,
                                                                     model.obj_neg_weight: obj_neg_weight})
            #LOGGER.info('Step %6d | Train Loss: %10.4f | Validation Loss: %10.4f | Obj Weight: %10.4f | Reg Weight: %10.4f', step, loss, valid_loss, baseline_weight, reg_weight)
            train_loss = obj_pos_loss + obj_neg_loss
            valid_loss = valid_pos_loss + valid_neg_loss
            LOGGER.info('Step %6d | Agg Loss: %10.4f | Train Loss %10.4f | Validation Loss: %10.4f', step, loss, train_loss, valid_loss)
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

        if step % CKPT_STEPS == 0 and step > NUM_PRETRAIN_STEPS:
            LOGGER.info('Saving checkpoint at %s.', CKPT_PATH)
            saver.save(session, CKPT_PATH)


if __name__ == '__main__':
    main()
