from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import os
import pickle
import tensorflow as tf
import yaml

from qa_cpg import data
from qa_cpg.models import ConvE
from qa_cpg.metrics import ranking_and_hits
from qa_cpg.utils.dict_with_attributes import AttributeDict

logger = logging.getLogger(__name__)


def _evaluate(data_iterator, data_iterator_handle, name, summary_writer, step):
    logger.info('Running %s at step %d...', name, step)
    session.run(data_iterator.initializer)
    mr, mrr, hits = ranking_and_hits(model, eval_path, data_iterator_handle, name, session)

    metrics = {'mr': mr, 'mrr': mrr}

    if cfg.eval.summary_steps is not None:
        summary = tf.Summary()
        for hits_level, hits_value in hits.items():
            summary.value.add(tag=name+'/hits@'+str(hits_level), simple_value=hits_value)
            metrics['hits@'+str(hits_level)] = hits_value
        summary.value.add(tag=name+'/mrr', simple_value=mrr)
        summary.value.add(tag=name+'/mr', simple_value=mr)
        summary_writer.add_summary(summary, step)
        summary_writer.flush()

    return metrics


# Parameters.
use_cpg = True
save_best_embeddings = True

# Load data.
data_loader = data.UMLSLoader()

# Load configuration parameters.
model_descr = 'cpg' if use_cpg else 'plain'
config_path = 'qa_cpg/configs/config_%s_%s.yaml' % (data_loader.dataset_name, model_descr)
with open(config_path, 'r') as file:
    cfg_dict = yaml.load(file)
print(cfg_dict)
cfg = AttributeDict(cfg_dict)

# Compose model name based on config params.
model_name = '{}-{}-ent_emb_{}-rel_emb_{}-batch_{}-prop_neg_{}-num_labels_{}'.format(
    model_descr,
    data_loader.dataset_name,
    cfg.model.entity_embedding_size,
    cfg.model.relation_embedding_size,
    cfg.training.batch_size,
    cfg.training.prop_negatives,
    cfg.training.num_labels)
# Add more CPG-specific params to the model name.
suffix = '-context_batchnorm_{}'.format(cfg.context.context_rel_use_batch_norm) if use_cpg else ''
suffix += '-OneIter-BNTrainPlaceholder-bn_momentum_0.99'
model_name += suffix

# Create directories for saving downloaded data, summaries, logs and checkpoints.
working_dir = os.path.join(os.getcwd(), 'temp', data_loader.dataset_name)
data_dir = os.path.join(working_dir, 'data')
log_dir = os.path.join(working_dir, 'models', model_name, 'logs')
summaries_dir = os.path.join(working_dir, 'summaries', model_name)
ckpt_dir = os.path.join(working_dir, 'checkpoints', model_name, 'model_weights.ckpt')
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, 'model_weights.ckpt')
eval_path = os.path.join(working_dir, 'evaluation', model_name)
os.makedirs(eval_path, exist_ok=True)
config_save_dir = os.path.join(working_dir, 'configs', model_name)
os.makedirs(config_save_dir, exist_ok=True)
config_save_path = os.path.join(config_save_dir, 'config.yml')
if save_best_embeddings:
    embed_file = os.path.join(eval_path, 'best_embeddings.ckpt')

# Save the config to file, to keep track of running configuration for the results.
with open(config_save_path, 'w') as outfile:
    yaml.dump(cfg_dict, outfile, default_flow_style=False)

if __name__ == '__main__':
    data_loader.create_tf_record_files(data_dir)

    # Create the model.
    with tf.device(cfg.training.device):
        # We are using resource variables because due to some implementation details, this allows us to
        # better utilize GPUs while training.
        with tf.variable_scope('variables', use_resource=True):
            model = ConvE(model_descriptors={
                'use_negative_sampling': cfg.training.num_labels is not None,
                'label_smoothing_epsilon': cfg.model.label_smoothing_epsilon,
                'num_ent': data_loader.num_ent,
                'num_rel': data_loader.num_rel,
                'ent_emb_size': cfg.model.entity_embedding_size,
                'rel_emb_size': cfg.model.relation_embedding_size,
                'concat_rel': cfg.model.concat_rel,
                'context_rel_conv': cfg.context.context_rel_conv,
                'context_rel_out':  cfg.context.context_rel_out,
                'context_rel_dropout': cfg.context.context_rel_dropout,
                'context_rel_use_batch_norm': cfg.context.context_rel_use_batch_norm,
                'input_dropout': cfg.model.input_dropout,
                'hidden_dropout': cfg.model.feature_map_dropout,
                'output_dropout': cfg.model.output_dropout,
                'learning_rate': cfg.training.learning_rate,
                'batch_size': cfg.training.batch_size,
                'add_loss_summaries': cfg.eval.add_loss_summaries,
                'add_variable_summaries': cfg.eval.add_variable_summaries,
                'add_tensor_summaries': cfg.eval.add_tensor_summaries,
                'batch_norm_momentum': 0.99})

    # Create dataset iterator initializers.
    train_dataset = data_loader.train_dataset(
        directory=data_dir,
        batch_size=cfg.training.batch_size,
        include_inv_relations=True,
        buffer_size=1024,
        prefetch_buffer_size=16,
        prop_negatives=cfg.training.prop_negatives,
        num_labels=cfg.training.num_labels,
        cache=cfg.training.cache_data)
    train_eval_dataset = data_loader.eval_dataset(
        directory=data_dir,
        dataset_type='train',
        batch_size=cfg.training.batch_size,
        include_inv_relations=False,
        buffer_size=1024,
        prefetch_buffer_size=16)
    dev_eval_dataset = data_loader.eval_dataset(
        directory=data_dir,
        dataset_type='dev',
        batch_size=cfg.training.batch_size,
        include_inv_relations=False,
        buffer_size=1024,
        prefetch_buffer_size=16)
    test_eval_dataset = data_loader.eval_dataset(
        directory=data_dir,
        dataset_type='test',
        batch_size=cfg.training.batch_size,
        include_inv_relations=False,
        buffer_size=1024,
        prefetch_buffer_size=16)

    train_iterator = train_dataset.make_one_shot_iterator()
    train_eval_iterator = train_eval_dataset.make_initializable_iterator()
    dev_eval_iterator = dev_eval_dataset.make_initializable_iterator()
    test_eval_iterator = test_eval_dataset.make_initializable_iterator()

    # Log some information.
    logger.info('Number of entities: %d', data_loader.num_ent)
    logger.info('Number of relations: %d', data_loader.num_rel)

    # Create a TensorFlow session and start training.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(summaries_dir, session.graph)

    # Initialize the values of all variables and the train dataset iterator.
    session.run(tf.global_variables_initializer())

    # Obtain the dataset iterator handles.
    train_iterator_handle = session.run(train_iterator.string_handle())
    train_eval_iterator_handle = session.run(train_eval_iterator.string_handle())
    dev_eval_iterator_handle = session.run(dev_eval_iterator.string_handle())
    test_eval_iterator_handle = session.run(test_eval_iterator.string_handle())

    validation_metric = cfg.eval.validation_metric
    best_metrics_dev = {validation_metric: -np.inf}
    metrics_test_at_best_dev = {validation_metric: -np.inf}
    best_iter = None
    for step in range(cfg.training.max_steps):
        feed_dict = {
            model.is_train: True,
            model.input_iterator_handle: train_iterator_handle}

        if model.summaries is not None and cfg.eval.summary_steps is not None and step % cfg.eval.summary_steps == 0:
            summaries, loss, _ = session.run((model.summaries, model.loss, model.train_op), feed_dict)
            summary_writer.add_summary(summaries, step)
            summary_writer.flush()
        else:
            loss, _ = session.run((model.loss, model.train_op), feed_dict)

        # Log the loss, if necessary.
        if step % cfg.eval.log_steps == 0:
            logger.info('Step %6d | Loss: %10.4f', step, loss)

        # Evaluate, if necessary.
        if step % cfg.eval.eval_steps == 0:
            # Perform evaluation.
            if cfg.eval.eval_on_train:
                _evaluate(train_eval_iterator, train_eval_iterator_handle, 'train_evaluation', summary_writer, step)
            if cfg.eval.eval_on_dev:
                metrics_dev = _evaluate(
                    dev_eval_iterator, dev_eval_iterator_handle, 'dev_evaluation', summary_writer, step)
            if cfg.eval.eval_on_test:
                metrics_test = _evaluate(
                    test_eval_iterator, test_eval_iterator_handle, 'test_evaluation', summary_writer, step)
            if cfg.eval.eval_on_dev and cfg.eval.eval_on_test:
                if best_metrics_dev[validation_metric] < metrics_dev[validation_metric]:
                    best_metrics_dev = metrics_dev
                    metrics_test_at_best_dev = metrics_test
                    best_iter = step
                    if save_best_embeddings:
                        # Save relation and entity embeddings at the best validation point.
                        rel_embed, ent_embed = session.run([model.variables['rel_emb'], model.variables['ent_emb']])
                        pickle.dump([rel_embed, ent_embed], open(embed_file, 'wb'))
                logger.info('Best dev %s so far is at step %d. Best dev metrics: %s',
                            validation_metric, best_iter, str(best_metrics_dev))
                logger.info('Test metrics at best dev: %s', str(metrics_test_at_best_dev))

        if step % cfg.eval.ckpt_steps == 0 and step > 0:
            logger.info('Step %d. Saving checkpoint at %s...', step, ckpt_path)
            saver.save(session, ckpt_path)

    if cfg.eval.eval_on_dev and cfg.eval.eval_on_test:
        logger.info('Best dev %s so far is at step %d. Best dev metrics: %s',
                    validation_metric, best_iter, str(best_metrics_dev))
        logger.info('Test metrics at best dev: %s', str(metrics_test_at_best_dev))
