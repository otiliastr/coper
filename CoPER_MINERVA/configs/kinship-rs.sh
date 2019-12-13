#!/usr/bin/env bash

data_dir="data/kinship"
model="point.rs.cpg-conve"
group_examples_by_query="False"
use_action_space_bucketing="True"

bandwidth=400
entity_dim=200
relation_dim=50
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=2
bucket_interval=10
num_epochs=1000
num_wait_epochs=400
num_peek_epochs=2
batch_size=64  
train_batch_size=64
dev_batch_size=32
learning_rate=0.001
baseline="n/a"
grad_norm=5
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.9
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.05
relation_only="False"
beam_size=128

learning_rate_decay=.995
learning_rate=.001
hidden_dropout_rate=.3
feat_dropout_rate=.3

cpg_conv_net=-1
cpg_fc_net=
cpg_dropout=.5
cpg_batch_norm=True
cpg_batch_norm_momentum=.1
cpg_use_bias=False

distmult_state_dict_path="model/kinship-distmult-xavier-200-200-0.003-0.3-0.1/model_best.tar"
complex_state_dict_path="model/kinship-complex-RV-xavier-200-200-0.003-0.3-0.1/model_best.tar"
#conve_state_dict_path="model/kinship-conve-RV-xavier-200-200-0.003-32-3-0.2-0.3-0.2-0.1/model_best.tar"
conve_state_dict_path="model/kinship-cpg-conve-RV-xavier-200-50-0.003-32-3-0.2-0.5-0.5-0.1-None-[]-False-True-0.1/model_best.tar"

num_paths_per_entity=-1
margin=-1