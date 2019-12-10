#!/usr/bin/env bash

data_dir="data/kinship"
model="cpg-conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
entity_dim=200
relation_dim=50
hidden_dropout_rate=.3
feat_dropout_rate=.2
num_rollouts=1
bucket_interval=10
num_epochs=2000
num_wait_epochs=500
batch_size=512
train_batch_size=512
dev_batch_size=64
learning_rate=0.003
grad_norm=0
emb_dropout_rate=0.2
beam_size=128

cpg_conv_net=None
cpg_fc_net=[]
cpg_dropout=.5
cpg_batch_norm=True
cpg_batch_norm_momentum=.1
cpg_use_bias=False

num_negative_samples=20
margin=0.5
