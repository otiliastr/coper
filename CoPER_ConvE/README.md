# CoPER-ConvE & ConvE

## Requirements
```$pip install -r requirements.txt```

## Experiment Pipeline
We describe how to run experiments using a demo example. Let us say that we wanted to examine the performance of CoPER-ConvE on the WN18RR dataset. 
1. We first need to specify the exact experiment configuration such as training time, model hyperparameters, and parameter generation architecture. Thus, navigate to 'qa_cpg/configs/config_[dataset]_[experiment_type].yaml', where '[dataset]' in this case is 'WN18RR' and '[experiment_type]' is 'cpg'. Please refer to section 'Configuration Parameters' below for additional information regarding possible configuration options.
2. Once you are satisfied with our configuration, navigate to 'ga_cpg/run_cpg.py' and open it. Change the lines (63-71) to:
```python
# Parameters.
model_type = 'cpg'  # Model type. Choose from ('cpg', 'param_lookup', 'plain')
save_best_embeddings = True   # save best entity and relation embeddings through training
model_load_path = None        # evaluate pretrained model 

# Load data.
data_loader = data.WN18RRLoader()   # desired dataset to experiment on
``` 
Note that model_type = 'cpg' indicates that you would like to use *parameter sharing* between relations (e.g. g_linear or g_MLP), while 'param_lookup' indicates embedding lookup (i.e. g_lookup). Additionally, 'plain' corresponds to ConvE.

Now we have everything we need to begin training!

3. Simply: 
```
$python -m qa_cpg.run_cpg
```
for cpu training or 
```
$CUDA_VARIABLE_DEVICES=0 python -m qa_cpg.run_cpg
```
for gpu.

### Configuring Datasets
We have support for the following popular benchmark datasets: Nations, UMLS, Kinship, FB15k, FB15k-237, WN18, WN18RR, NELL-995, and YAGO3-10. Each dataset can be loaded and experimented on simply be calling the corresponding 'Loader' class. The complete list of loaders can be found under 'qa_cpg/data.py'. **Note**: You do not need to worry about downloading the relevant data beforehand. The 'loader' class will do that for you in case it does not already exist. 

### Configuration Parameters
Below is an example config (from 'config_WN18RR_cpg.yaml') which explains experiment config options:
```
model:
  entity_embedding_size: 200 # Entity embedding size
  relation_embedding_size: 8 # Relation embedding size
  concat_rel: False # Set to `False` for plain ConvE, 'True' only if you want the relation to be concatenated in in the projection/output layer.
  input_dropout: 0.2 # input dropout
  feature_map_dropout: 0.3 # feature map dropout
  output_dropout: 0.2 # output dropout
  label_smoothing_epsilon: 0.1 # label smoothing for cross entropy loss
  batch_norm_momentum: 0.1 # batch norm momentum
  batch_norm_train_stats: True # If true, during training, batch norm will use a moving average of train samples.
context:
  context_rel_conv: # Leave empty for plain ConvE. Put list of hidden layer sizes for CPG. Empty list = g_linear
  context_rel_out: [] # Leave empty for plain ConvE. Put list of hidden layer sizes for CPG. Empty list = g_linear
  context_rel_dropout: 0.2 # Dropout in parameter generator. Note: dropout only applied for g_MLP
  context_rel_use_batch_norm: True # Whether to use batch normalization in parameter generator
training:
  learning_rate: 0.001
  batch_size: 512
  device: '/GPU:0'
  max_steps: 2000000 # Maximum training steps. Note: we do not train by *epochs* but instead by *batch steps*. However, you can achieve the mapping between steps to epochs by #steps_in_epoch = #data_size / #batch_size
  prop_negatives: 10.0 # Proportion of negatives to positives to use in negative sampling
  num_labels: 100 # Total number of labels considered during training when doing negative sampling. Must be > prop_negatives.
  one_positive_label_per_sample: False  # If True, it uses one positive answer per sample, and fills up tp num_labels with negatives.
  cache_data: True # whether to cache batch data 
eval:
  validation_metric: hits@1  # Metric specifying performance comparisons during training. Choose among: mr, mrr, hits@1, hits@10, hits@20.
  log_steps: 100 # Frequency of logging training behavior
  ckpt_steps: 50000 # Frequency of saving model checkpoints
  eval_steps: 5000 # Frequency of evaluating data
  summary_steps: 100 # Frequency of creating variable summaries
  eval_on_train: False # Whether to evaluate on training dataset
  eval_on_dev: True # Whether to evaluate on validation dataset
  eval_on_test: True # Whether to evaluate on test set
  add_loss_summaries: True # Whether to add loss summaries for tensorboard viz
  add_variable_summaries: False # Whether to add variable summaries for tensorboard viz
  add_tensor_summaries: False # Whether to add tensor summaries for tensorboard viz
```

### Testing Environments
All our code was tested on python3.6 and Tensorflow-gpu==1.14
