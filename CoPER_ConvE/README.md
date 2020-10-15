# CoPER-ConvE & ConvE

## Requirements
To install all requirements, you can use:
```
$ pip install -r requirements.txt
```

## Experimental Pipeline
We describe how to run experiments using a demo example. 
Let's say that we wanted to examine the performance of CoPER-ConvE on the WN18RR dataset. 
Here are the main steps:

1. We first need to specify the exact experiment configuration such as training time, model 
hyperparameters, and parameter generation architecture. Thus, navigate to 
`qa_cpg/configs/config_[dataset]_[experiment_type].yaml`, where `[dataset]` in this case 
is `WN18RR` and `[experiment_type]` is `cpg`. Please refer to section 
"Configuration Parameters" below for additional information regarding possible configuration options.

2. Once you are satisfied with the configuration, navigate to `ga_cpg/run_cpg.py` and open it. 
Change the lines (63-71) to:
```python
# Parameters.
model_type = 'cpg'                 # Choose a model type from ('cpg', 'param_lookup', 'plain')
save_best_embeddings = True        # Save best entity and relation embeddings after training.
model_load_path = None             # Provide a path to pretrained model.

# Load data.
data_loader = data.WN18RRLoader()  # Select the data loader for the desired dataset.
``` 
Note that `model_type = 'cpg'` indicates that you would like to use *parameter sharing* 
between relations (e.g. g_linear or g_MLP), while `model_type = 'param_lookup'` indicates 
embedding lookup (i.e. g_lookup). Additionally, `model_type = 'plain'` corresponds to ConvE.

### Training
Now we have everything we need to begin training!

3. Simply run the following command for CPU training:
```bash
$ python -m qa_cpg.run_cpg
```
or the following command for GPU training:
```bash
$ CUDA_VISIBLE_DEVICES=[id] python -m qa_cpg.run_cpg
```
where [id] is the gpu id.

### Configuring Datasets
We have provided files for the main datasets used in our experiments: UMLS, 
Kinship, FB15k-237, WN18RR, and NELL-995. To use them, run the following command:
```bash
$ unzip datasets.zip
```
However, new datasets can be easily added using our base `_DataLoader` in the 
`data.py` file. Each dataset can be loaded and experimented on by calling the 
corresponding `Loader` class. The complete list of loaders can be found under 
[qa_cpg/data.py](https://github.com/otiliastr/coper/blob/master/CoPER_ConvE/qa_cpg/data.py). 

**Note**: Even if you do not have the relevant data already downloaded, the 
`_DataLoader` class should be able to download the requested data in the correct
file location for you, provided the data url exists.

#### Note on NELL-995
Like previous work, we evaluate performance on NELL-995 by combining the 
training and validation datasets together to create the dataset NELL-995-test. 
To run on NELL-995-test, run:
```python
data_loader = data.NELL995Loader(is_test=True, needs_test_set_cleaning=True)
```
where the parameter `needs_test_set_cleaning` denotes whether to filter out all 
entities or relations from the dev or test set which do not appear in training. 
This should be done for both FB15k-237 and NELL-995. 

### Configuration Parameters
Below is an example config (from [config_WN18RR_cpg.yaml](https://github.com/otiliastr/coper/blob/master/CoPER_ConvE/qa_cpg/configs/config_WN18RR_cpg.yaml)) which explains experiment config options:
```yaml
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
All our code was tested on python3.6 and tensorflow-gpu==1.14
