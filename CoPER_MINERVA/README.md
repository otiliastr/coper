# This Directory contains all the source code necessary to experiment with MINERVA and CoPER-MINERVA. 

**Note**: Our code is built on top of the source code of MultiHop-KG (https://github.com/salesforce/MultiHopKG) from the paper,[Lin et. al. 2018. Multi-Hop Knowledge Graph Reasoning with Reward Shaping](https://arxiv.org/abs/1808.10568). 

Below is the selectively copied README (as it pertains to our models) from MultiHop-KG, which describes how to run MINERVA (and CoPER-MINERVA as it was created in the same ecosystem).

## Quick Start

### Environment Variables & Dependencies
#### Use Docker
Build the docker image
```
docker build -< Dockerfile -t multi_hop_kg:v1.0
```

Spin up a docker container and run experiments inside it.
```
nvidia-docker run -v `pwd`:/workspace/MultiHopKG -it multi_hop_kg:v1.0
```
*The rest of the readme assumes that one works interactively inside a container. If you prefer to run experiments outside a container, please change the commands accordingly.*

#### Mannually Set up 
Alternatively, you can install Pytorch (>=0.4.1) manually and use the Makefile to set up the rest of the dependencies. 
```
make setup
```

### Process data
First, unpack the data files 
```
tar xvzf data-release.tgz
```
and run the following command to preprocess the datasets.
```
./experiment.sh configs/<dataset>.sh --process_data <gpu-ID>
```

`<dataset>` is the name of any dataset folder in the `./data` directory. In our experiments, the five datasets used are: `umls`, `kinship`, `fb15k-237`, `wn18rr` and `nell-995`. 
`<gpu-ID>` is a non-negative integer number representing the GPU index.

### Train models
Then the following command can be used to train the proposed models and baselines in the paper. 

Train RL models (policy gradient)
```
./experiment.sh configs/<dataset>.sh --train <gpu-ID>
```

A note on Batch Size. When training CoPER models, you may find that a batch size of 128 is too large to fit inside gpu memory. For this reason, you can instead specify a smaller batchsize in the config file without needing to worry about CoPER training on a different batchsize compared to ConvE (i.e. training will only update once the minibatches equal the original 128 batch size). 

### Evaluate pretrained models
To generate the evaluation results of a pre-trained model, simply change the `--train` flag in the commands above to `--inference`. 

For example, the following command performs inference with the RL models (policy gradient + reward shaping) and prints the evaluation results (on both dev and test sets).
```
./experiment.sh configs/<dataset>.sh --inference <gpu-ID>
```

* Note for the NELL-995 dataset: 

  On this dataset we split the original training data into `train.triples` and `dev.triples`, and the final model to test has to be trained with these two files combined. 
  1. To obtain the correct test set results, you need to add the `--test` flag to all data pre-processing, training and inference commands.  
    ```
    ./experiment.sh configs/nell-995.sh --process_data <gpu-ID> --test
    ./experiment.sh configs/nell-995.sh --train <gpu-ID> --test
    ```
  2. Leave out the `--test` flag during development.

### Change the hyperparameters
To change the hyperparameters and other experiment set up, start from the [configuration files](configs).

### Notes on Implementation Details
We use mini-batch training in our experiments. To save the amount of paddings (which can cause memory issues and slow down computation for knowledge graphs that contain nodes with large fan-outs),
we group the action spaces of different nodes into buckets based on their sizes. Description of the bucket implementation can be found
[here](https://github.com/salesforce/MultiHopKG/blob/master/src/rl/graph_search/pn.py#L193) and 
[here](https://github.com/salesforce/MultiHopKG/blob/master/src/knowledge_graph.py#L164).

## Citation
If you find the resource in this repository helpful, please cite
```
@inproceedings{LinRX2018:MultiHopKG, 
  author = {Xi Victoria Lin and Richard Socher and Caiming Xiong}, 
  title = {Multi-Hop Knowledge Graph Reasoning with Reward Shaping}, 
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2018, Brussels, Belgium, October
               31-November 4, 2018},
  year = {2018} 
}
```
