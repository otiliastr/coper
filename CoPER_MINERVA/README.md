# MINERVA & CoPER-MINERVA

Our method, CoPER-MINERVA, is an improvement, using our methods CoPER, of the 
MINERVA method proposed by Das, Rajarshi, et al. [Go for a walk and arrive at 
the answer: Reasoning over paths in knowledge bases using reinforcement learning.](
https://arxiv.org/pdf/1711.05851.pdf)


Our implementation of MINERVA is built on top of the reimplementation of MINERVA
available at [MultiHop-KG](https://github.com/salesforce/MultiHopKG), which is the official
repository of the publication
[Lin et. al. 2018. Multi-Hop Knowledge Graph Reasoning with Reward Shaping](https://arxiv.org/abs/1808.10568). 

CoPER-MINERVA further extends this code using our proposed architecture.

Below is the selectively copied and adapted README (as it pertains to our 
models) from [MultiHop-KG](https://github.com/salesforce/MultiHopKG), 
which describes how to run MINERVA, and our extension CoPER-MINERVA.

## Quick Start

### Environment Variables & Dependencies
#### Using Docker
Build the docker image
```
docker build -< Dockerfile -t multi_hop_kg:v1.0
```

Spin up a docker container and run experiments inside it.
```
nvidia-docker run -v `pwd`:/workspace/MultiHopKG -it multi_hop_kg:v1.0
```
*The rest of the readme assumes that one works interactively inside a container. If you prefer to run experiments outside a container, please change the commands accordingly.*

#### Manual Setup 
Alternatively, you can install Pytorch (>=0.4.1) manually and use the Makefile to set up the rest of the dependencies. 
```
make setup
```

### Data preprocessing
First, unpack the data files 
```
tar xvzf data-release.tgz
```
and run the following command to preprocess the datasets:
```
./experiment.sh configs/<dataset>.sh --process_data <gpu-ID>
```

where `<dataset>` is the name of any dataset folder in the `./data` directory. In our experiments, the five datasets used are: `umls`, `kinship`, `fb15k-237`, `wn18rr` and `nell-995`. 

`<gpu-ID>` is a non-negative integer number representing the GPU index.

### Training
Then the following command can be used to train the proposed models and baselines in the paper. 

To train RL models (policy gradient), run:
```
./experiment.sh configs/<dataset>.sh --train <gpu-ID>
```

**Note on batch size:** When training CoPER models, you may find that a batch size 
of 128 is too large to fit inside gpu memory. For this reason, you can instead 
specify a smaller batch size in the config file without needing to worry about 
CoPER training on a different batch size compared to ConvE (i.e. training will
only update once the mini-batches equal the original 128 batch size). 

### Evaluating pretrained models
To generate the evaluation results of a pre-trained model, simply change the 
`--train` flag in the commands above to `--inference`. 

For example, the following command performs inference with the RL models 
(policy gradient + reward shaping) and prints the evaluation results (on both 
dev and test sets):
```
./experiment.sh configs/<dataset>.sh --inference <gpu-ID>
```

### Note for the NELL-995 dataset 

On this dataset we split the original training data into `train.triples` and 
`dev.triples`, and the final model to test has to be trained with these two 
files combined. 

To obtain the correct test set results, you need to add the `--test` flag to
     all data pre-processing, training and inference commands.  
    ```
    ./experiment.sh configs/nell-995.sh --process_data <gpu-ID> --test
    ./experiment.sh configs/nell-995.sh --train <gpu-ID> --test
    ```

Leave out the `--test` flag during development.

### Changing the hyperparameters
To change the hyperparameters and other experiment set up, start from the 
[configuration files](configs).

### Implementation Details
We use mini-batch training in our experiments. To save the amount of paddings (which can cause memory issues and slow down computation for knowledge graphs that contain nodes with large fan-outs),
we group the action spaces of different nodes into buckets based on their sizes. Description of the bucket implementation can be found
[here](https://github.com/salesforce/MultiHopKG/blob/master/src/rl/graph_search/pn.py#L193) and 
[here](https://github.com/salesforce/MultiHopKG/blob/master/src/knowledge_graph.py#L164).
