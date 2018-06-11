Files for running tensorflow configuration of ConvE: 
All these files have been adapted to work with tensorflow models.
- model.py:
  - Contains code for all models (DistMult, ComplEx, original ConvE in pytorch), and ConvE tensorflow (as ConvE_tf)
- main.py:
  - Contains code for training and evaluating models
- evaluation.py:
  - Contains evaluation functions for models 

Running ConvE:
1) Follow install directions 1-4 here: https://github.com/TimDettmers/ConvE
2) Replace model.py, main.py, evaluation.py in ConvE with these three files
3) Follow model training instructions at https://github.com/TimDettmers/ConvE. Note to train tensorflow ConvE, simply specify 
   ConvE_tf in the cmd line arguments inplace of ConvE or other model

Caveats with our model:
1) The original ConvE architecture uses a loss function called BCELoss, which is regular binary cross entropy loss but without
   the sigmoid activations. Tensorflow does not have such a loss function, instead it has tf.losses.sigmoid_cross_entropy, 
   which combines the sigmoid activation of inputs into the binary cross entropy obj function. Thus, to compare the tensorflow
   ConvE with the pytorch one I used torch.nn.BCEWithLogitsLoss as the pytorch ConvE loss function, which is equivalent to
   tensorflows tf.losses.sigmoid_cross_entropy. We need to see how critical BCELoss is to strong performance over BCEWithLogitsLoss.
 2) The tensorflow implementation runs 2.5x slower than the pytorch implementation. I suspect this is because the pytorch one 
    is integrated with cuda, but the tensorflow one is not - I'm not exactly sure how to add that functionality yet.

