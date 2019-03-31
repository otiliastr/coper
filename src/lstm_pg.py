import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import itertools
import numbers
import numpy as np
import os
from functools import reduce
from operator import mul


# Contextual Parameter Generator Class for ConvE and other methods
# network_structure: dimensions of input to all hidden layers of network
# - input is assumed to be first element of network_structure
# - last element is assumed to be last hidden layer of network
# output_shape: dimensions of the desired paired network parameters
# dropout: probability for dropout
# use_batch_norm: whether to use batch_norm
# batch_norm_momentum: momentum for batchnorm
# use_bias: whether CPG network should have bias
class ContextualParameterGenerator(nn.Module):
    def __init__(self, network_structure, output_shape, dropout, use_batch_norm=False,
                 batch_norm_momentum=0.99, use_bias=False):
        super(ContextualParameterGenerator, self).__init__()
        self.network_structure = network_structure
        self.output_shape = output_shape
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        # print('use bias: {}'.format(self.use_bias))
        self.flattened_output = reduce(mul, output_shape, 1)

        self.projections = nn.ModuleList([])
        layer_input = network_structure[0]
        for layer_output in self.network_structure[1:]:
            print('inside loop!')
            self.projections.append(nn.Linear(layer_input, layer_output, bias=self.use_bias))
            if use_batch_norm:
                self.projections.append(nn.BatchNorm1d(num_features=layer_output,
                                                       momentum=batch_norm_momentum))
            self.projections.append(nn.ReLU())
            self.projections.append(nn.Dropout(p=self.dropout))
            layer_input = layer_output

        self.projections.append(nn.Linear(layer_input, self.flattened_output, bias=self.use_bias))
        self.network = nn.Sequential(*self.projections)

    def forward(self, query_emb):
        #print('the device of the CPG network is: {}'.format(self.network.device))
        # print('query embedding device: {}'.format(query_emb.device))
        flat_params = self.network(query_emb)
        params = flat_params.view([-1] + self.output_shape)
        # print('CPG shape: {}'.format(params.shape))
        return params

class PGLSTM(nn.Module):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
       sequence.
       """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0., bias=True, context_info=None):
        super(PGLSTM, self).__init__()

        """
        This module performs a single through of an LSTM with the choice of using Parameter
        Generation or not. Currently, there is no stacking code
        
        :param input_size: size of the LSTM input
        :param hidden_size: size of the hidden/cell states
        :param num_layers: number of LSTM cells to stack
        :param context_info: whether to generate parameters of LSTM via some context
            - None: Vanilla LSTM
            - Dict: Parameter Generated LSTM
                - network_structure: DNN architecture of CPG
                - dropout: amount of dropout to apply in CPG network
                - use_batch_norm: whether to use batchnorm in DNN CPG network
                - batch_norm_momentum: batch norm momentum amount
                - use_bias: whether to use bias in CPG generation

        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.context_info = context_info
        self.use_cpg = self.context_info is not None
        # initialize respective parameter/instruction lists
        self.dropouts = nn.ModuleList()
        if self.use_cpg:
            self.weights = nn.ModuleList()
            self.biases = nn.ModuleList()
        else:
            self.all_gates = nn.ModuleList()
        # create stacked logic
        for layer in range(self.num_layers):

            if self.use_cpg:
                # generate each LSTM parameters via parameter generator
                weights = ContextualParameterGenerator(
                    network_structure=[self.input_size] + self.context_info['network_structure'],
                    output_shape=[self.input_size + self.hidden_size, 4 * self.hidden_size],
                    dropout=self.context_info['dropout'],
                    use_batch_norm=self.context_info['use_batch_norm'],
                    batch_norm_momentum=self.context_info['batch_norm_momentum'],
                    use_bias=self.context_info['use_bias'])
                biases = ContextualParameterGenerator(
                    network_structure=[self.input_size] + self.context_info['network_structure'],
                    output_shape=[4 * self.hidden_size],
                    dropout=self.context_info['dropout'],
                    use_batch_norm=self.context_info['use_batch_norm'],
                    batch_norm_momentum=self.context_info['batch_norm_momentum'],
                    use_bias=self.context_info['use_bias'])
                # append weights to parameter lists
                self.weights.append(weights)
                self.biases.append(biases)

            else:
                # create linear transformations for each LSTM cell
                all_gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
                # append to stacked LSTM list
                self.all_gates.append(module=all_gates)

            # use dropout on up to penultimate layer
            if self.dropout > 0. and (layer < (self.num_layers - 1)):
                self.dropouts.append(nn.Dropout(p=self.dropout))

    def forward(self, input, past_states, context=None):
        """
        Perform forward layer of Deep LSTM
        :param input: Input data, [BatchSize, 1, EntEmbSize]
        :param past_states: States from previous (ent, rel), [NumLayers, BatchSize, HiddenSize]
        :param context: Whether to use context PG or not
            - None: Use normal Deep LSTM
            - not None: use PG, [BatchSize, RelEmbSize]
        :return:
            - output: [BatchSize, 1, HiddenSize]
            - (hidden_states, cell_states):
                - hidden_states: [NumLayers, BatchSize, HiddenSize]
                - cell_states: [NumLayers, BatchSize, HiddenSize]
        """
        # record state changes
        hidden_states = None
        cell_states = None

        past_hidden_states = past_states[0]
        past_cell_states = past_states[1]

        for layer in range(self.num_layers):
            hidden_state = past_hidden_states[:, layer, :]
            cell_state = past_cell_states[:, layer, :]
            # print('input size: {} | hidden state size: {}'.format(input.size(), hidden_state.size()))
            cell_input = torch.cat((input, hidden_state), dim=-1)

            if self.use_cpg:
                weights = self.weights[layer](context)
                biases = self.biases[layer](context)
                # print(weights.size())
                # print(biases.size())
                all_gates = torch.einsum('ij,ijk->ik', cell_input, weights) + biases
                # all_gates = torch.bmm(cell_input, weights)
            else:
                #print('input size: {}'.format(input.size()))
                #print('hidden state size: {}'.format(hidden_state.size()))
                #print('Cell input size: {}'.format(cell_input.size()))
                all_gates = self.all_gates[layer](cell_input)

            input_gate, forget_gate, add_gate, output_gate = all_gates.chunk(4, -1)

            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            add_gate = torch.tanh(add_gate)
            output_gate = torch.sigmoid(output_gate)

            cell_state = (cell_state * forget_gate) + (input_gate * add_gate)
            hidden_state = torch.tanh(cell_state) * output_gate
            # add inter LSTM dropout
            if self.dropout > 0 and (layer < (self.num_layers - 1)):
                input = self.dropouts[layer](hidden_state)
            else:
                input = hidden_state
            # [BatchSize, HiddenSize] --> [1, BatchSize, HiddenSize] for stacking
            # we stack over depth to create the same output as torch.nn.LSTM
            hidden_state_ = hidden_state.unsqueeze(1)
            cell_state_ = cell_state.unsqueeze(1)

            if hidden_states is None:
                hidden_states = hidden_state_
                cell_states = cell_state_
            else:
                hidden_states = torch.cat((hidden_states, hidden_state_), dim=1)
                cell_states = torch.cat((cell_states, cell_state_), dim=1)

            # hidden_states.append(hidden_state)
            # cell_states.append(cell_state)

        output = input.view(-1, 1, self.hidden_size)
        # hidden_states = hidden_states.view(-1, 1, self.hidden_size)
        # cell_states = cell_states.view(-1, 1, self.hidden_size)
        # hidden_states = torch.cat(hidden_states, 0).view(input.size(0), *output[0].size())

        return output, (hidden_states, cell_states)


if __name__ == '__main__':
    print('analyze normal:')
    pglstm = PGLSTM(input_size=10, hidden_size=10, num_layers=3)
    rnn = nn.LSTM(10, 10, 3, batch_first=True)

    h0 = torch.randn(5, 3, 10)
    c0 = torch.randn(5, 3, 10)
    input = torch.rand(5, 10)

    output, (h1, c1) = pglstm(input=input, past_states=(h0, c0), context=None)
    # output_, (h1_, c1_) = rnn(input.view(5, 1, 10), (h0, c0))
    print(output.size())
    # print(output_.size())
    print('#'* 80)
    print(h1.size())
    # print(h1_.size())
    print('#' * 80)
    print(c1.size())
    # print(c1_.size())
    print(list(rnn.state_dict()))
    print(list(pglstm.state_dict()))
    for name, param in pglstm.named_parameters():
        print(name)
    # for name in pglstm.state_dict():
    #     print(name)

    print('analyze PG:')
    pglstm = PGLSTM(input_size=10, hidden_size=10, num_layers=3, context_info={'network_structure': [],
                                                                               'dropout': .5,
                                                                               'use_batch_norm': True,
                                                                               'batch_norm_momentum': .1,
                                                                               'use_bias': True})
    context = torch.rand(5, 10)
    output, (h1, c1) = pglstm(input=input, past_states=(h0, c0), context=context)
    print(output.size())
    print(h1.size())
    print(c1.size())
    # print(list(pglstm.named_parameters()))
    for name, param in pglstm.named_parameters():
        print(name)
    # for name in pglstm.named_parameters():
    #     print(name)

    # print(output)
    # print(output_)
    # print(output.view(5, 1, 10) == output_)


