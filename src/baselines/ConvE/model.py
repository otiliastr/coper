import torch
import tensorflow as tf
# from torch.nn import functional as F, Parameter
# from torch.autograd import Variable


from spodernet.utils.global_config import Config
from spodernet.utils.cuda_utils import CUDATimer
# from torch.nn.init import xavier_normal, xavier_uniform
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ConvE(object):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
#         self.num_entities = num_entities
#         self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
#         self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
#         self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
#         self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
#         self.inp_drop = torch.nn.Dropout(Config.input_dropout)
#         self.loss = torch.nn.BCELoss()
        
        self.emb_e = tf.get_variable("entity_embeddings", 
                                     [num_entities, Config.embedding_dim],
                                    initializer =  tf.contrib.layers.xavier_initializer())
        self.emb_rel = tf.get_variable("relation_embeddings", 
                                       [num_relations, Config.embedding_dim],
                                      initializer = tf.contrib.layers.xavier_initializer())
        self.out_bias = tf.get_variable("output_bias", 
                                        [num_entities],
                                       initializer = tf.zeros_initializer())
        self.inp_drop_prob = Config.input_dropout
        self.hidden_drop_prob = Config.dropout
        self.feature_map_drop_prob = Config.feature_map_dropout
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits
        
#         self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
#         self.conv1 = tf.nn.conv2d()
#         self.bn0 = torch.nn.BatchNorm2d(1)
#         self.bn1 = torch.nn.BatchNorm2d(32)
#         self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
#         self.register_parameter('b', Parameter(torch.zeros(num_entities)))
#         fc_bias = tf.get_variable("fc_bias", [-1, 10368])
#         self.fc = torch.nn.Linear(10368,Config.embedding_dim)
#         print(num_entities, num_relations)
    
#     def init(self):
#         xavier_normal(self.emb_e.weight.data)
#         xavier_normal(self.emb_rel.weight.data)

    def forward(self, e1, rel):
#         e1_embedded= self.emb_e(e1).view(-1, 1, 10, 20)
        e1_embedded = tf.reshape(tf.nn.embedding_lookup(self.emb_e, 
                                                        e1, 
                                                        name = "e1_embedding"), 
                                 [-1, 1, 10, 20])
#         rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)
        rel_embedded = tf.reshape(tf.nn.embedding_lookup(self.emb_rel, 
                                                         rel, 
                                                         name = "rel_embedding"), 
                                  [-1, 1, 10, 20])
        

#         stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = tf.concat([e1_embedded, rel_embedded], 2)
#         stacked_inputs = self.bn0(stacked_inputs)
        stacked_inputs = tf.nn.fused_batch_norm(stacked_inputs, [1.], [0.])
#         x= self.inp_drop(stacked_inputs)
        stacked_dropout = tf.nn.dropout(stacked_inputs, self.inp_drop_prob)
#         conv1 = self.conv1(stacked_dropout)
        conv1 = tf.nn.conv2d(input = stacked_dropout, 
                         filter = [3, 3, 1, 32], 
                         strides = 1, 
                         padding = 0)
#         x= self.bn1(x)
        conv1_bn = tf.nn.fused_batch_norm(conv1, [1.], [0.])
        x= F.relu(x)
        conv1_relu = tf.nn.relu(conv1_bn)
#         x = self.feature_map_drop(x)
        conv1_dropout = tf.nn.dropout(conv1_relu, self.feature_map_drop_prob)
#         x = x.view(Config.batch_size, -1)
        flat_tensor = tf.reshape(conv1_dropout, [-1])
        #print(x.size())
#         x = self.fc(x)
        fc = tf.layers.dense(flat_tensor, 10368)
#         x = self.hidden_drop(x)
        fc_dropout = tf.nn.dropout(fc, self.hidden_drop_prob)
#         x = self.bn2(x)
        fc_bn = tf.nn.fused_batch_norm(fc_dropout, [1.], [0.])
#         x = F.relu(x)
        fc_relu = tf.nn.relu(fc_bn)
#         x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        mat_prod = tf.matmul(fc_relu, self.emb_e.transpose())

#         x += self.b.expand_as(x)
        mat_prod = mat_prod + tf.expand_dims(self.out_bias, 0)
#         pred = F.sigmoid(x)
        pred = tf.sigmoid(mat_prod)

        return pred
