import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
import tensorflow as tf

from spodernet.utils.global_config import Config
from spodernet.utils.cuda_utils import CUDATimer
from torch.nn.init import xavier_normal, xavier_uniform
from spodernet.utils.cuda_utils import CUDATimer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

timer = CUDATimer()


class Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.emb_e_real.weight.data)
        xavier_normal(self.emb_e_img.weight.data)
        xavier_normal(self.emb_rel_real.weight.data)
        xavier_normal(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred



class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        #print("The number of entities is: {}".format(num_entities))
        #print("The number of relations is: {}".format(num_relations))
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        #self.loss = torch.nn.BCELoss()
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368,Config.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, 10, 20)
        #print("The number of embedding_dims is: {}".format(Config.embedding_dim))
        #print("The entity size is: {}".format(e1.size()))
        #print("The embedding size is: {}".format(self.emb_e.weight.size()))
        #print("The embedded entity size is: {}".format(e1_embedded.size()))
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        #print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        #print("The size of x before matmul is: {}".format(x.size()))
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)
        
        return pred

class ConvE_tf(object):
    def __init__(self, num_entities, num_relations):
        super(ConvE_tf, self).__init__()
        # Placeholders
        # if not -1 should be Config.batch_size
        self.e1 = tf.placeholder(dtype=tf.int32, shape = [Config.batch_size, 1], name = "entity1")
        self.rel = tf.placeholder(dtype=tf.int32, shape = [Config.batch_size, 1], name = "relation")
        self.e2_multi = tf.placeholder(dtype=tf.float32, shape = [Config.batch_size, num_entities], name = "entity2_multi")

        with tf.variable_scope("Embedding"):
            self.emb_e = tf.get_variable("entity_embeddings",
                                         [num_entities, Config.embedding_dim],
                                         dtype = tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.emb_rel = tf.get_variable("relation_embeddings",
                                           [num_relations, Config.embedding_dim],
                                           dtype=tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope("Output_bias"):
            self.out_bias = tf.get_variable("output_bias",
                                            [num_entities],
                                            dtype=tf.float32,
                                            initializer=tf.zeros_initializer())

        self.e1_embedded = tf.reshape(tf.nn.embedding_lookup(self.emb_e,
                                                        self.e1,
                                                        name="e1_embedding"),
                                 [-1, 10, 20, 1])
        self.rel_embedded = tf.reshape(tf.nn.embedding_lookup(self.emb_rel,
                                                         self.rel,
                                                         name="rel_embedding"),
                                  [-1, 10, 20, 1])
        self.stacked_inputs = tf.concat([self.e1_embedded, self.rel_embedded], 1)
        # self.stacked_inputs = tf.nn.fused_batch_norm(self.stacked_inputs, [1.], [0.])
        self.stacked_inputs = tf.contrib.layers.batch_norm(self.stacked_inputs)
        self.stacked_dropout = tf.nn.dropout(self.stacked_inputs, 1 - Config.input_dropout)
        self.conv1 = tf.layers.conv2d(inputs=self.stacked_dropout,
                                      filters = 32,
                                      kernel_size = [3, 3],
                                      padding = 'valid')

        self.conv1_bn = tf.contrib.layers.batch_norm(self.conv1, [1.], [0.])
        # x = F.relu(x)
        self.conv1_relu = tf.nn.relu(self.conv1_bn)
        #         x = self.feature_map_drop(x)
        self.conv1_dropout = tf.nn.dropout(self.conv1_relu, 1 - Config.feature_map_dropout)
        #         x = x.view(Config.batch_size, -1)
        self.flat_tensor = tf.reshape(self.conv1_dropout, [Config.batch_size, 10368])#[1, Config.batch_size, -1, 1])
        # print(x.size())
        #         x = self.fc(x)
        self.fc = tf.contrib.layers.fully_connected(self.flat_tensor, Config.embedding_dim)
        #         x = self.hidden_drop(x)
        self.fc_dropout = tf.nn.dropout(self.fc, 1 - Config.dropout)
        #         x = self.bn2(x)
        self.fc_bn = tf.contrib.layers.batch_norm(self.fc_dropout)
        #         x = F.relu(x)
        self.fc_relu = tf.nn.relu(self.fc_bn)
        # self.fc_reshaped = tf.reshape(self.fc_relu, [Config.batch_size, -1])
        #         x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        self.mat_prod = tf.matmul(self.fc_relu, tf.transpose(self.emb_e))

        #         x += self.b.expand_as(x)
        self.mat_prod = self.mat_prod + tf.expand_dims(self.out_bias, 0)
        #         pred = F.sigmoid(x)
        self.pred = tf.sigmoid(self.mat_prod)
        self.elem_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.e2_multi, logits = self.pred)
        self.loss = tf.reduce_mean(self.elem_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = Config.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

# Add your own model here

class MyModel(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)

        # Add your model function here
        # The model function should operate on the embeddings e1 and rel
        # and output scores for all entities (you will need a projection layer
        # with output size num_relations (from constructor above)

        # generate output scores here
        prediction = F.sigmoid(output)

        return prediction
