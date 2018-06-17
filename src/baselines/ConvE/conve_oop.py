import tensorflow as tf
import functools

# taken from: https://danijar.com/structuring-your-tensorflow-models/
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class ConvE(object):

    def __init__(self, model_descriptors,
                 entities, relations, targets):

        self.num_entities = model_descriptors['num_entities']
        self.num_relations = model_descriptors['num_relations']
        self.embedding_dim = model_descriptors['embedding_dim']
        self.batch_size = model_descriptors['batch_size']
        self.learning_rate = model_descriptors['learning_rate']

        self.input_dropout = model_descriptors['input_dropout']
        self.hidden_dropout = model_descriptors['hidden_dropout']
        self.output_dropout = model_descriptors['output_dropout']

        self.entities = entities
        self.relations = relations
        self.targets = targets

        # constructor methods
        self.prediction
        self.loss
        self.optimize
        # self.evaluate

    @lazy_property
    def prediction(self):
        # Create variables
        emb_e = tf.get_variable("entity_embeddings",
                                [self.num_entities, self.embedding_dim],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        emb_rel = tf.get_variable("relation_embeddings",
                                  [self.num_relations, self.embedding_dim],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        out_bias = tf.get_variable("output_bias",
                                   [self.num_entities],
                                   dtype=tf.float32,
                                   initializer=tf.zeros_initializer())
        e1_embedded = tf.reshape(tf.nn.embedding_lookup(emb_e,
                                                        self.entities,
                                                        name="e1_embedding"),
                                      [-1, 10, 20, 1])
        rel_embedded = tf.reshape(tf.nn.embedding_lookup(emb_rel,
                                                         self.relations,
                                                         name="rel_embedding"),
                                       [-1, 10, 20, 1])
        stacked_inputs = tf.concat([e1_embedded, rel_embedded], 1)
        stacked_inputs = tf.contrib.layers.batch_norm(stacked_inputs)
        stacked_dropout = tf.nn.dropout(stacked_inputs, 1 - self.input_dropout)
        conv1 = tf.layers.conv2d(inputs=stacked_dropout,
                                 filters=32,
                                 kernel_size=[3, 3],
                                 padding='valid')

        conv1_bn = tf.contrib.layers.batch_norm(conv1, [1.], [0.])
        conv1_relu = tf.nn.relu(conv1_bn)
        conv1_dropout = tf.nn.dropout(conv1_relu, 1 - self.hidden_dropout)
        flat_tensor = tf.reshape(conv1_dropout, [self.batch_size, 10368])
        fc = tf.contrib.layers.fully_connected(flat_tensor, self.embedding_dim)
        fc_dropout = tf.nn.dropout(fc, 1 - self.output_dropout)
        fc_bn = tf.contrib.layers.batch_norm(fc_dropout)
        fc_relu = tf.nn.relu(fc_bn)
        mat_prod = tf.matmul(fc_relu, tf.transpose(emb_e))
        mat_prod = mat_prod + tf.expand_dims(out_bias, 0)
        return tf.sigmoid(mat_prod)

    @lazy_property
    def optimize(self):
        # elem_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.targets, logits = self.prediction)
        # loss = tf.reduce_mean(elem_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.loss)

    @lazy_property
    def loss(self):
        elem_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.targets, logits=self.prediction)
        loss = tf.reduce_mean(elem_loss)
        return loss

    @property
    def get_vars_not_updated(self):
        return ['e1_embedding:0', 'rel_embedding:0']


entity = tf.placeholder(dtype=tf.int32, shape = [None, 1], name = "entity")
relation = tf.placeholder(dtype=tf.int32, shape = [None, 1], name = "relation")
labels = tf.placeholder(dtype=tf.float32, shape = [None, 14543], name = "labels")

# a = ConvE_tf(14543, 476)

model_descriptors = {'num_entities': 14543,
                     'num_relations': 476,
                     'embedding_dim': 128,
                     'batch_size': 128,
                     'input_dropout': 0,
                     'hidden_dropout': 0,
                     'output_dropout': 0,
                     'learning_rate': .001}

ConvE(model_descriptors = model_descriptors,
      entities = entity,
      relations = relation,
      targets = labels)

