import tensorflow as tf
import numpy as np
import logging
import sys



    
class TFModel(object):
    def __init__(self):
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler(sys.stdout))
        root.setLevel(logging.DEBUG)
        return
    def get_next_batch(self, x, y, batch_size):
        """
        Shuffle a dataset and randomly fetch next batch
        """
        _positions = np.random.choice(x.shape[0], size=batch_size, replace=False)
        batch_data = x[_positions]
        batch_labels = y[_positions]
        return batch_data, batch_labels
    def get_input(self):
        pass
    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)
        return
    def nn_layer(self,input_tensor, output_dim, layer_name, act=tf.nn.relu):
        input_dim = input_tensor.get_shape()[-1].value
        return self.nn_layer_(input_tensor, input_dim, output_dim, layer_name, act=act)
    def nn_layer_(self,input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                self.variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.histogram_summary(layer_name + '/pre_activations', preactivate)               
            activations = act(preactivate, 'activation')
            tf.histogram_summary(layer_name + '/activations', activations)
            
        return activations
    def dropout_layer(self, to_be_dropped_layer):
        layer_id = int(to_be_dropped_layer.name.split('/')[0][-1])
        with tf.name_scope('dropout' + str(layer_id)):
            dropped = tf.nn.dropout(to_be_dropped_layer, self.keep_prob_placeholder)
            return dropped

    def add_inference_node(self):
        pass
    def add_loss_node(self):
        pass
    def add_optimizer_node(self):
        pass
    def add_evalmetrics_node(self):
        pass
    def add_visualize_node(self):
        pass
    def __build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.get_input()
            self.add_inference_node()
            self.add_loss_node()
            self.add_optimizer_node()
            self.add_evalmetrics_node()
            self.add_visualize_node()
        return
    def run_graph(self):
        return
    def clear_prev_summary(self):
        if tf.gfile.Exists(self.summaries_dir):
            tf.gfile.DeleteRecursively(self.summaries_dir)
        tf.gfile.MakeDirs(self.summaries_dir)
        return
    def run(self):
        self.clear_prev_summary()
        self.__build_graph()
        self.run_graph()
        return