import tensorflow as tf
import numpy as np
import logging
from bokeh.util.logconfig import level
import sys

from tensorflow.examples.tutorials.mnist import input_data


    
class TFModel(object):
    def __init__(self):
        self.num_steps = 100
        self.batch_size = 100
        self.summaries_dir = '/tmp/mnist_logs2'
        self.dropout= 0.9
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
#             tf.scalar_summary('dropout_keep_probability' + str(layer_id), self.keep_prob)
            dropped = tf.nn.dropout(to_be_dropped_layer, self.keep_prob)
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
    
    
class MnistTFModel(TFModel):
    def __init__(self):
        TFModel.__init__(self)
        return
    def add_visualize_node(self):
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(self.summaries_dir+ '/train',
                                        self.graph)
        self.test_writer = tf.train.SummaryWriter(self.summaries_dir + '/test')

        return
    def get_input(self):
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        self.mnist = input_data.read_data_sets('data',
                                    one_hot=True,
                                    fake_data=False)
        # Input placehoolders
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            self.y_true = tf.placeholder(tf.float32, [None, 10], name='y-input')
        self.keep_prob = tf.placeholder(tf.float32, name='drop_out')
        # below is just for the sake of visualization
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.x, [-1, 28, 28, 1])
            tf.image_summary('input', image_shaped_input, 10)
        
        return
    def add_inference_node(self):
        #output node self.pred
#         hidden1 = self.nn_layer(self.x, 784, 500, 'layer1')
#         dropped = self.dropout_layer(hidden1)
#         self.y_pred = self.nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)
        hidden1 = self.nn_layer(self.x, 500, 'layer1')
        dropped = self.dropout_layer(hidden1)
        
        hidden1 = self.nn_layer(dropped, 200, 'layer2')
        dropped = self.dropout_layer(hidden1)
        
        self.y_pred = self.nn_layer(dropped, 10, 'layer3', act=tf.nn.softmax)
        return
    def add_loss_node(self):
        #output node self.loss
        with tf.name_scope('loss'):
            diff = self.y_true * tf.log(self.y_pred)
            with tf.name_scope('total'):
                self.loss = -tf.reduce_mean(diff)
            tf.scalar_summary('cross entropy', self.loss)
        return
    
    def add_optimizer_node(self):
        #output node self.train_step
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        return
    def add_accuracy_node(self):
        #output node self.accuracy
        with tf.name_scope('evaluationmetrics'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y_true, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', self.accuracy)
        return
    def add_evalmetrics_node(self):
        self.add_accuracy_node()
        return
    def feed_dict(self,train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = self.mnist.train.next_batch(self.batch_size)
            k = self.dropout
        else:
            xs, ys = self.mnist.test.images, self.mnist.test.labels
            k = 1.0
        return {self.x: xs, self.y_true: ys, self.keep_prob: k}

    def run_graph(self):
        logging.debug("computeGraph")
        with tf.Session(graph=self.graph) as sess:
            tf.initialize_all_variables().run()
            logging.debug("Initialized")
            for step in range(self.num_steps + 1):
                summary, _ , acc_train= sess.run([self.merged, self.train_step, self.accuracy], feed_dict=self.feed_dict(True))
                self.train_writer.add_summary(summary, step)
                
                if step % 10 == 0:
                    summary, acc_test = sess.run([self.merged, self.accuracy], feed_dict=self.feed_dict(False))
                    self.test_writer.add_summary(summary, step)
                    logging.info("Step {}/{}, train: {:.3f}, test {:.3f}".format(step, self.num_steps, acc_train, acc_test))
        return


if __name__ == "__main__":   
    obj= MnistTFModel()
    obj.run()
