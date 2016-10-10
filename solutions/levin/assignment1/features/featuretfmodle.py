import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
import numpy as np
import logging
from bokeh.util.logconfig import level
import sys
from utility.tfbasemodel import TFModel
from assignment1.features.featuresmodel import FeaturesModel
from sklearn.preprocessing import OneHotEncoder

class PrepareData(FeaturesModel):
    def __init__(self):
        FeaturesModel.__init__(self)
        return
    def get_train_validationset(self):
        self.load_data()
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_val = self.y_val.reshape(-1, 1)
        self.y_test = self.y_test.reshape(-1, 1)
        logging.info("train:{}, val:{}, test{}:".format(self.y_train.shape, self.y_val.shape, self.y_test.shape))
        return self.X_train_feats, self.y_train, self.X_val_feats,self.y_val, self.X_test_feats,self.y_test,self.X_test
class FeatureTFModel(TFModel):
    def __init__(self):
        TFModel.__init__(self)
        
        self.batch_size = 128
        self.num_steps = self.batch_size*20

        self.summaries_dir = './logs/cifar'
        self.keep_dropout= 1.0
       
        logging.getLogger().addHandler(logging.FileHandler('logs/cifarnerual.log', mode='w'))
        return
    def add_visualize_node(self):
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(self.summaries_dir+ '/train',
                                        self.graph)
        self.test_writer = tf.train.SummaryWriter(self.summaries_dir + '/val')

        return
    def get_input(self):
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        prepare_data = PrepareData()
        num_class = 10
        self.x_train, self.y_train,self.x_validation,self.y_val, self.x_test,self.y_test,_= prepare_data.get_train_validationset()
        enc = OneHotEncoder(n_values=num_class, sparse=False)

        self.y_train  = enc.fit(self.y_train).transform(self.y_train )
        self.y_val  = enc.fit(self.y_val).transform(self.y_val )
        self.y_test  = enc.fit(self.y_test).transform(self.y_test )
        
        
        self.inputlayer_num = self.x_train.shape[1]
        self.outputlayer_num = num_class
        
        # Input placehoolders
        with tf.name_scope('input'):
            self.x_placeholder = tf.placeholder(tf.float32, [None, self.inputlayer_num], name='x_placeholder-input')
            self.y_true_placeholder = tf.placeholder(tf.float32, [None, self.outputlayer_num ], name='y-input')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='drop_out')
        
        return
    def add_inference_node(self):
        #output node self.pred
        hidden1 = self.nn_layer(self.x_placeholder, 500, 'layer1')
        dropped = self.dropout_layer(hidden1)
        
        output_layer = self.nn_layer(dropped, self.outputlayer_num, 'layer2')

        
        self.y_pred = tf.nn.softmax(output_layer)
        return
    def add_loss_node(self):
        #output node self.loss
        self.__add_crossentropy_loss()
        return
    def __add_crossentropy_loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('crossentropy'):
                self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_true_placeholder * tf.log(self.y_pred), reduction_indices=[1]))
            tf.scalar_summary('crossentropy', self.loss)
        return
    def euclidean_norm(self, tensor):
        with tf.name_scope("euclidean_norm"): #need to have this for tf to work
            squareroot_tensor = tf.square(tensor)
            euclidean_norm = tf.sqrt(tf.reduce_sum(squareroot_tensor))
            return euclidean_norm
    def add_optimizer_node(self):
        #output node self.train_step
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(1.0e-3)
#             optimizer = tf.train.GradientDescentOptimizer(5.0e-1)
#             grads_and_vars = optimizer.compute_gradients(self.loss)
#             self.ratio_w1 = self.euclidean_norm(grads_and_vars[0][0])/self.euclidean_norm(grads_and_vars[0][1])
#             self.ratio_w2 = self.euclidean_norm(grads_and_vars[2][0])/self.euclidean_norm(grads_and_vars[2][1])
#             grads = [item[0] for item in grads_and_vars]
#             vars   = [item[1] for item in grads_and_vars]
#             grads_l2norm = [self.euclidean_norm(item) for item in grads]
#             vars_l2norm = [self.euclidean_norm(item) for item in vars]
#             self.param_udpate_ratio = [] 
#             for i in range(len(grads_l2norm)):
#                 self.param_udpate_ratio.append(grads_l2norm[i] / vars_l2norm[i])
            
#             self.train_step = optimizer.apply_gradients(grads_and_vars)
            self.train_step = optimizer.minimize(self.loss)
        return
    def add_accuracy_node(self):
        #output node self.accuracy
        with tf.name_scope('evaluationmetrics'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.y_pred,1), tf.argmax(self.y_true_placeholder,1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', self.accuracy)
        return
    def add_evalmetrics_node(self):
        self.add_accuracy_node()
        return
    def feed_dict(self,feed_type):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if feed_type == "train":
            xs, ys = self.get_next_batch(self.x_train, self.y_train, self.batch_size)
            k = self.keep_dropout
            return {self.x_placeholder: xs, self.y_true_placeholder: ys, self.keep_prob_placeholder: k}
        if feed_type == "validation":
            xs, ys = self.x_validation, self.y_val
            k = 1.0
            return {self.x_placeholder: xs, self.y_true_placeholder: ys, self.keep_prob_placeholder: k}
        if feed_type == "wholetrain":
            xs, ys = self.x_train, self.y_train
            k = 1.0
            return {self.x_placeholder: xs, self.y_true_placeholder: ys, self.keep_prob_placeholder: k}
        # Now we are feeding test data into the neural network
        if feed_type == "test":
            xs, ys = self.x_test, self.y_test
            k = 1.0
            return {self.x_placeholder: xs, self.y_true_placeholder: ys, self.keep_prob_placeholder: k}
    def debug_training(self, sess, step, train_metrics,train_loss):
        if step % self.batch_size != 0:
            return
        summary, validation_loss, validation_metrics = sess.run([self.merged, self.loss, self.accuracy], feed_dict=self.feed_dict("validation"))
        self.test_writer.add_summary(summary, step)
        logging.info("Epoch {}/{}, train/val: {:.3f}/{:.3f}, train/val loss: {:.3f}/{:.3f}".format(step / self.batch_size, 
                                                                                                     self.num_steps / self.batch_size, 
                                                                                                     train_metrics, validation_metrics,\
                                                                                                            train_loss, validation_loss))

        return
    def get_final_result(self, sess, feed_dict):
        accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
        return accuracy
    def run_graph(self):
        logging.debug("computeGraph")
        with tf.Session(graph=self.graph) as sess:
            tf.initialize_all_variables().run()
            logging.debug("Initialized")
            for step in range(0, self.num_steps + 1):
                summary,  _ , train_loss, train_metrics =sess.run([self.merged, self.train_step, self.loss, self.accuracy], feed_dict=self.feed_dict("train"))
                self.train_writer.add_summary(summary, step)
                self.debug_training(sess, step, train_metrics, train_loss)
                
     
            train_accuracy = self.get_final_result(sess, self.feed_dict("wholetrain"))
            val_accuracy = self.get_final_result(sess, self.feed_dict("validation"))
            test_accuracy = self.get_final_result(sess, self.feed_dict("test"))
            logging.info("train:{:.3f}, val:{:.3f},test:{:.3f}".format(train_accuracy, val_accuracy, test_accuracy))  
        return


if __name__ == "__main__":   
    obj= FeatureTFModel()
    obj.run()