import sys
import os
from astropy.units import ys

sys.path.insert(0, os.path.abspath('..'))

import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n import data_utils
from assignment1.cs231n.features import *
from utility.dumpload import DumpLoad
from assignment1.cs231n.classifiers.linear_classifier import LinearSVM
from assignment1.cs231n.classifiers.neural_net import TwoLayerNet
import cv2




class FeaturesModel(object):
    def __init__(self):
       
        return
    
    def get_CIFAR10_data(self, num_training=49000, num_validation=1000, num_test=1000):
        # Load the raw CIFAR-10 data
        cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
        # Subsample the data
        mask = range(num_training, num_training + num_validation)
        X_val = X_train[mask]
        y_val = y_train[mask]
        mask = range(num_training)
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = range(num_test)
        X_test = X_test[mask]
        y_test = y_test[mask]
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        return
    def save_sample_images(self):
        cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        num_training = X_train.shape[0]
        sample_indiecs = np.random.choice(num_training, size=5)
        sample_images= X_train[sample_indiecs]
        img_id = 0
        for sample in sample_images:
            img_id += 1
            image_name = './temp/img_' + str(img_id) + '.jpg'
            cv2.imwrite(image_name,sample)
            
        return
    def explore_sift(self):
        cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        sift_len = []
        for img in X_train:
            sift = cv2.SIFT()
            gray= cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)
            kp = sift.detect(gray,None)
            kp,des = sift.compute(gray, kp)
            if len(kp) == 0:
                image_name = './temp/zero_sift'+ '.jpg'
                cv2.imwrite(image_name, img)
                return
            sift_len.append(len(kp))
        print min(sift_len)
        print max(sift_len)
        print np.mean(sift_len)
        return
    def extract_features(self):
        num_color_bins = 10 # Number of bins in the color histogram
        feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
        self.X_train_feats = extract_features(self.X_train, feature_fns, verbose=True)
        self.X_val_feats = extract_features(self.X_val, feature_fns)
        self.X_test_feats = extract_features(self.X_test, feature_fns)
        
        # Preprocessing: Subtract the mean feature
        mean_feat = np.mean(self.X_train_feats, axis=0, keepdims=True)
        self.X_train_feats -= mean_feat
        self.X_val_feats -= mean_feat
        self.X_test_feats -= mean_feat
        
        # Preprocessing: Divide by standard deviation. This ensures that each feature
        # has roughly the same scale.
        std_feat = np.std(self.X_train_feats, axis=0, keepdims=True)
        self.X_train_feats /= std_feat
        self.X_val_feats /= std_feat
        self.X_test_feats /= std_feat
        
        # Preprocessing: Add a bias dimension
        self.X_train_feats = np.hstack([self.X_train_feats, np.ones((self.X_train_feats.shape[0], 1))])
        self.X_val_feats = np.hstack([self.X_val_feats, np.ones((self.X_val_feats.shape[0], 1))])
        self.X_test_feats = np.hstack([self.X_test_feats, np.ones((self.X_test_feats.shape[0], 1))])
        return
    def load_data(self):
        dump_load = DumpLoad('./temp/hogsdata.pickle')
        if not dump_load.isExisiting():
            self.get_CIFAR10_data()
            self.extract_features()
            preprocessed_dataset = self.X_train_feats, self.y_train, self.X_val_feats,self.y_val, self.X_test_feats,self.y_test, self.X_test
            dump_load.dump(preprocessed_dataset)
        
        self.X_train_feats, self.y_train, self.X_val_feats,self.y_val, self.X_test_feats,self.y_test,self.X_test = dump_load.load()    
        return
    def visulize_mistake(self,y_test_pred):
        examples_per_class = 8
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for cls, cls_name in enumerate(classes):
            idxs = np.where((self.y_test != cls) & (y_test_pred == cls))[0]
            idxs = np.random.choice(idxs, examples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
                plt.imshow(self.X_test[idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls_name)
        plt.show()
        return
    def svm_classifier(self):
        learning_rates = [1e-7, 3e-7,5e-7]
        regularization_strengths = [5e4, 1e4]
        
        results = {}
        best_val = -1   # The highest validation accuracy that we have seen so far.
        best_svm = None # The LinearSVM object that achieved the highest validation rate.
        num_iters = 1000
        for learning_rate in learning_rates:
            for regularization_strength in regularization_strengths:
                print "learning_rage {:.2e}, regularization_strength {:.2e}".format(learning_rate, regularization_strength)
                #train it
                svm = LinearSVM()
                svm.train(self.X_train_feats, self.y_train, learning_rate=learning_rate, reg=regularization_strength,
                              num_iters=num_iters, verbose=True)
                #predict
                y_train_pred = svm.predict(self.X_train_feats)
                training_accuracy = np.mean(self.y_train == y_train_pred)
                y_val_pred = svm.predict(self.X_val_feats)
                validation_accuracy = np.mean(self.y_val == y_val_pred)
                results[(learning_rate,regularization_strength)] = training_accuracy, validation_accuracy
                print "train accurcy {}, validation {}".format(training_accuracy, validation_accuracy)
                if validation_accuracy > best_val:
                    best_val = validation_accuracy
                    best_svm = svm
                
        # Print out results.
        for lr, reg in sorted(results):
            train_accuracy, val_accuracy = results[(lr, reg)]
            print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                        lr, reg, train_accuracy, val_accuracy)
            
        print 'best validation accuracy achieved during cross-validation: %f' % best_val
        
        # Evaluate your trained SVM on the test set
        y_test_pred = best_svm.predict(self.X_test_feats)
        test_accuracy = np.mean(self.y_test == y_test_pred)
        print test_accuracy
#         self.visulize_mistake(y_test_pred)
        return
    def neural_network_classifier(self):
        input_dim = self.X_train_feats.shape[1]
        hidden_dim = 500
        num_classes = 10
        num_iters = 1800
        batch_size=200
        # hyperparameters
        learning_rate = 5e-1
        reg = 1e-6
        learning_rate_decay = 0.95
        
        net = TwoLayerNet(input_dim, hidden_dim, num_classes)
        net.train(self.X_train_feats, self.y_train, self.X_val_feats, self.y_val,
                                      num_iters=num_iters,
                                      batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      learning_rate_decay= learning_rate_decay,
                                      reg=reg,
                                      verbose=False)
        # Predict on the validation set
        val_acc = (net.predict(self.X_val_feats) == self.y_val).mean()
        train_acc = (net.predict(self.X_train_feats) == self.y_train).mean()
        print 'Train accuracy:{}, Validation accuracy:{}'.format(train_acc, val_acc)

        test_acc = (net.predict(self.X_test_feats) == self.y_test).mean()
        print test_acc
        return
    def run(self):
#         self.explore_sift()
#         self.save_sample_images()
        self.load_data()
#         self.svm_classifier()
        self.neural_network_classifier()
        
        return





if __name__ == "__main__":   
    obj= FeaturesModel()
    obj.run()