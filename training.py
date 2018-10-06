import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.learn.python.learn.datasets import base

print tf.__version__

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

training_set = base.load_csv_with_header(filename = IRIS_TRAINING, features_dtype = np.float32, target_dtype = np.int)
test_set = base.load_csv_with_header(filename = IRIS_TEST, features_dtype = np.float32, target_dtype = np.int)

print training_set.data
print training_set.target

feature_name = 'flower-features'
feature_columns = [tf.feature_column.numeric_column(feature_name, shape = [4])]
print feature_columns

classifier = tf.estimator.LinearClassifier(
    feature_columns = feature_columns,
    n_classes = 3,
    model_dir = "tmp/iris_model" )

def input_fn(dataset):
    def _fn():
        features = {feature_name: tf.constant(dataset.data)}
        labels = tf.constant(dataset.target)
        return features, labels
    return _fn

print input_fn(training_set)()

classifier.train(input_fn = input_fn(training_set), steps = 1000)
print 'fit done.'

accuracy_score = classifier.evaluate(input_fn = input_fn(test_set), steps = 100)["accuracy"]
print '\nAccuracy: {0:f}'.format(accuracy_score)