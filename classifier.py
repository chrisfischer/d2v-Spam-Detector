import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silence warnings

log_dir = '/tmp/tensorflow/spam/logs/spam_with_summaries'

'''
Based off:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py
'''


class Batchifer(object):
    # take in numpy array
    def __init__(self, xs, ys, batch_size):
        self.xs = xs
        self.ys = ys
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, self.ys.size, self.batch_size):
            yield (self.xs[i:i+self.batch_size, :],
                   self.ys[i:i+self.batch_size])

    def get_batch_count(self):
        return int(len(self.xs) / self.batch_size)


def train_classifier(xs_train, ys_train, xs_test, ys_test):
    # Parameters
    learning_rate = 0.003
    training_epochs = 600
    batch_size = 64
    display_step = 1
    test_display_step = 20

    batchifer_train = Batchifer(xs_train, ys_train, batch_size)
    total_batch_count = batchifer_train.get_batch_count()

    # Network Parameters
    n_hidden_1 = 256    # 1st layer number of features
    n_hidden_2 = 128    # 2nd layer number of features
    # n_hidden_3 = 64     # 2nd layer number of features
    n_input = 302
    n_classes = 1

    # get session
    sess = tf.InteractiveSession()

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 1])

    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        with tf.name_scope('fully_connected1'):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        with tf.name_scope('fully_connected2'):
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
        # Hidden layer with RELU activation
        '''
        with tf.name_scope('fully_connected3'):
            layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
            layer_3 = tf.nn.relu(layer_3)
        '''
        # Output layer with linear activation
        with tf.name_scope('fully_connected4'):
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        # 'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        # 'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.squared_difference(y, pred))
    tf.summary.scalar('cost', cost)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.round(y), tf.round(pred))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Initializing the variables
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    tf.global_variables_initializer().run()

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        run_metadata = tf.RunMetadata()
        # Loop over all batches
        for batch_x, batch_y in batchifer_train:

            # Run optimization op (backprop) and cost op (to get loss value)
            summary, c = sess.run([optimizer, cost],
                                  feed_dict={x: batch_x, y: batch_y},
                                  run_metadata=run_metadata)

            # Compute average loss
            avg_cost += c

        # Display logs per epoch step
        if epoch % display_step == 0:
            print 'Epoch: %3d' % (epoch+1), 'cost=%.9f' % (avg_cost / total_batch_count)

            # For TensorBoard
            train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
            train_writer.add_summary(summary, epoch)

        # Display testing accuracy in TensorBoard
        if epoch % test_display_step == 0:
            summary_test, acc_test = sess.run([merged, accuracy],
                                              feed_dict={
                                                x: xs_test,
                                                y: ys_test
                                              })
            test_writer.add_summary(summary, epoch)
            summary_train, acc_train = sess.run([merged, accuracy],
                                                feed_dict={
                                                    x: xs_train,
                                                    y: ys_train
                                                })
            train_writer.add_summary(summary_train, epoch)
            test_writer.add_summary(summary_test, epoch)
            print '\tVal accuracy at epoch %s: %s' % (epoch, acc_test)
    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.round(pred), tf.round(y))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    val = accuracy.eval({x: xs_test,
                         y: ys_test})

    print "Accuracy:", val

    train_writer.close()
    test_writer.close()
