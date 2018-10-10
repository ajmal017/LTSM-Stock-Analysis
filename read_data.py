import numpy as np
import backtrader as bt
import ib
import datetime
import os.path 
import sys
import numpy as np
import tensorflow as tf
import random
from tensorflow.contrib import rnn

# Parameters
learning_rate = 0.001
training_steps = 10000
display_step = 20
num_hidden = 128
num_input = 5
timesteps = 125
batch_size = 32
num_classes = 2

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def load_data(csv_file):
    current_data = []
    file = open(csv_file)
    for line in file:
        line_stats = []
        split_line = line.split(";", 7)
        for i in range(2, 7):
            line_stats.append(float(split_line[i].strip("\n")))
        current_data.append(line_stats)
    return current_data 

def run(args=None):
    current_data = load_data("historical_data/aaba-1m.csv")
    logits = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)

	# Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

	# Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

	# Start training
    with tf.Session() as sess:
	    # Run the initializer
        sess.run(init)

        for step in range(0, training_steps):
            batch_x = np.zeros((batch_size, timesteps, num_input), dtype=float)
            batch_y = np.zeros((batch_size, num_classes), dtype=int)
            for batch in range(0, batch_size):
                random_position = random.randint(0, len(current_data)-timesteps-2)
                for i in range(random_position, random_position+timesteps):
                    batch_x[batch][i-random_position] = current_data[i]
                    if i == random_position+timesteps-1:
                        if batch_x[batch][i-random_position][3] < current_data[i+1][3]:
                            batch_y[batch][1] = 1
                        else:
                            batch_y[batch][0] = 1
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            train, pred = sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

if __name__ == '__main__':
    run()