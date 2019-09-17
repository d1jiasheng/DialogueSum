import tensorflow as tf
import numpy as np
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32,shape = (32,100))
h0 = lstm_cell.zero_state(32, np.float32)
output, h1 = lstm_cell.__call__(inputs, h0)
print(h1.h)