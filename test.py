<<<<<<< HEAD
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import utils
import os

in_path = './data/valid/new_in'
da_path = './data/valid/new_da'
sum_path = './data/valid/new_sum'
in_vocab = utils.loadVocabulary(os.path.join('./vocab', 'in_vocab'))
da_vocab = utils.loadVocabulary(os.path.join('./vocab', 'da_vocab'))

def restore_model_ckpt(in_data,das_input,seq_len,da_weigh,sum_weight,sum_len,sum_data):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./model_with_diact/summary_only_size_256_epochs_200.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model_with_diact'))

    input_data = sess.graph.get_tensor_by_name('inputs:0')
    das_data = sess.graph.get_tensor_by_name('das_input:0')
    sequence_length = sess.graph.get_tensor_by_name('sequence_length:0')
    da_weights = sess.graph.get_tensor_by_name('da_weights:0')
    sum_weights = sess.graph.get_tensor_by_name('sum_weights:0')
    sum_length = sess.graph.get_tensor_by_name('sum_length:0')

    sum_output = sess.graph.get_tensor_by_name('my_output:0')


    feed_dict = {input_data:in_data,sequence_length:seq_len,da_weights:da_weigh,sum_weights:sum_weight,sum_length:sum_len,das_data:das_input}

    ret = sess.run(tf.nn.softmax(sum_output),feed_dict)

    pred_sums = []
    correct_sums = []
    for batch in ret:
        tmp = []
        for time_i in batch:
            tmp.append(np.argmax(time_i))
        pred_sums.append(tmp)
    for i in sum_data:
        correct_sums.append(i.tolist())
    vocab_dict = utils.get_vocab_dict()
    for pred, corr in zip(pred_sums, correct_sums):
        print('pred:'+utils.idstosentence(pred,vocab_dict))
        print('corr:'+utils.idstosentence(corr,vocab_dict)+'\n')


if __name__ == '__main__':
    data = utils.DataProcessor(in_path,da_path,sum_path,in_vocab,da_vocab)
    for i in range(10):
        in_data, da_data, das_input,pos_input, da_weight, sequence_length, sum_data, sum_weight, sum_length, _, _, _ ,_= data.get_batch(16)
        restore_model_ckpt(in_data,das_input,sequence_length,da_weight,sum_weight,sum_length,sum_data)


=======
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import utils
import os

in_path = './data/valid/new_in'
da_path = './data/valid/new_da'
sum_path = './data/valid/new_sum'
in_vocab = utils.loadVocabulary(os.path.join('./vocab', 'in_vocab'))
da_vocab = utils.loadVocabulary(os.path.join('./vocab', 'da_vocab'))

def restore_model_ckpt(in_data,das_input,seq_len,da_weigh,sum_weight,sum_len,sum_data):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./model_with_diact/summary_only_size_256_epochs_200.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model_with_diact'))

    input_data = sess.graph.get_tensor_by_name('inputs:0')
    das_data = sess.graph.get_tensor_by_name('das_input:0')
    sequence_length = sess.graph.get_tensor_by_name('sequence_length:0')
    da_weights = sess.graph.get_tensor_by_name('da_weights:0')
    sum_weights = sess.graph.get_tensor_by_name('sum_weights:0')
    sum_length = sess.graph.get_tensor_by_name('sum_length:0')

    sum_output = sess.graph.get_tensor_by_name('my_output:0')


    feed_dict = {input_data:in_data,sequence_length:seq_len,da_weights:da_weigh,sum_weights:sum_weight,sum_length:sum_len,das_data:das_input}

    ret = sess.run(tf.nn.softmax(sum_output),feed_dict)

    pred_sums = []
    correct_sums = []
    for batch in ret:
        tmp = []
        for time_i in batch:
            tmp.append(np.argmax(time_i))
        pred_sums.append(tmp)
    for i in sum_data:
        correct_sums.append(i.tolist())
    vocab_dict = utils.get_vocab_dict()
    for pred, corr in zip(pred_sums, correct_sums):
        print('pred:'+utils.idstosentence(pred,vocab_dict))
        print('corr:'+utils.idstosentence(corr,vocab_dict)+'\n')


if __name__ == '__main__':
    data = utils.DataProcessor(in_path,da_path,sum_path,in_vocab,da_vocab)
    for i in range(10):
        in_data, da_data, das_input,pos_input, da_weight, sequence_length, sum_data, sum_weight, sum_length, _, _, _ ,_= data.get_batch(16)
        restore_model_ckpt(in_data,das_input,sequence_length,da_weight,sum_weight,sum_length,sum_data)


>>>>>>> cfe53d38eaadf61745eca7efc701413e8cc538d6
