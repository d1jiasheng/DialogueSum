<<<<<<< HEAD
import os
import argparse
import logging
import sys
import shutil
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

from utils import loadVocabulary
from utils import computeAccuracy
from utils import DataProcessor
import rouge


class MultiDialSum():
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        self.config.gpu_options.allow_growth = True



        # model和vocab的路径
        self.data_path = './data'
        self.model_path = './model_with_diact'
        self.vocab_path = './vocab'
        self.result_path = './result'

        self.num_units = 256  # 循环网络隐藏单元
        self.batch_size = 16  # 每一批放入的数量
        self.max_epochs = 100  # 最大训练次数

        self.layer_size = 256

        self.inference_outputs = None
        # 数据的地址
        self.train_data_path = 'train'
        self.test_data_path = 'test'
        self.valid_data_path = 'valid'
        self.input_file = 'new_in'
        self.da_file = 'new_da'
        self.sum_file = 'new_sum'
        self.log_dir = './MultiDiaSum_LOG'

        # 参数初始化
        self.input_data = tf.placeholder(tf.int32, [None, None, None], name='inputs')
        self.input_das = tf.placeholder(tf.int32, [None, None, None], name = 'das_input')
        self.input_pos = tf.placeholder(tf.int32, [None, None, None], name = 'pos_input')
        self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.das = tf.placeholder(tf.int32, [None, None], name='das')
        self.da_weights = tf.placeholder(tf.float32, [None, None], name='da_weights')
        self.summ = tf.placeholder(tf.int32, [None, None], name='summ')
        self.sum_weights = tf.placeholder(tf.float32, [None, None], name='sum_weights')
        self.sum_length = tf.placeholder(tf.int32, [None], name='sum_length')
        self.in_sen_similar = tf.placeholder(tf.float32, [None,None],name = 'in_sen_similar')

    def init_path_and_voc(self):
        full_train_path = os.path.join(self.data_path, self.train_data_path)
        full_test_path = os.path.join(self.data_path, self.test_data_path)
        full_valid_path = os.path.join(self.data_path, self.valid_data_path)
        vocab_path = self.vocab_path
        in_vocab = loadVocabulary(os.path.join(vocab_path, 'in_vocab'))
        da_vocab = loadVocabulary(os.path.join(vocab_path, 'da_vocab'))
        return full_train_path, full_test_path, full_valid_path, in_vocab, da_vocab

    def train_model(self, input_data, input_size, sequence_length, das_input, pos_input, da_size, decoder_sequence_length, model_type, layer_size=256,
                    isTraining=True):

        w_alpha = 0.01
        h_alpha = 0.1
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(layer_size)
        if isTraining == True:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=0.5,
                                                    output_keep_prob=0.5)

        embedding = tf.get_variable('embedding', [input_size, layer_size])  # 8887*256
        inputs = tf.nn.embedding_lookup(embedding, input_data)  # batch_size*sequence_length*8887*256 ?
        inputs = tf.reduce_sum(inputs, 2)  # batch_size*sequence_length*256

        state_outputs, final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,sequence_length=sequence_length, dtype=tf.float32)

        final_state = final_state.h


        state_shape = state_outputs.get_shape()  # batch_size*max_time*512

        with tf.variable_scope('attention'):

            attn_size = state_shape[2].value  # 512维

            sum_input = final_state
            with tf.variable_scope('sum_attn'):
                BOS_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='BOS') * 2
                BOS_step_embedded = tf.nn.embedding_lookup(embedding, BOS_time_slice)
                pad_step_embedded = tf.zeros([self.batch_size, layer_size], dtype=tf.float32)

                # 自定义一个decoder的结构，原理需理解
                def initial_fn():
                    initial_elements_finished = (0 >= decoder_sequence_length)  # all False at the initial step
                    initial_input = BOS_step_embedded
                    return initial_elements_finished, initial_input

                def sample_fn(time, outputs, state):
                    prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
                    return prediction_id

                def next_inputs_fn(time, outputs, state, sample_ids):
                    # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
                    pred_embedding = tf.nn.embedding_lookup(embedding, sample_ids)
                    next_input = pred_embedding
                    elements_finished = (
                            time >= decoder_sequence_length)  # this operation produces boolean tensor of [batch_size]
                    all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
                    next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
                    next_state = state
                    return elements_finished, next_inputs, next_state

                my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)
                decoder_cell = tf.contrib.rnn.BasicLSTMCell(final_state.get_shape().as_list()[1])
                if isTraining == True:
                    decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=0.5,
                                                                 output_keep_prob=0.5)
                attn_mechanism = tf.contrib.seq2seq.LuongAttention(state_shape[2].value, state_outputs,
                                                                   memory_sequence_length=sequence_length)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attn_mechanism,
                                                                attention_layer_size=state_shape[2].value,
                                                                alignment_history=True, name='sum_attention')
                sum_out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, input_size)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=sum_out_cell, helper=my_helper,
                                                          initial_state=sum_out_cell.zero_state(dtype=tf.float32,
                                                                                                batch_size=self.batch_size))
                decoder_final_outputs, decoder_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, impute_finished=True, maximum_iterations=tf.reduce_max(decoder_sequence_length))



        # da的大小是 16*17  rnn_output的大小是16*sequence_length*8887
        outputs = [decoder_final_outputs.rnn_output, decoder_final_outputs.sample_id]

        return outputs

    def valid_model(self, in_path, da_path, sum_path, sess):
        # return accuracy for dialogue act, rouge-1,2,3,L for summary
        # some useful items are also calculated
        # da_outputs, correct_das: predicted / ground truth of dialogue act
        full_train_path, full_test_path, full_valid_path, in_vocab, da_vocab = self.init_path_and_voc()
        rouge_1 = []
        rouge_2 = []
        rouge_3 = []
        rouge_L = []


        data_processor_valid = DataProcessor(in_path, da_path, sum_path, in_vocab, da_vocab)
        while True:
            # get a batch of data
            in_data, da_data, das_input,pos_input, da_weight, length, sums, sum_weight, sum_lengths, in_seq, da_seq, sum_seq, in_sen_similar = data_processor_valid.get_batch(
                self.batch_size)
            feed_dict = {self.input_data: in_data, self.sequence_length: length, self.sum_length: sum_lengths, self.input_das: das_input,self.input_pos:pos_input}
            if data_processor_valid.end != 1:
                ret = sess.run(self.inference_outputs, feed_dict)

                # summary part
                pred_sums = []
                correct_sums = []
                for batch in ret[0]:
                    tmp = []
                    for time_i in batch:
                        tmp.append(np.argmax(time_i))
                    pred_sums.append(tmp)
                for i in sums:
                    correct_sums.append(i.tolist())
                for pred, corr in zip(pred_sums, correct_sums):
                    # 输出预测的形式
                    # logging.info('pred'+str(pred))
                    # logging.info('corr' + str(corr))
                    rouge_score_map = rouge.rouge(pred, corr)
                    rouge1 = 100 * rouge_score_map['rouge_1/f_score']
                    rouge2 = 100 * rouge_score_map['rouge_2/f_score']
                    rouge3 = 100 * rouge_score_map['rouge_3/f_score']
                    rougeL = 100 * rouge_score_map['rouge_l/f_score']
                    rouge_1.append(rouge1)
                    rouge_2.append(rouge2)
                    rouge_3.append(rouge3)
                    rouge_L.append(rougeL)


            if data_processor_valid.end == 1:
                break

        logging.info('sum rouge1: ' + str(np.mean(rouge_1)))
        logging.info('sum rouge2: ' + str(np.mean(rouge_2)))
        logging.info('sum rouge3: ' + str(np.mean(rouge_3)))
        logging.info('sum rougeL: ' + str(np.mean(rouge_L)))

        data_processor_valid.close()
        return np.mean(rouge_1), np.mean(rouge_2), np.mean(rouge_3), np.mean(rouge_L)

    def train(self, grade_clip=True, model_type = 'encoder_with_dialogue_act'):
        full_train_path, full_test_path, full_valid_path, in_vocab, da_vocab = self.init_path_and_voc()
        data_processor = DataProcessor(os.path.join(full_train_path, self.input_file),
                                       os.path.join(full_train_path, self.da_file),
                                       os.path.join(full_train_path, self.sum_file), in_vocab, da_vocab)
        with tf.variable_scope('training_model'):
            training_outputs = self.train_model(self.input_data, len(in_vocab['vocab']), self.sequence_length, self.input_das,self.input_pos,
                                                len(da_vocab['vocab']), self.sum_length, model_type, self.layer_size)


        # 这个是用来看输出的，主要是给这个输出节点命名，不知道有没有其他方法
        sum_output = tf.multiply(1.0, training_outputs[0], name='my_output')
        with tf.variable_scope('sum_loss'):
            sum_loss = tf.contrib.seq2seq.sequence_loss(logits=sum_output, targets=self.summ, weights=self.sum_weights,
                                                        average_across_timesteps=False)

        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(learning_rate=0.0001)

        sum_params = []

        for p in params:
            if not 'da_' in p.name:
                sum_params.append(p)

        gradients_sum = tf.gradients(sum_loss, sum_params)
        clipped_gradients_sum, norm_sum = tf.clip_by_global_norm(gradients_sum, 5.0)
        gradient_norm_sum = norm_sum
        update_sum = opt.apply_gradients(zip(clipped_gradients_sum, sum_params), global_step=self.global_step)

        # if grade_clip == False:
        #     update_da = opt.minimize(da_loss)
        #     update_sum = opt.minimize(sum_loss)

        training_outputs = [self.global_step, sum_loss, update_sum,gradient_norm_sum]
        inputs = [self.input_data, self.sequence_length, self.das, self.da_weights, self.summ, self.sum_weights,
                  self.sum_length]

        tf.summary.scalar('sum_loss', tf.reduce_mean(sum_loss))
        # Create Inference Model
        with tf.variable_scope('training_model',reuse=True):
            self.inference_outputs = self.train_model(self.input_data, len(in_vocab['vocab']), self.sequence_length, self.input_das,self.input_pos,
                                                      len(da_vocab['vocab']),self.sum_length, model_type, self.layer_size,
                                                      isTraining=False)

        inference_sum_output = tf.nn.softmax(self.inference_outputs[0], name='sum_output')

        self.inference_outputs = [inference_sum_output]
        inference_inputs = [self.input_data, self.sequence_length, self.sum_length]

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        saver = tf.train.Saver()
        best_sent_saver = tf.train.Saver()

        merged = tf.summary.merge_all()
        # 开始训练
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.log_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(self.log_dir + '/test')
            sess.run(tf.global_variables_initializer())

            data_processor = None
            epochs = 0
            step = 0
            sum_loss = 0.0
            num_loss = 0
            no_improve = 0

            v_r1 = -1
            v_r2 = -1
            v_r3 = -1
            v_rL = -1
            t_r1 = -1
            t_r2 = -1
            t_r3 = -1
            t_rL = -1

            logging.info('Training Start')

            while True:
                if data_processor == None:
                    data_processor = DataProcessor(os.path.join(full_train_path, self.input_file),
                                                   os.path.join(full_train_path, self.da_file),
                                                   os.path.join(full_train_path, self.sum_file), in_vocab, da_vocab)
                in_data, da_data, das_input, pos_input, da_weight, length, sums, sum_weight, sum_lengths, _, _, _, in_sen_similar = data_processor.get_batch(
                    self.batch_size)

                feed_dict = {self.input_data.name: in_data, self.das.name: da_data, self.input_das.name:das_input, self.input_pos:pos_input, self.da_weights.name: da_weight,
                             self.sequence_length.name: length, self.summ.name: sums, self.sum_weights.name: sum_weight,
                             self.sum_length.name: sum_lengths}
                if data_processor.end != 1:
                    # in case training data can be divided by batch_size,
                    # which will produce an "empty" batch that has no data with data_processor.end==1
                    ret, summary = sess.run([training_outputs, merged], feed_dict)
                    sum_loss += np.mean(ret[1])
                    step = ret[0]
                    num_loss += 1
                    train_writer.add_summary(summary)

                if data_processor.end == 1:
                    data_processor.close()
                    data_processor = None
                    epochs += 1
                    logging.info('Step: ' + str(step))
                    logging.info('Epochs: ' + str(epochs))
                    logging.info('Int. Loss: ' + str(sum_loss / num_loss))
                    num_loss = 0
                    sum_loss = 0.0

                    save_path = os.path.join(self.model_path, 'summary_only')
                    save_path += '_size_' + str(self.layer_size) + '_epochs_' + str(epochs) + '.ckpt'
                    saver.save(sess, save_path)

                    logging.info('Valid:')
                    # variable starts wih e stands for current epoch
                    e_v_r1, e_v_r2, e_v_r3, e_v_rL= self.valid_model(
                        os.path.join(full_valid_path, self.input_file), os.path.join(full_valid_path, self.da_file),
                        os.path.join(full_valid_path, self.sum_file), sess)
                    logging.info('Test:')
                    e_t_r1, e_t_r2, e_t_r3, e_t_rL= self.valid_model(
                        os.path.join(full_test_path, self.input_file), os.path.join(full_test_path, self.da_file),
                        os.path.join(full_test_path, self.sum_file), sess)

                    if e_v_r2 <= v_r2:
                        no_improve += 1
                    else:
                        no_improve = 0

                    if e_v_r2 > v_r2:
                        v_r2 = e_v_r2
                    if e_v_r1 > v_r1:
                        v_r1 = e_v_r1
                    if e_v_r3 > v_r3:
                        v_r3 = e_v_r3
                    if e_v_rL > v_rL:
                        v_rL = e_v_rL
                    if e_t_r1 > t_r1:
                        t_r1 = e_t_r1
                    if e_t_r2 > t_r2:
                        t_r2 = e_t_r2
                    if e_t_r3 > t_r3:
                        t_r3 = e_t_r3
                    if e_t_rL > t_rL:
                        t_rL = e_t_rL

                        # save best model
                        save_path = os.path.join(self.model_path,
                                                 'best_sent_' + str(self.layer_size) + '/') + 'epochs_' + str(
                            epochs) + '.ckpt'
                        best_sent_saver.save(sess, save_path)

                    if epochs == self.max_epochs:
                        break

                    if t_r2 == -1 or v_r2 == -1:
                        print('something in validation or testing goes wrong! did not update error.')
                        exit(1)
            train_writer.close()
            test_writer.close()
        header = self.result_path

        with open(os.path.join(header, 'valid_r1_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(v_r1) + '\n')
        with open(os.path.join(header, 'test_r1_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(t_r1) + '\n')
        with open(os.path.join(header, 'valid_r2_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(v_r2) + '\n')
        with open(os.path.join(header, 'test_r2_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(t_r2) + '\n')
        with open(os.path.join(header, 'valid_r3_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(v_r3) + '\n')
        with open(os.path.join(header, 'test_r3_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(t_r3) + '\n')
        with open(os.path.join(header, 'valid_rL_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(v_rL) + '\n')
        with open(os.path.join(header, 'test_rL_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(t_rL) + '\n')

        print('*' * 20 + 'summary_only' + ' ' + str(self.layer_size) + '*' * 20)


if __name__ == '__main__':
    md = MultiDialSum()
    md.train(model_type='encoder_with_dialogue_act')
=======
import os
import argparse
import logging
import sys
import shutil
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

from utils import loadVocabulary
from utils import computeAccuracy
from utils import DataProcessor
import rouge


class MultiDialSum():
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        self.config.gpu_options.allow_growth = True



        # model和vocab的路径
        self.data_path = './data'
        self.model_path = './model_with_diact'
        self.vocab_path = './vocab'
        self.result_path = './result'

        self.num_units = 256  # 循环网络隐藏单元
        self.batch_size = 16  # 每一批放入的数量
        self.max_epochs = 100  # 最大训练次数

        self.layer_size = 256

        self.inference_outputs = None
        # 数据的地址
        self.train_data_path = 'train'
        self.test_data_path = 'test'
        self.valid_data_path = 'valid'
        self.input_file = 'new_in'
        self.da_file = 'new_da'
        self.sum_file = 'new_sum'
        self.log_dir = './MultiDiaSum_LOG'

        # 参数初始化
        self.input_data = tf.placeholder(tf.int32, [None, None, None], name='inputs')
        self.input_das = tf.placeholder(tf.int32, [None, None, None], name = 'das_input')
        self.input_pos = tf.placeholder(tf.int32, [None, None, None], name = 'pos_input')
        self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.das = tf.placeholder(tf.int32, [None, None], name='das')
        self.da_weights = tf.placeholder(tf.float32, [None, None], name='da_weights')
        self.summ = tf.placeholder(tf.int32, [None, None], name='summ')
        self.sum_weights = tf.placeholder(tf.float32, [None, None], name='sum_weights')
        self.sum_length = tf.placeholder(tf.int32, [None], name='sum_length')
        self.in_sen_similar = tf.placeholder(tf.float32, [None,None],name = 'in_sen_similar')

    def init_path_and_voc(self):
        full_train_path = os.path.join(self.data_path, self.train_data_path)
        full_test_path = os.path.join(self.data_path, self.test_data_path)
        full_valid_path = os.path.join(self.data_path, self.valid_data_path)
        vocab_path = self.vocab_path
        in_vocab = loadVocabulary(os.path.join(vocab_path, 'in_vocab'))
        da_vocab = loadVocabulary(os.path.join(vocab_path, 'da_vocab'))
        return full_train_path, full_test_path, full_valid_path, in_vocab, da_vocab

    def train_model(self, input_data, input_size, sequence_length, das_input, pos_input, da_size, decoder_sequence_length, model_type, layer_size=256,
                    isTraining=True):

        w_alpha = 0.01
        h_alpha = 0.1
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(layer_size)
        if isTraining == True:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=0.5,
                                                    output_keep_prob=0.5)

        embedding = tf.get_variable('embedding', [input_size, layer_size])  # 8887*256
        inputs = tf.nn.embedding_lookup(embedding, input_data)  # batch_size*sequence_length*8887*256 ?
        inputs = tf.reduce_sum(inputs, 2)  # batch_size*sequence_length*256

        state_outputs, final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,sequence_length=sequence_length, dtype=tf.float32)

        final_state = final_state.h


        state_shape = state_outputs.get_shape()  # batch_size*max_time*512

        with tf.variable_scope('attention'):

            attn_size = state_shape[2].value  # 512维

            sum_input = final_state
            with tf.variable_scope('sum_attn'):
                BOS_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='BOS') * 2
                BOS_step_embedded = tf.nn.embedding_lookup(embedding, BOS_time_slice)
                pad_step_embedded = tf.zeros([self.batch_size, layer_size], dtype=tf.float32)

                # 自定义一个decoder的结构，原理需理解
                def initial_fn():
                    initial_elements_finished = (0 >= decoder_sequence_length)  # all False at the initial step
                    initial_input = BOS_step_embedded
                    return initial_elements_finished, initial_input

                def sample_fn(time, outputs, state):
                    prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
                    return prediction_id

                def next_inputs_fn(time, outputs, state, sample_ids):
                    # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
                    pred_embedding = tf.nn.embedding_lookup(embedding, sample_ids)
                    next_input = pred_embedding
                    elements_finished = (
                            time >= decoder_sequence_length)  # this operation produces boolean tensor of [batch_size]
                    all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
                    next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
                    next_state = state
                    return elements_finished, next_inputs, next_state

                my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)
                decoder_cell = tf.contrib.rnn.BasicLSTMCell(final_state.get_shape().as_list()[1])
                if isTraining == True:
                    decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=0.5,
                                                                 output_keep_prob=0.5)
                attn_mechanism = tf.contrib.seq2seq.LuongAttention(state_shape[2].value, state_outputs,
                                                                   memory_sequence_length=sequence_length)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attn_mechanism,
                                                                attention_layer_size=state_shape[2].value,
                                                                alignment_history=True, name='sum_attention')
                sum_out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, input_size)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=sum_out_cell, helper=my_helper,
                                                          initial_state=sum_out_cell.zero_state(dtype=tf.float32,
                                                                                                batch_size=self.batch_size))
                decoder_final_outputs, decoder_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, impute_finished=True, maximum_iterations=tf.reduce_max(decoder_sequence_length))



        # da的大小是 16*17  rnn_output的大小是16*sequence_length*8887
        outputs = [decoder_final_outputs.rnn_output, decoder_final_outputs.sample_id]

        return outputs

    def valid_model(self, in_path, da_path, sum_path, sess):
        # return accuracy for dialogue act, rouge-1,2,3,L for summary
        # some useful items are also calculated
        # da_outputs, correct_das: predicted / ground truth of dialogue act
        full_train_path, full_test_path, full_valid_path, in_vocab, da_vocab = self.init_path_and_voc()
        rouge_1 = []
        rouge_2 = []
        rouge_3 = []
        rouge_L = []


        data_processor_valid = DataProcessor(in_path, da_path, sum_path, in_vocab, da_vocab)
        while True:
            # get a batch of data
            in_data, da_data, das_input,pos_input, da_weight, length, sums, sum_weight, sum_lengths, in_seq, da_seq, sum_seq, in_sen_similar = data_processor_valid.get_batch(
                self.batch_size)
            feed_dict = {self.input_data: in_data, self.sequence_length: length, self.sum_length: sum_lengths, self.input_das: das_input,self.input_pos:pos_input}
            if data_processor_valid.end != 1:
                ret = sess.run(self.inference_outputs, feed_dict)

                # summary part
                pred_sums = []
                correct_sums = []
                for batch in ret[0]:
                    tmp = []
                    for time_i in batch:
                        tmp.append(np.argmax(time_i))
                    pred_sums.append(tmp)
                for i in sums:
                    correct_sums.append(i.tolist())
                for pred, corr in zip(pred_sums, correct_sums):
                    # 输出预测的形式
                    # logging.info('pred'+str(pred))
                    # logging.info('corr' + str(corr))
                    rouge_score_map = rouge.rouge(pred, corr)
                    rouge1 = 100 * rouge_score_map['rouge_1/f_score']
                    rouge2 = 100 * rouge_score_map['rouge_2/f_score']
                    rouge3 = 100 * rouge_score_map['rouge_3/f_score']
                    rougeL = 100 * rouge_score_map['rouge_l/f_score']
                    rouge_1.append(rouge1)
                    rouge_2.append(rouge2)
                    rouge_3.append(rouge3)
                    rouge_L.append(rougeL)


            if data_processor_valid.end == 1:
                break

        logging.info('sum rouge1: ' + str(np.mean(rouge_1)))
        logging.info('sum rouge2: ' + str(np.mean(rouge_2)))
        logging.info('sum rouge3: ' + str(np.mean(rouge_3)))
        logging.info('sum rougeL: ' + str(np.mean(rouge_L)))

        data_processor_valid.close()
        return np.mean(rouge_1), np.mean(rouge_2), np.mean(rouge_3), np.mean(rouge_L)

    def train(self, grade_clip=True, model_type = 'encoder_with_dialogue_act'):
        full_train_path, full_test_path, full_valid_path, in_vocab, da_vocab = self.init_path_and_voc()
        data_processor = DataProcessor(os.path.join(full_train_path, self.input_file),
                                       os.path.join(full_train_path, self.da_file),
                                       os.path.join(full_train_path, self.sum_file), in_vocab, da_vocab)
        with tf.variable_scope('training_model'):
            training_outputs = self.train_model(self.input_data, len(in_vocab['vocab']), self.sequence_length, self.input_das,self.input_pos,
                                                len(da_vocab['vocab']), self.sum_length, model_type, self.layer_size)


        # 这个是用来看输出的，主要是给这个输出节点命名，不知道有没有其他方法
        sum_output = tf.multiply(1.0, training_outputs[0], name='my_output')
        with tf.variable_scope('sum_loss'):
            sum_loss = tf.contrib.seq2seq.sequence_loss(logits=sum_output, targets=self.summ, weights=self.sum_weights,
                                                        average_across_timesteps=False)

        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(learning_rate=0.0001)

        sum_params = []

        for p in params:
            if not 'da_' in p.name:
                sum_params.append(p)

        gradients_sum = tf.gradients(sum_loss, sum_params)
        clipped_gradients_sum, norm_sum = tf.clip_by_global_norm(gradients_sum, 5.0)
        gradient_norm_sum = norm_sum
        update_sum = opt.apply_gradients(zip(clipped_gradients_sum, sum_params), global_step=self.global_step)

        # if grade_clip == False:
        #     update_da = opt.minimize(da_loss)
        #     update_sum = opt.minimize(sum_loss)

        training_outputs = [self.global_step, sum_loss, update_sum,gradient_norm_sum]
        inputs = [self.input_data, self.sequence_length, self.das, self.da_weights, self.summ, self.sum_weights,
                  self.sum_length]

        tf.summary.scalar('sum_loss', tf.reduce_mean(sum_loss))
        # Create Inference Model
        with tf.variable_scope('training_model',reuse=True):
            self.inference_outputs = self.train_model(self.input_data, len(in_vocab['vocab']), self.sequence_length, self.input_das,self.input_pos,
                                                      len(da_vocab['vocab']),self.sum_length, model_type, self.layer_size,
                                                      isTraining=False)

        inference_sum_output = tf.nn.softmax(self.inference_outputs[0], name='sum_output')

        self.inference_outputs = [inference_sum_output]
        inference_inputs = [self.input_data, self.sequence_length, self.sum_length]

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        saver = tf.train.Saver()
        best_sent_saver = tf.train.Saver()

        merged = tf.summary.merge_all()
        # 开始训练
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.log_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(self.log_dir + '/test')
            sess.run(tf.global_variables_initializer())

            data_processor = None
            epochs = 0
            step = 0
            sum_loss = 0.0
            num_loss = 0
            no_improve = 0

            v_r1 = -1
            v_r2 = -1
            v_r3 = -1
            v_rL = -1
            t_r1 = -1
            t_r2 = -1
            t_r3 = -1
            t_rL = -1

            logging.info('Training Start')

            while True:
                if data_processor == None:
                    data_processor = DataProcessor(os.path.join(full_train_path, self.input_file),
                                                   os.path.join(full_train_path, self.da_file),
                                                   os.path.join(full_train_path, self.sum_file), in_vocab, da_vocab)
                in_data, da_data, das_input, pos_input, da_weight, length, sums, sum_weight, sum_lengths, _, _, _, in_sen_similar = data_processor.get_batch(
                    self.batch_size)

                feed_dict = {self.input_data.name: in_data, self.das.name: da_data, self.input_das.name:das_input, self.input_pos:pos_input, self.da_weights.name: da_weight,
                             self.sequence_length.name: length, self.summ.name: sums, self.sum_weights.name: sum_weight,
                             self.sum_length.name: sum_lengths}
                if data_processor.end != 1:
                    # in case training data can be divided by batch_size,
                    # which will produce an "empty" batch that has no data with data_processor.end==1
                    ret, summary = sess.run([training_outputs, merged], feed_dict)
                    sum_loss += np.mean(ret[1])
                    step = ret[0]
                    num_loss += 1
                    train_writer.add_summary(summary)

                if data_processor.end == 1:
                    data_processor.close()
                    data_processor = None
                    epochs += 1
                    logging.info('Step: ' + str(step))
                    logging.info('Epochs: ' + str(epochs))
                    logging.info('Int. Loss: ' + str(sum_loss / num_loss))
                    num_loss = 0
                    sum_loss = 0.0

                    save_path = os.path.join(self.model_path, 'summary_only')
                    save_path += '_size_' + str(self.layer_size) + '_epochs_' + str(epochs) + '.ckpt'
                    saver.save(sess, save_path)

                    logging.info('Valid:')
                    # variable starts wih e stands for current epoch
                    e_v_r1, e_v_r2, e_v_r3, e_v_rL= self.valid_model(
                        os.path.join(full_valid_path, self.input_file), os.path.join(full_valid_path, self.da_file),
                        os.path.join(full_valid_path, self.sum_file), sess)
                    logging.info('Test:')
                    e_t_r1, e_t_r2, e_t_r3, e_t_rL= self.valid_model(
                        os.path.join(full_test_path, self.input_file), os.path.join(full_test_path, self.da_file),
                        os.path.join(full_test_path, self.sum_file), sess)

                    if e_v_r2 <= v_r2:
                        no_improve += 1
                    else:
                        no_improve = 0

                    if e_v_r2 > v_r2:
                        v_r2 = e_v_r2
                    if e_v_r1 > v_r1:
                        v_r1 = e_v_r1
                    if e_v_r3 > v_r3:
                        v_r3 = e_v_r3
                    if e_v_rL > v_rL:
                        v_rL = e_v_rL
                    if e_t_r1 > t_r1:
                        t_r1 = e_t_r1
                    if e_t_r2 > t_r2:
                        t_r2 = e_t_r2
                    if e_t_r3 > t_r3:
                        t_r3 = e_t_r3
                    if e_t_rL > t_rL:
                        t_rL = e_t_rL

                        # save best model
                        save_path = os.path.join(self.model_path,
                                                 'best_sent_' + str(self.layer_size) + '/') + 'epochs_' + str(
                            epochs) + '.ckpt'
                        best_sent_saver.save(sess, save_path)

                    if epochs == self.max_epochs:
                        break

                    if t_r2 == -1 or v_r2 == -1:
                        print('something in validation or testing goes wrong! did not update error.')
                        exit(1)
            train_writer.close()
            test_writer.close()
        header = self.result_path

        with open(os.path.join(header, 'valid_r1_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(v_r1) + '\n')
        with open(os.path.join(header, 'test_r1_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(t_r1) + '\n')
        with open(os.path.join(header, 'valid_r2_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(v_r2) + '\n')
        with open(os.path.join(header, 'test_r2_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(t_r2) + '\n')
        with open(os.path.join(header, 'valid_r3_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(v_r3) + '\n')
        with open(os.path.join(header, 'test_r3_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(t_r3) + '\n')
        with open(os.path.join(header, 'valid_rL_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(v_rL) + '\n')
        with open(os.path.join(header, 'test_rL_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(t_rL) + '\n')

        print('*' * 20 + 'summary_only' + ' ' + str(self.layer_size) + '*' * 20)


if __name__ == '__main__':
    md = MultiDialSum()
    md.train(model_type='encoder_with_dialogue_act')
>>>>>>> cfe53d38eaadf61745eca7efc701413e8cc538d6
