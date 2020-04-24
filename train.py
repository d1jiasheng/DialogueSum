import os
import argparse
import logging
import sys
import shutil
import tensorflow as tf

import numpy as np

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
        self.max_epochs = 200  # 最大训练次数

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

    # def atte_detail_train_model(self, input_data, input_size, sequence_length, das_input, da_size, decoder_sequence_length, model_type, max_length, layer_size=256,
    #                             isTraining=True):
    #
    #     if model_type == 'encoder_with_dialogue_act':
    #         input_data = tf.concat([input_data, das_input], axis=2)
    #         input_size = input_size + da_size
    #     cell_fw = tf.contrib.rnn.BasicLSTMCell(layer_size)
    #     cell_bw = tf.contrib.rnn.BasicLSTMCell(layer_size)
    #     if isTraining == True:
    #         cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5,
    #                                                 output_keep_prob=0.5)
    #         cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5,
    #                                                 output_keep_prob=0.5)
    #     embedding = tf.get_variable('embedding', [input_size, layer_size])  # 8887*256
    #     inputs = tf.nn.embedding_lookup(embedding, input_data)  # batch_size*sequence_length*8887*256 ?
    #     inputs = tf.reduce_sum(inputs, 2)  # batch_size*sequence_length*256
    #
    #     my_sequence_length = []
    #     for i in range(16):
    #         my_sequence_length.append(15)
    #     #encoder
    #     state_outputs,final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,sequence_length = my_sequence_length,dtype=tf.float32)
    #     final_state = tf.concat([final_state[0].h, final_state[1].h], 1)
    #     state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)
    #     state_shape = state_outputs.get_shape()
    #
    #     da_inputs = state_outputs
    #     attn_size = state_shape[2].value
    #     da_inputs = tf.reshape(da_inputs,[-1,attn_size])
    #
    #     length = sequence_length[tf.arg_max(sequence_length,0)]
    #     w_alpha = 0.01
    #     h_alpha = 0.1
    #     w1 = tf.Variable(w_alpha * tf.random_normal([512,512]),name = "W1")
    #     h1 = tf.Variable(h_alpha * tf.random_normal([512]),name = "H2")
    #
    #     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=512)
    #
    #     h0 = lstm_cell.zero_state(16,np.float32)
    #
    #     sum_input = final_state
    #     sum_outputs = ''
    #     formar_outputs = ''
    #
    #     w_attn = tf.Variable(w_alpha * tf.random_normal([7680, 512]), name="W_attn")
    #     h_attn = tf.Variable(h_alpha * tf.random_normal([512]), name="H_attn")
    #     for i in range(15):
    #         final_state = tf.reshape(final_state,[16,512])
    #         final_state = tf.add(tf.matmul(final_state,w1),h1)
    #         final_state = tf.expand_dims(final_state,axis=-1)
    #
    #         atten = tf.reshape(tf.matmul(state_outputs,final_state),[self.batch_size,-1])
    #         atten_dist = tf.nn.softmax(atten)
    #         atten_dist = tf.expand_dims(atten_dist,axis=-1)
    #
    #         atten_state_outputs = tf.multiply(atten_dist,state_outputs)
    #
    #         atten_state_outputs = tf.reshape(atten_state_outputs,[16,7680])
    #         atten_state_outputs = tf.add(tf.matmul(atten_state_outputs,w_attn),h_attn)
    #
    #         atten_output = tf.reduce_sum(atten_state_outputs, axis=1)
    #
    #
    #
    #         outputs, states = lstm_cell.__call__(atten_state_outputs,h0)
    #
    #         if i ==0:
    #             sum_outputs = outputs
    #             formar_outputs = outputs
    #         if i>0 :
    #             sum_outputs = tf.concat([sum_outputs,tf.add(tf.multiply(0.5,outputs),tf.multiply(0.5,formar_outputs))],axis=1)
    #             formar_outputs = outputs
    #         final_state = states.h
    #
    #     attn = tf.reshape(sum_outputs,[16, -1, 512])
    #     attn = tf.reduce_mean(attn, axis=2)
    #     attn = tf.expand_dims(attn,-1)
    #     d = tf.reduce_sum(attn * state_outputs,axis=1)
    #     sum_output = tf.concat([d, sum_input], 1)
    #     with tf.variable_scope('sentence_gated'):
    #
    #         sum_gate = core_rnn_cell._linear(sum_output, attn_size, True)
    #         sum_gate = tf.reshape(sum_gate, [-1, 1, sum_gate.get_shape()[1].value])
    #         v1 = tf.get_variable('gateV', [attn_size])
    #
    #         sentence_gate = v1 * tf.tanh(state_outputs + sum_gate)  # batch_size*?*512
    #         gate_value = tf.reduce_sum(sentence_gate, [2], name='gate_value')
    #         sentence_gate = tf.expand_dims(gate_value, -1)
    #         sentence_gate = state_outputs * sentence_gate
    #         sentence_gate = tf.reshape(sentence_gate, [-1, attn_size])
    #         da_output = tf.concat([sentence_gate, da_inputs], 1)
    #     with tf.variable_scope('da_proj'):
    #         da = core_rnn_cell._linear(da_output, da_size, True)
    #
    #     sum_outputs = tf.reshape(sum_outputs,[16,-1,512])
    #
    #     w2 = tf.Variable(w_alpha * tf.random_normal([16, 512, 8887]), name="W2")
    #     h2 = tf.Variable(h_alpha * tf.random_normal([15,8887]), name="H2")
    #     sum_outputs = tf.add(tf.matmul(sum_outputs, w2), h2)
    #     outputs = [da, sum_outputs]
    #
    #     return outputs

    def train_model(self, input_data, input_size, sequence_length, das_input, pos_input, da_size, decoder_sequence_length, model_type, layer_size=256,
                    isTraining=True):

        #计算dialogue_act权重
        input_data_one = tf.cast(input_data,dtype=tf.float32)
        das_input = tf.cast(das_input,dtype=tf.float32)

        w_alpha = 0.01
        h_alpha = 0.1
        w_das = tf.Variable(w_alpha * tf.random_normal([16, 256, 17]), name="w_das")
        h_das = tf.Variable(h_alpha * tf.random_normal([256]), name="h_das")

        das_trans_input_ = tf.reshape(das_input,[16,17,-1])
        das_trans_input_ = tf.add(tf.transpose(tf.matmul(w_das,das_trans_input_),[0,2,1]),h_das)
        das_trans_input_t = tf.transpose(das_trans_input_,[0,2,1])

        das_attn = tf.nn.softmax(tf.reduce_sum(tf.matmul(das_trans_input_,das_trans_input_t),axis=-1))
        das_attn = tf.expand_dims(das_attn,axis=-1)

        das_input = das_attn * das_input

        if model_type == 'encoder_with_dialogue_act':
            input_data_one = tf.concat([input_data_one,das_input],axis=2)
            input_size = input_size + da_size
        input_data_one = tf.cast(input_data_one, dtype=tf.int32)
        # input_data = tf.cast(input_data,dtype=tf.int32)

        # input_data_1 = tf.reshape(input_data,[self.batch_size, -1])
        #
        # input_data_1 = core_rnn_cell._linear(input_data_1,256,True)
        # with tf.variable_scope("input_data_attn"):
        #
        #     input_data_t = tf.transpose(input_data_1,[1,0])
        #     input_data_attn = tf.nn.softmax(tf.reduce_sum(tf.matmul(input_data_1,input_data_t),axis=1))
        #     input_data_attn = tf.expand_dims(input_data_attn,axis=-1)
        #     input_data_attn = core_rnn_cell._linear(input_data_attn,8904,True)
        #
        # input_data = tf.cast(tf.reshape(input_data_attn,[self.batch_size,-1,8904]),dtype=tf.int32)



        cell_fw = tf.contrib.rnn.BasicLSTMCell(layer_size)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(layer_size)
        if isTraining == True:
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5,
                                                    output_keep_prob=0.5)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5,
                                                    output_keep_prob=0.5)
        embedding = tf.get_variable('embedding', [input_size, layer_size])  # 8887*256

        inputs = tf.nn.embedding_lookup(embedding, input_data_one)  # batch_size*sequence_length*8887*256 ?
        inputs = tf.reduce_sum(inputs, 2)  # batch_size*sequence_length*256


        with tf.variable_scope('topic_change_info'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(layer_size)
            state_outputs_info, final_state_info = tf.nn.dynamic_rnn(lstm_cell,inputs,sequence_length=sequence_length, dtype=tf.float32)
            final_state_info = final_state_info.h
            final_state_info = tf.expand_dims(final_state_info,-1)
            info_attn = tf.nn.softmax(tf.matmul(inputs,final_state_info))

            topic_changed_info = tf.nn.softmax(tf.math.subtract(0.5, info_attn + das_attn)) ##reverse the weight
        embedding1 = tf.get_variable('embedding1', [input_size, layer_size])
        input_change_info = inputs * topic_changed_info
        inputs = tf.nn.embedding_lookup(embedding1, input_data)
        inputs = tf.reduce_sum(inputs, 2)



        # input_data = tf.cast(input_data, dtype=tf.int32)
        # inputs = tf.nn.embedding_lookup(embedding, input_data)  # batch_size*sequence_length*8887*256 ?
        # inputs = tf.reduce_sum(inputs, 2)  # batch_size*sequence_length*256
        #
        # inputs = tf.cast(inputs, dtype=tf.float32)
        # encoder 层
        state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                                     sequence_length=sequence_length, dtype=tf.float32)
        final_state = tf.concat([final_state[0].h, final_state[1].h], 1)  # batch_size*512
        state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)  # batch_size*max_time*512

        state_shape = state_outputs.get_shape()  # batch_size*max_time*512

        with tf.variable_scope('attention'):
            da_inputs = state_outputs
            attn_size = state_shape[2].value  # 512维
            da_inputs = tf.reshape(da_inputs, [-1, attn_size])  #

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
                    next_input = pred_embedding + input_change_info[:,time,:]
                    time+=1
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
                attn = tf.transpose(decoder_final_state.alignment_history.stack(), [1, 2, 0])  # 这个部分不解

                attn = tf.reduce_mean(attn, axis=2)
                attn = tf.expand_dims(attn, -1)
                d = tf.reduce_sum(attn * state_outputs, axis=1)

                sum_output = tf.concat([d, sum_input], 1)  # 16*1024
        with tf.variable_scope('sentence_gated'):
            sum_gate = core_rnn_cell._linear(sum_output, attn_size, True)
            sum_gate = tf.reshape(sum_gate, [-1, 1, sum_gate.get_shape()[1].value])
            v1 = tf.get_variable('gateV', [attn_size])

            sentence_gate = v1 * tf.tanh(state_outputs + sum_gate)  # batch_size*?*512  此处将tanh函数改成了relu
            gate_value = tf.reduce_sum(sentence_gate, [2], name='gate_value')
            sentence_gate = tf.expand_dims(gate_value, -1)
            sentence_gate = state_outputs * sentence_gate

            sentence_gate = tf.reshape(sentence_gate, [-1, attn_size])
            da_output = tf.concat([sentence_gate, da_inputs], 1)
        with tf.variable_scope('da_proj'):
            da = core_rnn_cell._linear(da_output, da_size, True)

        # da的大小是 16*17  rnn_output的大小是16*sequence_length*8887
        outputs = [da, decoder_final_outputs.rnn_output, decoder_final_outputs.sample_id, das_attn]

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
        da_outputs = []
        correct_das = []

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
                for batch in ret[1]:
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

                # dialogue act part
                pred_das = ret[0].reshape((da_data.shape[0], da_data.shape[1], -1))
                for p, t, i, l in zip(pred_das, da_data, in_data, length):
                    p = np.argmax(p, 1)
                    tmp_pred = []
                    tmp_correct = []
                    for j in range(l):
                        tmp_pred.append(da_vocab['rev'][p[j]])
                        tmp_correct.append(da_vocab['rev'][t[j]])
                    da_outputs.append(tmp_pred)
                    correct_das.append(tmp_correct)

            if data_processor_valid.end == 1:
                break

        precision = computeAccuracy(correct_das, da_outputs)
        logging.info('da precision: ' + str(precision))
        logging.info('sum rouge1: ' + str(np.mean(rouge_1)))
        logging.info('sum rouge2: ' + str(np.mean(rouge_2)))
        logging.info('sum rouge3: ' + str(np.mean(rouge_3)))
        logging.info('sum rougeL: ' + str(np.mean(rouge_L)))

        data_processor_valid.close()
        return np.mean(rouge_1), np.mean(rouge_2), np.mean(rouge_3), np.mean(rouge_L), precision

    def train(self, grade_clip=True, model_type = 'encoder_with_dialogue_act'):
        full_train_path, full_test_path, full_valid_path, in_vocab, da_vocab = self.init_path_and_voc()
        data_processor = DataProcessor(os.path.join(full_train_path, self.input_file),
                                       os.path.join(full_train_path, self.da_file),
                                       os.path.join(full_train_path, self.sum_file), in_vocab, da_vocab)
        with tf.variable_scope('training_model'):
            training_outputs = self.train_model(self.input_data, len(in_vocab['vocab']), self.sequence_length, self.input_das,self.input_pos,
                                                len(da_vocab['vocab']), self.sum_length, model_type, self.layer_size)
        das_shape = tf.shape(self.das)
        das_reshape = tf.reshape(self.das, [-1])
        da_outputs = training_outputs[0]
        with tf.variable_scope('da_loss'):
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=das_reshape, logits=da_outputs)
            crossent = tf.reshape(crossent, das_shape)
            da_loss = tf.reduce_sum(crossent * self.da_weights, 1)
            total_size = tf.reduce_sum(self.da_weights, 1)
            total_size += 1e-12
            da_loss = da_loss / total_size


        # 这个是用来看输出的，主要是给这个输出节点命名，不知道有没有其他方法
        sum_output = tf.multiply(1.0, training_outputs[1], name='my_output')
        with tf.variable_scope('sum_loss'):
            sum_loss = tf.contrib.seq2seq.sequence_loss(logits=sum_output, targets=self.summ, weights=self.sum_weights,
                                                        average_across_timesteps=False)

        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(learning_rate=0.0001)

        sum_params = []
        da_params = []
        for p in params:
            if not 'da_' in p.name:
                sum_params.append(p)
            if 'da_' in p.name or 'bidirectional_rnn' in p.name or 'embedding' in p.name:
                da_params.append(p)
        gradients_da = tf.gradients(da_loss, da_params)
        gradients_sum = tf.gradients(sum_loss, sum_params)

        clipped_gradients_da, norm_da = tf.clip_by_global_norm(gradients_da, 5.0)
        clipped_gradients_sum, norm_sum = tf.clip_by_global_norm(gradients_sum, 5.0)

        gradient_norm_da = norm_da
        gradient_norm_sum = norm_sum
        update_da = opt.apply_gradients(zip(clipped_gradients_da, da_params))
        update_sum = opt.apply_gradients(zip(clipped_gradients_sum, sum_params), global_step=self.global_step)

        # if grade_clip == False:
        #     update_da = opt.minimize(da_loss)
        #     update_sum = opt.minimize(sum_loss)

        training_outputs = [self.global_step, da_loss, sum_loss, update_sum, update_da, gradient_norm_da,
                            gradient_norm_sum]
        inputs = [self.input_data, self.sequence_length, self.das, self.da_weights, self.summ, self.sum_weights,
                  self.sum_length]

        tf.summary.scalar('da_loss', tf.reduce_mean(da_loss))
        tf.summary.scalar('sum_loss', tf.reduce_mean(sum_loss))
        # Create Inference Model
        with tf.variable_scope('training_model',reuse=True):
            self.inference_outputs = self.train_model(self.input_data, len(in_vocab['vocab']), self.sequence_length, self.input_das,self.input_pos,
                                                      len(da_vocab['vocab']),self.sum_length, model_type, self.layer_size,
                                                      isTraining=False)

        inference_da_output = tf.nn.softmax(self.inference_outputs[0], name='da_output')
        inference_sum_output = tf.nn.softmax(self.inference_outputs[1], name='sum_output')

        self.inference_outputs = [inference_da_output, inference_sum_output]
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
            loss = 0.0
            sum_loss = 0.0
            num_loss = 0
            no_improve = 0

            valid_da = -1
            test_da = -1
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
                    loss += np.mean(ret[1])
                    sum_loss += np.mean(ret[2])
                    step = ret[0]
                    num_loss += 1
                    train_writer.add_summary(summary)

                if data_processor.end == 1:
                    data_processor.close()
                    data_processor = None
                    epochs += 1
                    logging.info('Step: ' + str(step))
                    logging.info('Epochs: ' + str(epochs))
                    logging.info('DA Loss: ' + str(loss / num_loss))
                    logging.info('Int. Loss: ' + str(sum_loss / num_loss))
                    # logging.info('das: ' + str(da_data))
                    # logging.info('das_atten_weight: ' + str(training_outputs[3]))

                    num_loss = 0
                    loss = 0.0
                    sum_loss = 0.0

                    save_path = os.path.join(self.model_path, 'summary_only')
                    save_path += '_size_' + str(self.layer_size) + '_epochs_' + str(epochs) + '.ckpt'
                    saver.save(sess, save_path)


                    logging.info('Valid:')
                    # variable starts wih e stands for current epoch
                    e_v_r1, e_v_r2, e_v_r3, e_v_rL, e_valid_da = self.valid_model(
                        os.path.join(full_valid_path, self.input_file), os.path.join(full_valid_path, self.da_file),
                        os.path.join(full_valid_path, self.sum_file), sess)
                    logging.info('Test:')
                    e_t_r1, e_t_r2, e_t_r3, e_t_rL, e_test_da = self.valid_model(
                        os.path.join(full_test_path, self.input_file), os.path.join(full_test_path, self.da_file),
                        os.path.join(full_test_path, self.sum_file), sess)

                    if e_v_r2 <= v_r2 and e_valid_da <= valid_da:
                        no_improve += 1
                    else:
                        no_improve = 0

                    if e_valid_da > valid_da:
                        valid_da = e_valid_da
                        test_da = e_test_da

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

                        builder = tf.saved_model.builder.SavedModelBuilder('./model_pb')
                        # SavedModelBuilder里面放的是你想要保存的路径，比如我的路径是根目录下的model2文件
                        builder.add_meta_graph_and_variables(sess, ["mytag"])
                        # 第二步必需要有，它是给你的模型贴上一个标签，这样再次调用的时候就可以根据标签来找。我给它起的标签名是"mytag"，你也可以起别的名字，不过你需要记住你起的名字是什么。
                        builder.save()

                    if epochs == self.max_epochs:
                        break

                    if test_da == -1 or valid_da == -1 or t_r2 == -1 or v_r2 == -1:
                        print('something in validation or testing goes wrong! did not update error.')
                        exit(1)
            train_writer.close()
            test_writer.close()
        header = self.result_path
        with open(os.path.join(header, 'valid_da_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(valid_da) + '\n')
        with open(os.path.join(header, 'test_da_' + 'summary_only' + str(self.layer_size) + '.txt'), 'a') as f:
            f.write(str(test_da) + '\n')

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
