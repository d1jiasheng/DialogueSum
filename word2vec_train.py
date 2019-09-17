from gensim.models import word2vec
import logging



def main():
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
    sentences = word2vec.LineSentence('D:\PythonProject\MultiDialSum\data\word2vectrain.txt')
    model = word2vec.Word2Vec(sentences, size=100)
    # 保存模型
    model.save("model/word2vec.model")



if __name__=="__main__":
    main()

input_data_one = tf.cast(input_data, dtype=tf.float32)
das_input = tf.cast(das_input, dtype=tf.float32)

w_alpha = 0.01
h_alpha = 0.1
w_das = tf.Variable(w_alpha * tf.random_normal([16, 256, 17]), name="w_das")
h_das = tf.Variable(h_alpha * tf.random_normal([256]), name="h_das")

das_trans_input_ = tf.reshape(das_input, [16, 17, -1])
das_trans_input_ = tf.add(tf.transpose(tf.matmul(w_das, das_trans_input_), [0, 2, 1]), h_das)
das_trans_input_t = tf.transpose(das_trans_input_, [0, 2, 1])

das_attn = tf.nn.softmax(tf.reduce_sum(tf.matmul(das_trans_input_, das_trans_input_t), axis=-1))
das_attn = tf.expand_dims(das_attn, axis=-1)

das_input = das_attn * das_input

if model_type == 'encoder_with_dialogue_act':
    input_data_one = tf.concat([input_data_one, das_input], axis=2)
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
    state_outputs_info, final_state_info = tf.nn.dynamic_rnn(lstm_cell, inputs, sequence_length=sequence_length,
                                                             dtype=tf.float32)
    final_state_info = final_state_info.h
    final_state_info = tf.expand_dims(final_state_info, -1)
    info_attn = tf.matmul(inputs, final_state_info)

    topic_changed_info = tf.nn.sigmoid(tf.math.subtract(1.0, info_attn * das_attn))

input_data = tf.cast(input_data, dtype=tf.float32)

input_data = tf.concat([input_data, das_input], axis=2)
input_data = tf.reshape(tf.expand_dims(tf.reduce_sum(input_data, -1), -1), [-1, 1])
input_data = core_rnn_cell._linear(input_data, 256, True)
inputs = tf.reshape(input_data, [16, -1, 256])
inputs = inputs * topic_changed_info
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