<<<<<<< HEAD
import numpy as np
import os


def get_vocab_dict():
    vocab_path = './vocab/in_vocab'
    vocab = open(vocab_path, 'r+')
    vocab_dict = {}
    count = 0
    for item in vocab.readlines():
        vocab_dict[count] = item
        count += 1
    return vocab_dict

def idstosentence(ids,vocab_dict):
    str = ''
    for id in ids:
        str+=vocab_dict[id]

    str = str.replace('\n',' ').replace('_PAD','')
    return str

def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []

    with open(path) as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x,y) for (y,x) in enumerate(rev)])

    return {'vocab': vocab, 'rev': rev}

def sentenceToIds(data, vocab):
    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    vocab = vocab['vocab']
    if isinstance(data, str):
        if data.find('<EOS>') != -1: #input sequence case
            tmp = data.split('<EOS>')[:-1]
            words = []
            for i in tmp:
                words.append(i.split())
        else:
            words = data.split()
    elif isinstance(data, list):
        raise TypeError('list type data is not implement yet')
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')

    ids = []
    for w in words:
        if isinstance(w,list): #input sequence case
            sent = []
            for i in w:
                if str.isdigit(i) == True:
                    i = '0'
                sent.append(vocab.get(i, vocab['_UNK']))
            ids.append(sent)
        else:
            if str.isdigit(w) == True:
                w = '0'
            ids.append(vocab.get(w, vocab['_UNK']))
    return ids

def padSentence(s, max_length, vocab, word_in_sent_length=0):
    if isinstance(s[0],list): #input sequence case
        if len(s)>max_length:
            return s[0:max_length]
        for _ in range(max_length-len(s)):
            s.append([vocab['vocab']['_PAD']]*word_in_sent_length)
        return s
    else:
        if len(s)>max_length:
            return s[0:max_length]
        return s + [vocab['vocab']['_PAD']]*(max_length - len(s))

def computeAccuracy(correct_das, pred_das):
    correctChunkCnt = 0
    foundPredCnt = 0
    for correct_da, pred_da in zip(correct_das, pred_das):
        for c, p in zip(correct_da, pred_da):
            correctTag = c
            predTag = p
            if predTag == correctTag:
                correctChunkCnt += 1
            foundPredCnt += 1

    if foundPredCnt > 0:
        precision = 100*correctChunkCnt/foundPredCnt
    else:
        precision = 0

    return precision

def danumber2vector(da_number):
    result_vector = []
    for i in range(17):
        result_vector.append(0)
    if da_number==0:
        return result_vector
    result_vector[da_number-1] = 1
    return result_vector

def position2vector(position_number,max_len):
    result_vector = []
    for i in range(max_len):
        result_vector.append(0)
    result_vector[position_number] = 1
    return result_vector

def calculate_similar(vector1, vector2):
    fen_mu = 1e-10
    fen_zi = 0
    vec1 = 0
    vec2 = 0
    for i in range(len(vector1)):
        fen_zi += vector1[i]*vector2[i]
        vec1 += vector1[i]*vector1[i]
        vec2 += vector2[i]*vector2[i]
    fen_mu+= pow(vec1*vec2,1/2)
    return fen_zi/fen_mu

def sent2vec(sent,model):
    result_vec = np.zeros(100)
    word_list = sent.split(' ')
    for word in word_list:
        try:
            result_vec = result_vec+model[word]
        except:
            continue
    return list(result_vec)

class DataProcessor(object):
    def __init__(self, in_path, da_path, sum_path, in_vocab, da_vocab):
        self.__fd_in = open(in_path, 'r')
        self.__fd_da = open(da_path, 'r')
        self.__fd_sum = open(sum_path, 'r')
        self.__in_vocab = in_vocab
        self.__da_vocab = da_vocab
        self.end = 0

    def close(self):
        self.__fd_in.close()
        self.__fd_da.close()
        self.__fd_sum.close()

    def get_max_length(self,batch_size):

        max_len = 0
        for i in range(batch_size):
            inp = self.__fd_in.readline()
            if inp == '':
                self.end = 1
                break
            inp = inp.rstrip()

            inp = sentenceToIds(inp, self.__in_vocab)

            if len(inp) > max_len:
                max_len = len(inp)
        return max_len

    def get_batch(self, batch_size):
        in_sen_similar = []
        in_data = []
        new_in_data = []
        da_data = []
        das_input = []
        position_input = []
        da_weight = []
        length = []
        sum_data = []
        sum_weight = []
        sum_length = []

        batch_in = []
        batch_da = []
        batch_sum = []
        max_len = 0
        max_sum_len = 0
        max_word_in_sent = 0

        #used to record word(not id)
        in_seq = []
        da_seq = []
        sum_seq = []
        for i in range(batch_size):
            inp = self.__fd_in.readline()
            if inp == '':
                self.end = 1
                break
            da = self.__fd_da.readline()
            summ = self.__fd_sum.readline()
            inp = inp.rstrip()
            da = da.rstrip()
            summ = summ.rstrip()

            in_seq.append(inp)
            da_seq.append(da)
            sum_seq.append(summ)

            inp = sentenceToIds(inp, self.__in_vocab)
            da = sentenceToIds(da, self.__da_vocab)
            summ = sentenceToIds(summ, self.__in_vocab)
            batch_in.append(np.array(inp))
            batch_da.append(np.array(da))
            batch_sum.append(np.array(summ))
            length.append(len(inp))
            sum_length.append(len(summ))
            if len(inp) > max_len:
                max_len = len(inp)
            if len(summ) > max_sum_len:
                max_sum_len = len(summ)
            if len(max(inp,key=len)) > max_word_in_sent:
                max_word_in_sent = len(max(inp,key=len))

        length = np.array(length)
        sum_length = np.array(sum_length)

        # for seq_ in in_seq:
        #     new_a_ = []
        #     new_a = []
        #     sent_list = seq_.split('<EOS>')
        #
        #     sent_list.pop()
        #     for sent in sent_list:
        #         new_a.append(sent2vec(sent[0:-1],model))
        #     if max_len-len(sent_list)<0:
        #         for i in range(max_len-len(sent_list)):
        #             new_a.pop()
        #     else:
        #         for i in range(max_len - len(sent_list)):
        #             new_a.append(list(np.zeros(100)))
        #     new_a_ = np.array(new_a)
        #     new_in_data.append(new_a)
        # new_in_data = np.array(new_in_data)

        for i, s, ints in zip(batch_in, batch_da, batch_sum):
            a = []
            for sent in i:
                a.append(padSentence(list(sent), max_word_in_sent, self.__in_vocab))
            in_data.append(padSentence(list(a), max_len, self.__in_vocab, max_word_in_sent))
            da_data.append(padSentence(list(s), max_len, self.__da_vocab))
            sum_data.append(padSentence(list(ints), max_sum_len, self.__in_vocab))

        for dialogue_arr in in_data:
            a = []
            a.append(1)
            for i in range(len(dialogue_arr)-1):
                a.append(calculate_similar(dialogue_arr[i],dialogue_arr[i-1]))
            in_sen_similar.append(a)
        in_sen_similar = np.array(in_sen_similar)
        in_data = np.array(in_data)
        da_data = np.array(da_data)
        sum_data = np.array(sum_data)

        for s in da_data:
            weight = np.not_equal(s, np.zeros(s.shape))
            weight = weight.astype(np.float32)
            da_weight.append(weight)
        da_weight = np.array(da_weight)
        for i in sum_data:
            weight = np.not_equal(i, np.zeros(i.shape))
            weight = weight.astype(np.float32)
            sum_weight.append(weight)
        sum_weight = np.array(sum_weight)

        for dialogue_da in batch_da:
            dialogue_da_list = []
            position_list = []
            for sentence_da in dialogue_da:
                dialogue_da_list.append(danumber2vector(sentence_da))
            for i in range(len(dialogue_da)):
                position_list.append(position2vector(i,max_len))
            das_input.append(padSentence(list(dialogue_da_list), max_len, self.__in_vocab, 17))
            position_input.append(padSentence(list(position_list), max_len, self.__in_vocab, max_len))
        das_input = np.array(das_input)
        position_input = np.array(position_input)


        return in_data, da_data, das_input,position_input, da_weight, length, sum_data, sum_weight, sum_length, in_seq, da_seq, sum_seq, in_sen_similar

if __name__ == '__main__':
    in_path = './data/valid/in'
    da_path = './data/valid/da'
    sum_path = './data/valid/sum'
    in_vocab = loadVocabulary(os.path.join('./vocab', 'in_vocab'))
    da_vocab = loadVocabulary(os.path.join('./vocab', 'da_vocab'))
    dp = DataProcessor(in_path,da_path,sum_path,in_vocab,da_vocab)
=======
import numpy as np
import os
from gensim.models import Word2Vec


def get_vocab_dict():
    vocab_path = './vocab/in_vocab'
    vocab = open(vocab_path, 'r+')
    vocab_dict = {}
    count = 0
    for item in vocab.readlines():
        vocab_dict[count] = item
        count += 1
    return vocab_dict

def idstosentence(ids,vocab_dict):
    str = ''
    for id in ids:
        str+=vocab_dict[id]

    str = str.replace('\n',' ').replace('_PAD','')
    return str

def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []

    with open(path) as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x,y) for (y,x) in enumerate(rev)])

    return {'vocab': vocab, 'rev': rev}

def sentenceToIds(data, vocab):
    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    vocab = vocab['vocab']
    if isinstance(data, str):
        if data.find('<EOS>') != -1: #input sequence case
            tmp = data.split('<EOS>')[:-1]
            words = []
            for i in tmp:
                words.append(i.split())
        else:
            words = data.split()
    elif isinstance(data, list):
        raise TypeError('list type data is not implement yet')
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')

    ids = []
    for w in words:
        if isinstance(w,list): #input sequence case
            sent = []
            for i in w:
                if str.isdigit(i) == True:
                    i = '0'
                sent.append(vocab.get(i, vocab['_UNK']))
            ids.append(sent)
        else:
            if str.isdigit(w) == True:
                w = '0'
            ids.append(vocab.get(w, vocab['_UNK']))
    return ids

def padSentence(s, max_length, vocab, word_in_sent_length=0):
    if isinstance(s[0],list): #input sequence case
        if len(s)>max_length:
            return s[0:max_length]
        for _ in range(max_length-len(s)):
            s.append([vocab['vocab']['_PAD']]*word_in_sent_length)
        return s
    else:
        if len(s)>max_length:
            return s[0:max_length]
        return s + [vocab['vocab']['_PAD']]*(max_length - len(s))

def computeAccuracy(correct_das, pred_das):
    correctChunkCnt = 0
    foundPredCnt = 0
    for correct_da, pred_da in zip(correct_das, pred_das):
        for c, p in zip(correct_da, pred_da):
            correctTag = c
            predTag = p
            if predTag == correctTag:
                correctChunkCnt += 1
            foundPredCnt += 1

    if foundPredCnt > 0:
        precision = 100*correctChunkCnt/foundPredCnt
    else:
        precision = 0

    return precision

def danumber2vector(da_number):
    result_vector = []
    for i in range(17):
        result_vector.append(0)
    if da_number==0:
        return result_vector
    result_vector[da_number-1] = 1
    return result_vector

def position2vector(position_number,max_len):
    result_vector = []
    for i in range(max_len):
        result_vector.append(0)
    result_vector[position_number] = 1
    return result_vector

def calculate_similar(vector1, vector2):
    fen_mu = 1e-10
    fen_zi = 0
    vec1 = 0
    vec2 = 0
    for i in range(len(vector1)):
        fen_zi += vector1[i]*vector2[i]
        vec1 += vector1[i]*vector1[i]
        vec2 += vector2[i]*vector2[i]
    fen_mu+= pow(vec1*vec2,1/2)
    return fen_zi/fen_mu

def sent2vec(sent,model):
    result_vec = np.zeros(100)
    word_list = sent.split(' ')
    for word in word_list:
        try:
            result_vec = result_vec+model[word]
        except:
            continue
    return list(result_vec)

class DataProcessor(object):
    def __init__(self, in_path, da_path, sum_path, in_vocab, da_vocab):
        self.__fd_in = open(in_path, 'r')
        self.__fd_da = open(da_path, 'r')
        self.__fd_sum = open(sum_path, 'r')
        self.__in_vocab = in_vocab
        self.__da_vocab = da_vocab
        self.end = 0

    def close(self):
        self.__fd_in.close()
        self.__fd_da.close()
        self.__fd_sum.close()

    def get_max_length(self,batch_size):

        max_len = 0
        for i in range(batch_size):
            inp = self.__fd_in.readline()
            if inp == '':
                self.end = 1
                break
            inp = inp.rstrip()

            inp = sentenceToIds(inp, self.__in_vocab)

            if len(inp) > max_len:
                max_len = len(inp)
        return max_len

    def get_batch(self, batch_size):
        in_sen_similar = []
        in_data = []
        new_in_data = []
        da_data = []
        das_input = []
        position_input = []
        da_weight = []
        length = []
        sum_data = []
        sum_weight = []
        sum_length = []

        batch_in = []
        batch_da = []
        batch_sum = []
        max_len = 0
        max_sum_len = 0
        max_word_in_sent = 0

        #used to record word(not id)
        in_seq = []
        da_seq = []
        sum_seq = []
        for i in range(batch_size):
            inp = self.__fd_in.readline()
            if inp == '':
                self.end = 1
                break
            da = self.__fd_da.readline()
            summ = self.__fd_sum.readline()
            inp = inp.rstrip()
            da = da.rstrip()
            summ = summ.rstrip()

            in_seq.append(inp)
            da_seq.append(da)
            sum_seq.append(summ)

            inp = sentenceToIds(inp, self.__in_vocab)
            da = sentenceToIds(da, self.__da_vocab)
            summ = sentenceToIds(summ, self.__in_vocab)
            batch_in.append(np.array(inp))
            batch_da.append(np.array(da))
            batch_sum.append(np.array(summ))
            length.append(len(inp))
            sum_length.append(len(summ))
            if len(inp) > max_len:
                max_len = len(inp)
            if len(summ) > max_sum_len:
                max_sum_len = len(summ)
            if len(max(inp,key=len)) > max_word_in_sent:
                max_word_in_sent = len(max(inp,key=len))

        length = np.array(length)
        sum_length = np.array(sum_length)

        # for seq_ in in_seq:
        #     new_a_ = []
        #     new_a = []
        #     sent_list = seq_.split('<EOS>')
        #
        #     sent_list.pop()
        #     for sent in sent_list:
        #         new_a.append(sent2vec(sent[0:-1],model))
        #     if max_len-len(sent_list)<0:
        #         for i in range(max_len-len(sent_list)):
        #             new_a.pop()
        #     else:
        #         for i in range(max_len - len(sent_list)):
        #             new_a.append(list(np.zeros(100)))
        #     new_a_ = np.array(new_a)
        #     new_in_data.append(new_a)
        # new_in_data = np.array(new_in_data)

        for i, s, ints in zip(batch_in, batch_da, batch_sum):
            a = []
            for sent in i:
                a.append(padSentence(list(sent), max_word_in_sent, self.__in_vocab))
            in_data.append(padSentence(list(a), max_len, self.__in_vocab, max_word_in_sent))
            da_data.append(padSentence(list(s), max_len, self.__da_vocab))
            sum_data.append(padSentence(list(ints), max_sum_len, self.__in_vocab))

        for dialogue_arr in in_data:
            a = []
            a.append(1)
            for i in range(len(dialogue_arr)-1):
                a.append(calculate_similar(dialogue_arr[i],dialogue_arr[i-1]))
            in_sen_similar.append(a)
        in_sen_similar = np.array(in_sen_similar)
        in_data = np.array(in_data)
        da_data = np.array(da_data)
        sum_data = np.array(sum_data)

        for s in da_data:
            weight = np.not_equal(s, np.zeros(s.shape))
            weight = weight.astype(np.float32)
            da_weight.append(weight)
        da_weight = np.array(da_weight)
        for i in sum_data:
            weight = np.not_equal(i, np.zeros(i.shape))
            weight = weight.astype(np.float32)
            sum_weight.append(weight)
        sum_weight = np.array(sum_weight)

        for dialogue_da in batch_da:
            dialogue_da_list = []
            position_list = []
            for sentence_da in dialogue_da:
                dialogue_da_list.append(danumber2vector(sentence_da))
            for i in range(len(dialogue_da)):
                position_list.append(position2vector(i,max_len))
            das_input.append(padSentence(list(dialogue_da_list), max_len, self.__in_vocab, 17))
            position_input.append(padSentence(list(position_list), max_len, self.__in_vocab, max_len))
        das_input = np.array(das_input)
        position_input = np.array(position_input)


        return in_data, da_data, das_input,position_input, da_weight, length, sum_data, sum_weight, sum_length, in_seq, da_seq, sum_seq, in_sen_similar

if __name__ == '__main__':
    in_path = './data/valid/in'
    da_path = './data/valid/da'
    sum_path = './data/valid/sum'
    in_vocab = loadVocabulary(os.path.join('./vocab', 'in_vocab'))
    da_vocab = loadVocabulary(os.path.join('./vocab', 'da_vocab'))
    dp = DataProcessor(in_path,da_path,sum_path,in_vocab,da_vocab)
>>>>>>> cfe53d38eaadf61745eca7efc701413e8cc538d6
    dp.get_batch(16)