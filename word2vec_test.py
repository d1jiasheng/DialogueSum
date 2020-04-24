#coding = utf-8
from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load("model/word2vec.model")
# word = 'apple'
# res = model.most_similar(word,topn=10)
# print('与'+word+'最相近的几个词')
# for i in res:
#     print(i)
vec_1 = np.zeros(shape=100)
vec_2 = np.zeros(shape=100)
with open('C:\\Users\Administrator\Desktop\similar1.txt','r+') as f:
    for line in f.readlines():
        line = line.replace('.','').replace(',','').replace('\'','').replace('\"','').replace('“','')
        for word in line.split(' '):
            try:
                vec_1+=model[word]
            except:
                continue
f.close()
with open('C:\\Users\Administrator\Desktop\similar2.txt','r+') as f:
    for line in f.readlines():
        line = line.replace('.','').replace(',','').replace('\'','').replace('\"','').replace('“','')
        for word in line.split(' '):
            try:
                vec_2+=model[word]
            except:
                continue
f.close()

fen_zi = 0
fen_mu1 = 0
fen_mu2 = 0
for i in range(100):
    fen_zi+=vec_1[i]*vec_2[i]
    fen_mu1 += vec_1[i]*vec_1[i]
    fen_mu2 += vec_2[i]*vec_2[i]
cos_similar = fen_zi/((fen_mu1*fen_mu2)**0.5)
print(cos_similar)