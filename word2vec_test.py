from gensim.models import Word2Vec

model = Word2Vec.load("model/word2vec.model")
print(model['.'])
res = model.most_similar('what',topn=5)
for i in res:
    print(i)