from gensim.models import word2vec
import logging



def main():
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
    sentences = word2vec.LineSentence('D:\PythonProject\MultiDialSum\data\\test.txt')
    model = word2vec.Word2Vec(sentences, size=100, sg=0,iter=5)  #size设置训练出来的词向量大小  sg设置使用的模型 iter选择迭代训练次数
    # 保存模型
    model.save("model/test1.pkl")



if __name__=="__main__":
    main()
